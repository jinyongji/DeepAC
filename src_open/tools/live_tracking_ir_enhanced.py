"""
增强版IR实时追踪程序
整合梯度直方图、特征点检测、轮廓线分布三个模块
专门针对RealSense IR相机和魔方追踪优化
"""
import argparse
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src_open.dataset.utils import numpy_image_to_torch, crop, resize, zero_pad
from src_open.models import get_model
from src_open.models.deep_ac import calculate_basic_line_data
from src_open.utils.geometry.wrappers import Camera, Pose
from src_open.utils.lightening_utils import (
    MyLightningLogger,
    convert_old_model,
    load_model_weight,
)
from src_open.utils.utils import (
    get_bbox_from_p2d,
    get_closest_template_view_index,
    project_correspondences_line,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced DeepAC IR tracking runner")
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Path to live tracking yaml config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device, e.g. cuda:0 / cpu",
    )
    return parser.parse_args()


def load_model(cfg, device, logger):
    """加载增强的DeepAC模型"""
    train_cfg = OmegaConf.load(cfg.model.load_cfg)
    model_cfg = train_cfg.models if "models" in train_cfg else train_cfg
    
    # 使用增强的DeepAC模型
    model_name = cfg.model.get("model_name", "DeepACIREnhanced")
    if model_name == "DeepACIREnhanced":
        from src_open.models.deep_ac_ir_enhanced import DeepACIREnhanced
        model = DeepACIREnhanced(model_cfg)
    else:
        model = get_model(model_cfg.name)(model_cfg)
    
    ckpt = torch.load(cfg.model.load_model, map_location="cpu")
    if "pytorch-lightning_version" not in ckpt:
        ckpt = convert_old_model(ckpt)
    load_model_weight(model, ckpt, logger)
    model.to(device).eval()
    
    logger.info(f"Loaded enhanced IR tracking model from {cfg.model.load_model}")
    return model, train_cfg


def load_template(pre_render_path, device):
    """加载预渲染模板"""
    with open(pre_render_path, "rb") as f:
        pre = pickle.load(f)
    
    head = pre.get("head", {})
    num_sample = head.get("num_sample_contour_point") or head.get("num_sample")
    if num_sample is None:
        raise RuntimeError("pre_render pkl missing `num_sample_contour_point` in head")
    
    template_view = pre.get("template_view", None)
    if template_view is None:
        template_view = pre.get("template_views", None)
    if template_view is None:
        raise RuntimeError("pre_render pkl missing `template_view(s)` field")
    
    template_view = np.array(template_view)
    if template_view.ndim == 2:
        if template_view.shape[1] != 8:
            raise RuntimeError("Unexpected template_view shape; expected (*, 8)")
        if template_view.shape[0] % num_sample != 0:
            raise RuntimeError("template_view rows must be multiple of num_sample")
        num_views = template_view.shape[0] // num_sample
        template_view = template_view.reshape(num_views, num_sample, 8)
    
    orientations = pre.get("orientation_in_body", None)
    if orientations is None:
        orientations = pre.get("orientations_in_body", None)
    if orientations is None:
        raise RuntimeError("pre_render pkl missing `orientation_in_body` field")
    orientations = torch.from_numpy(np.array(orientations)).float()
    
    template_views = torch.from_numpy(template_view).float()
    return template_views.to(device), orientations.to(device), num_sample


def build_camera_tensor(camera_cfg, frame_shape, device):
    """构建相机张量"""
    height, width = frame_shape[:2]
    fx = float(camera_cfg.fx)
    fy = float(camera_cfg.fy)
    cx = float(camera_cfg.cx)
    cy = float(camera_cfg.cy)
    
    intrinsic_param = torch.tensor(
        [width, height, fx, fy, cx, cy],
        dtype=torch.float32,
        device=device,
    )
    return intrinsic_param


def preprocess_image(img, bbox2d, camera_cpu, data_conf, device):
    """预处理图像"""
    if isinstance(bbox2d, torch.Tensor):
        bbox2d = bbox2d.detach().cpu().numpy()
    elif not isinstance(bbox2d, np.ndarray):
        bbox2d = np.array(bbox2d)
    
    is_grayscale_model = getattr(data_conf, 'grayscale', False)
    if is_grayscale_model and len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif is_grayscale_model and len(img.shape) == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    
    bbox2d = bbox2d.copy()
    bbox2d[2:] += data_conf.crop_border * 2
    img_crop, camera_out, _ = crop(img, bbox2d, camera=camera_cpu, return_bbox=True)
    
    if img_crop.size == 0:
        img_crop = img
        camera_out = camera_cpu
    
    if isinstance(data_conf.resize, int) and data_conf.resize > 0:
        img_crop, scales = resize(
            img_crop, data_conf.resize, 
            fn=max if data_conf.resize_by == "max" else min
        )
        camera_out = camera_out.scale(scales)
    
    if isinstance(data_conf.pad, int) and data_conf.pad > 0:
        img_crop, = zero_pad(data_conf.pad, img_crop)
    
    img_tensor = numpy_image_to_torch(img_crop.astype(np.float32))
    _, h_t, w_t = img_tensor.shape
    pad_h = (32 - h_t % 32) % 32
    pad_w = (32 - w_t % 32) % 32
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h))
        cam_data = camera_out._data.clone()
        cam_data[..., 0] += pad_w
        cam_data[..., 1] += pad_h
        camera_out = Camera(cam_data)
    
    return img_tensor.to(device), camera_out.to(device)


def preprocess_frame(frame_bgr, camera_cfg, device, grayscale=False):
    """预处理帧"""
    if grayscale:
        if len(frame_bgr.shape) == 2:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
        elif frame_bgr.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = frame_bgr
    else:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    height, width = frame_rgb.shape[:2]
    fx = float(camera_cfg.fx)
    fy = float(camera_cfg.fy)
    cx = float(camera_cfg.cx)
    cy = float(camera_cfg.cy)
    
    intrinsic_param = torch.tensor(
        [width, height, fx, fy, cx, cy],
        dtype=torch.float32,
        device=device,
    )
    ori_camera = Camera(intrinsic_param)
    
    return frame_rgb, ori_camera


def draw_overlay(image, bbox, centers, valid, color=(0, 255, 0), input_is_bgr=True):
    """绘制叠加层"""
    overlay = image.copy()
    if input_is_bgr:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
    
    if centers is not None and valid is not None:
        centers_np = centers[0][valid[0]].detach().cpu().numpy()
        for pt in centers_np:
            cv2.circle(overlay, (int(pt[0]), int(pt[1])), 2, color, -1)
    
    return overlay


def initialize_realsense_ir(cfg, logger):
    """初始化RealSense IR相机"""
    try:
        import pyrealsense2 as rs
        
        pipeline = rs.pipeline()
        config = rs.config()
        
        ir_index = cfg.camera.get("ir_index", 1)
        width = cfg.camera.get("set_width", 1280)
        height = cfg.camera.get("set_height", 720)
        
        config.enable_stream(rs.stream.infrared, ir_index, width, height, rs.format.y8, 30)
        
        profile = pipeline.start(config)
        
        # 禁用结构光发射器
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
            logger.info("RealSense emitter disabled")
        
        # 获取内参
        ir_stream = profile.get_stream(rs.stream.infrared, ir_index)
        intrinsics = ir_stream.as_video_stream_profile().get_intrinsics()
        
        logger.info(
            f"RealSense IR camera (index {ir_index}) initialized: "
            f"{width}x{height}, fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}"
        )
        
        return pipeline, intrinsics
    except ImportError:
        logger.error("pyrealsense2 not installed. Install with: pip install pyrealsense2")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize RealSense IR camera: {e}")
        raise


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger = MyLightningLogger("enhanced_ir_tracking")
    
    # 加载模型
    model, train_cfg = load_model(cfg, device, logger)
    data_conf = train_cfg.data
    
    # 加载模板
    template_views, orientations, num_sample = load_template(
        cfg.object.pre_render_pkl, device
    )
    
    # 初始化RealSense IR相机
    realsense_pipeline, intrinsics = initialize_realsense_ir(cfg, logger)
    
    # 更新相机内参
    cfg.camera.fx = intrinsics.fx
    cfg.camera.fy = intrinsics.fy
    cfg.camera.cx = intrinsics.ppx
    cfg.camera.cy = intrinsics.ppy
    
    grayscale = cfg.tracking.get("grayscale", True)
    initialized = False
    current_pose = None
    fore_hist = None
    back_hist = None
    last_bbox = None
    last_centers = None
    last_valid = None
    
    # 初始位姿
    init_depth = cfg.tracking.get("init_depth", 0.45)
    initial_view_index = cfg.tracking.get("initial_view_index", 0)
    initial_pose = Pose.from_aa(
        orientations[initial_view_index:initial_view_index+1],
        torch.tensor([[0.0, 0.0, init_depth]], device=device)
    )
    initial_template = template_views[initial_view_index:initial_view_index+1]
    
    logger.info("Enhanced IR tracking started. Press SPACE to start tracking after alignment.")
    
    frame_idx = 0
    while True:
        try:
            frames = realsense_pipeline.wait_for_frames(timeout_ms=5000)
            ir_index = cfg.camera.get("ir_index", 1)
            ir_frame = frames.get_infrared_frame(ir_index)
            
            if not ir_frame:
                continue
            
            frame_bgr = np.asanyarray(ir_frame.get_data())
            if len(frame_bgr.shape) == 2:
                frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
            
            frame_rgb, ori_camera_cpu = preprocess_frame(
                frame_bgr, cfg.camera, device, grayscale=grayscale
            )
            ori_camera = ori_camera_cpu.to(device)
            
            if not initialized:
                # 显示引导框
                _, _, guide_centers, guide_valid, _, _, _, _ = \
                    calculate_basic_line_data(
                        initial_template, initial_pose._data, ori_camera._data, 1, 0
                    )
                guide_bbox = None
                if guide_valid[0].any():
                    guide_bbox = get_bbox_from_p2d(
                        guide_centers[0][guide_valid[0]], trim_ratio=0.0
                    )
                
                overlay = draw_overlay(
                    frame_rgb, guide_bbox, guide_centers, guide_valid, color=(0, 255, 255)
                )
                
                cv2.putText(
                    overlay,
                    "Align cube with guide, then press SPACE",
                    (16, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                )
            else:
                # 追踪阶段
                img_tensor, camera_tensor = preprocess_image(
                    frame_rgb, last_bbox, ori_camera_cpu, data_conf, device
                )
                
                # 准备数据
                data = {
                    'image': img_tensor.unsqueeze(0),
                    'camera': camera_tensor.unsqueeze(0),
                    'body2view_pose': current_pose.unsqueeze(0),
                    'closest_orientations_in_body': orientations.unsqueeze(0),
                    'closest_template_views': template_views.unsqueeze(0),
                    'fore_hist': fore_hist,
                    'back_hist': back_hist,
                }
                
                # 运行追踪
                with torch.no_grad():
                    result = model._forward(data, tracking=True)
                    new_pose = result['opt_body2view_pose'][-1]
                    
                    # 注意：直方图更新应该在tracking_step中完成
                    # 这里保持使用传入的fore_hist和back_hist
                
                current_pose = Pose(new_pose._data[0])
                
                # 计算投影
                index = get_closest_template_view_index(
                    current_pose.unsqueeze(0), orientations.unsqueeze(0)
                )
                template_view = template_views[index[0]:index[0]+1]
                
                _, _, centers_in_image, centers_valid, _, _, _, _ = \
                    calculate_basic_line_data(
                        template_view, current_pose._data.unsqueeze(0),
                        ori_camera._data.unsqueeze(0), 1, 0
                    )
                
                last_centers = centers_in_image
                last_valid = centers_valid
                if centers_valid[0].any():
                    last_bbox = get_bbox_from_p2d(
                        centers_in_image[0][centers_valid[0]], trim_ratio=0.0
                    )
                
                overlay = draw_overlay(
                    frame_rgb, last_bbox, last_centers, last_valid, color=(0, 255, 0)
                )
                
                cv2.putText(
                    overlay,
                    "Tracking...",
                    (16, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
            
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imshow("Enhanced IR Tracking", overlay_bgr)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and not initialized:
                # 开始追踪
                current_pose = initial_pose
                
                img_tensor, camera_tensor = preprocess_image(
                    frame_rgb, guide_bbox, ori_camera_cpu, data_conf, device
                )
                
                data = {
                    'image': img_tensor.unsqueeze(0),
                    'camera': camera_tensor.unsqueeze(0),
                    'body2view_pose': current_pose.unsqueeze(0),
                    'closest_orientations_in_body': orientations.unsqueeze(0),
                    'closest_template_views': template_views.unsqueeze(0),
                    'gt_body2view_pose': current_pose.unsqueeze(0),
                }
                
                with torch.no_grad():
                    fore_hist, back_hist = model.init_histogram(data)
                
                initialized = True
                logger.info("Tracking started!")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            break
    
    realsense_pipeline.stop()
    cv2.destroyAllWindows()
    logger.info("Enhanced IR tracking stopped")


if __name__ == "__main__":
    main()

