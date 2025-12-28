"""
DeepAC实时追踪 - 带模型对齐功能
参考M3T的实现方式：
1. 初始化阶段：持续显示模型轮廓（蓝色引导框），让用户手动对齐魔方
2. 对齐后按SPACE键开始追踪
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

# 导入必要的模块
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
    get_closest_k_template_view_index,
    project_correspondences_line,
)

# 直接导入live_tracking模块的函数（使用绝对导入）
import importlib.util
import sys
from pathlib import Path

# 获取live_tracking.py的路径
live_tracking_path = Path(__file__).parent / "live_tracking.py"
spec = importlib.util.spec_from_file_location("live_tracking", live_tracking_path)
live_tracking_module = importlib.util.module_from_spec(spec)
sys.modules["live_tracking"] = live_tracking_module
spec.loader.exec_module(live_tracking_module)

# 从模块中提取需要的函数
load_template = live_tracking_module.load_template
prepare_initial_pose = live_tracking_module.prepare_initial_pose
build_camera_tensor = live_tracking_module.build_camera_tensor
preprocess_frame = live_tracking_module.preprocess_frame
preprocess_image = live_tracking_module.preprocess_image
initialize_from_pose_with_preprocess = live_tracking_module.initialize_from_pose_with_preprocess
tracking_step_with_preprocess = live_tracking_module.tracking_step_with_preprocess
draw_overlay = live_tracking_module.draw_overlay
load_model = live_tracking_module.load_model
smooth_pose = live_tracking_module.smooth_pose
detect_once_with_preprocess = live_tracking_module.detect_once_with_preprocess

def parse_args():
    parser = argparse.ArgumentParser(description="DeepAC live tracking with model alignment")
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
    parser.add_argument(
        "--camera_id",
        type=int,
        default=None,
        help="Override camera ID",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    logger = MyLightningLogger("deepac-live-align", str(save_dir))
    logger.dump_cfg(cfg, "live_cfg.yaml")

    model, train_cfg = load_model(cfg, device, logger)
    logger.dump_cfg(train_cfg, "train_cfg.yaml")

    # 同步数据配置
    data_conf = train_cfg.data
    data_conf.crop_border = cfg.tracking.get("crop_border", 10)
    data_conf.resize = cfg.tracking.get("resize", 320)
    data_conf.pad = cfg.tracking.get("pad", 0)
    data_conf.resize_by = cfg.tracking.get("resize_by", "max")
    data_conf.grayscale = cfg.tracking.get("grayscale", False)
    template_top_k = cfg.tracking.get("template_top_k", 10)
    template_skip = cfg.tracking.get("template_skip", 1)
    if template_top_k is not None and hasattr(data_conf, "get_top_k_template_views"):
        data_conf.get_top_k_template_views = max(data_conf.get_top_k_template_views, template_top_k)
    if template_skip is not None and hasattr(data_conf, "skip_template_view"):
        data_conf.skip_template_view = template_skip

    template_views, orientations, _ = load_template(cfg.object.pre_render_pkl, device)
    initial_view_idx, initial_pose, initial_template = prepare_initial_pose(
        cfg, template_views, orientations, device
    )
    logger.info(f"Guide overlay uses template view index {initial_view_idx}")

    # 初始化RealSense RGB相机（如果可用）
    use_realsense_rgb = False
    realsense_rgb_pipeline = None
    cap = None
    
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) > 0:
            logger.info(f"Found RealSense device: {devices[0].get_info(rs.camera_info.name)}")
            logger.info("Attempting to use RealSense RGB stream directly")
            
            pipeline = rs.pipeline()
            config = rs.config()
            
            requested_width = int(cfg.camera.set_width)
            requested_height = int(cfg.camera.set_height)
            
            rs_device = devices[0]
            sensors = rs_device.query_sensors()
            rgb_sensor = None
            for sensor in sensors:
                if sensor.get_info(rs.camera_info.name) == "RGB Camera":
                    rgb_sensor = sensor
                    break
            
            if rgb_sensor:
                profiles = rgb_sensor.get_stream_profiles()
                rgb_profiles = [p for p in profiles if p.stream_type() == rs.stream.color]
                
                matching_profile = None
                for profile in rgb_profiles:
                    vp = profile.as_video_stream_profile()
                    if vp.width() == requested_width and vp.height() == requested_height:
                        matching_profile = profile
                        break
                
                if matching_profile is None and len(rgb_profiles) > 0:
                    matching_profile = rgb_profiles[0]
                    vp = matching_profile.as_video_stream_profile()
                    requested_width = vp.width()
                    requested_height = vp.height()
                    logger.info(f"Using available RGB resolution: {requested_width}x{requested_height}@{vp.fps()}fps")
                
                if matching_profile:
                    vp = matching_profile.as_video_stream_profile()
                    config.enable_stream(rs.stream.color, vp.width(), vp.height(), rs.format.rgb8, vp.fps())
                    rs_pipeline_profile = pipeline.start(config)
                    
                    color_stream = rs_pipeline_profile.get_stream(rs.stream.color)
                    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
                    
                    cfg.camera.fx = intrinsics.fx
                    cfg.camera.fy = intrinsics.fy
                    cfg.camera.cx = intrinsics.ppx
                    cfg.camera.cy = intrinsics.ppy
                    cfg.camera.set_width = intrinsics.width
                    cfg.camera.set_height = intrinsics.height
                    
                    use_realsense_rgb = True
                    realsense_rgb_pipeline = pipeline
                    
                    logger.info(f"Successfully initialized RealSense RGB stream:")
                    logger.info(f"  Resolution: {intrinsics.width}x{intrinsics.height}")
                    logger.info(f"  Intrinsics: fx={intrinsics.fx:.6f}, fy={intrinsics.fy:.6f}, "
                               f"cx={intrinsics.ppx:.6f}, cy={intrinsics.ppy:.6f}")
    except ImportError:
        logger.info("pyrealsense2 not available, will use OpenCV")
    except Exception as e:
        logger.info(f"Could not initialize RealSense RGB stream: {e}")
        logger.info("Falling back to OpenCV camera")
    
    if not use_realsense_rgb:
        camera_id = args.camera_id if args.camera_id is not None else int(cfg.camera.get("camera_id", 0))
        logger.info(f"Using camera ID: {camera_id}")
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.camera.set_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.camera.set_height))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open camera id={camera_id}")
        
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info(f"Camera opened (id={camera_id}) at {actual_width}x{actual_height}")

    grayscale = bool(cfg.tracking.get("grayscale", False))
    bbox_trim_ratio = float(cfg.tracking.get("bbox_trim_ratio", 0.08))
    smooth_alpha = float(cfg.tracking.get("smooth_alpha", 0.5))
    use_smoothing = smooth_alpha > 0.0
    
    # 第一阶段：模型对齐阶段
    logger.info("=" * 60)
    logger.info("PHASE 1: Model Alignment")
    logger.info("=" * 60)
    logger.info("A reference model (blue wireframe) is being rendered.")
    logger.info("Please align your cube with the rendered model.")
    logger.info("Press SPACE to start tracking after alignment.")
    logger.info("Press ESC or 'q' to exit.")
    logger.info("=" * 60)
    
    alignment_mode = True
    frame_idx = 0
    
    while alignment_mode:
        # 读取帧
        if use_realsense_rgb and realsense_rgb_pipeline is not None:
            try:
                frames = realsense_rgb_pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                frame_rgb_array = np.asanyarray(color_frame.get_data())
                frame_bgr = cv2.cvtColor(frame_rgb_array, cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.info(f"RealSense RGB frame error: {e}")
                continue
        else:
            ret, frame_bgr = cap.read()
            if not ret:
                break
        
        if grayscale and len(frame_bgr.shape) == 2:
            pass
        elif grayscale:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        frame_rgb, ori_camera_cpu = preprocess_frame(frame_bgr, cfg.camera, device, grayscale=grayscale)
        ori_camera = ori_camera_cpu.to(device)
        
        # 渲染模型轮廓（蓝色引导框）
        (
            _,
            _,
            guide_centers,
            guide_valid,
            _,
            _,
            _,
            _,
        ) = calculate_basic_line_data(initial_template, initial_pose._data, ori_camera._data, 1, 0)
        
        guide_bbox = None
        if guide_valid[0].any():
            guide_bbox = get_bbox_from_p2d(
                guide_centers[0][guide_valid[0]], trim_ratio=bbox_trim_ratio
            )
        
        # 绘制模型轮廓（蓝色，半透明）
        overlay = draw_overlay(frame_rgb, guide_bbox, guide_centers, guide_valid, color=(255, 128, 0))
        
        # 添加提示文字
        cv2.putText(
            overlay,
            "Align cube with blue model...",
            (16, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 200, 255),
            2,
        )
        cv2.putText(
            overlay,
            "Press SPACE to start tracking",
            (16, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            overlay,
            "Press ESC to exit",
            (16, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        
        cv2.imshow("DeepAC Live - Alignment Phase", overlay)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") or key == ord("s"):
            alignment_mode = False
            logger.info("Alignment phase complete. Starting tracking...")
        elif key == 27 or key == ord("q"):
            logger.info("Exiting...")
            return
        
        frame_idx += 1
    
    # 第二阶段：追踪阶段
    logger.info("=" * 60)
    logger.info("PHASE 2: Tracking")
    logger.info("=" * 60)
    
    # 关键修复：在对齐后自动检测最佳初始位姿，而不是使用固定的initial_pose
    logger.info("Detecting initial pose after alignment...")
    
    # 使用自动检测找到最佳初始位姿
    detect_result = detect_once_with_preprocess(
        frame_rgb,
        ori_camera_cpu,
        initial_pose,  # 使用initial_pose作为参考
        template_views,
        orientations,
        model,
        device,
        cfg.tracking,
        data_conf,
        bbox_trim_ratio=bbox_trim_ratio,
    )
    
    if detect_result is not None:
        current_pose = detect_result["pose"]
        logger.info(f"Detection successful: view_index={detect_result['idx']}, score={detect_result['score']:.6f}")
        pose_t = current_pose.t
        if pose_t.ndim > 1:
            pose_t = pose_t[0]
        logger.info(f"Detected pose: t=({pose_t[0]:.3f}, {pose_t[1]:.3f}, {pose_t[2]:.3f})m")
    else:
        logger.info("Detection failed, using initial pose")
        current_pose = Pose(initial_pose._data.clone()).to(device)
    
    # 初始化直方图（使用检测到的位姿）
    (
        fore_hist,
        back_hist,
        last_bbox,
        last_centers,
        last_valid,
    ) = initialize_from_pose_with_preprocess(
        frame_rgb,
        ori_camera_cpu,
        current_pose,
        template_views,
        orientations,
        model,
        device,
        data_conf,
        bbox_trim_ratio=bbox_trim_ratio,
    )
    
    initialized = True
    frame_idx = 0
    pose_smooth = Pose(current_pose._data.clone()) if use_smoothing else None
    
    logger.info("Tracking started. Press ESC or 'q' to stop.")
    
    while True:
        # 读取帧
        if use_realsense_rgb and realsense_rgb_pipeline is not None:
            try:
                frames = realsense_rgb_pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                frame_rgb_array = np.asanyarray(color_frame.get_data())
                frame_bgr = cv2.cvtColor(frame_rgb_array, cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.info(f"RealSense RGB frame error: {e}")
                continue
        else:
            ret, frame_bgr = cap.read()
            if not ret:
                break
        
        if grayscale and len(frame_bgr.shape) == 2:
            pass
        elif grayscale:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        frame_rgb, ori_camera_cpu = preprocess_frame(frame_bgr, cfg.camera, device, grayscale=grayscale)
        
        # 追踪步骤
        (
            new_pose,
            fore_hist,
            back_hist,
            last_bbox,
            last_centers,
            last_valid,
        ) = tracking_step_with_preprocess(
            frame_rgb,
            ori_camera_cpu,
            current_pose,
            template_views,
            orientations,
            model,
            fore_hist,
            back_hist,
            cfg.tracking,
            data_conf,
            device,
            bbox_trim_ratio=bbox_trim_ratio,
        )
        
        # 应用pose平滑
        if use_smoothing:
            pose_smooth = smooth_pose(pose_smooth, new_pose, smooth_alpha)
            current_pose = pose_smooth
        else:
            current_pose = new_pose.detach() if hasattr(new_pose, 'detach') else new_pose
        
        # 绘制追踪结果（绿色）
        overlay = draw_overlay(
            frame_rgb,
            last_bbox,
            last_centers,
            last_valid,
            color=(0, 255, 0),
        )
        
        cv2.putText(
            overlay,
            f"Tracking... Frame {frame_idx}",
            (16, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        
        cv2.imshow("DeepAC Live - Tracking Phase", overlay)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            logger.info("Stopping tracking...")
            break
        
        frame_idx += 1
    
    # 清理
    if cap is not None:
        cap.release()
    if realsense_rgb_pipeline is not None:
        try:
            realsense_rgb_pipeline.stop()
        except:
            pass
    cv2.destroyAllWindows()
    logger.info("Live tracking finished")

if __name__ == "__main__":
    main()

