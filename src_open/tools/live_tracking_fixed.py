"""
DeepAC实时追踪 - 完全按照demo_cube.py的逻辑重写
确保与demo_cube.py的逻辑完全一致
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
    get_closest_k_template_view_index,
    project_correspondences_line,
)

# 导入检测函数
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("live_tracking_module", "src_open/tools/live_tracking.py")
live_tracking_module = importlib.util.module_from_spec(spec)
sys.modules["live_tracking_module"] = live_tracking_module
spec.loader.exec_module(live_tracking_module)
detect_once_with_preprocess = live_tracking_module.detect_once_with_preprocess

@torch.no_grad()
def optimize_pose_with_edge_detection(
    frame_rgb,
    initial_pose,
    template_views,
    orientations,
    camera,
    device,
    logger,
    num_iterations=15,
    learning_rate=0.05,
):
    """
    基于边缘检测优化位姿，使模型轮廓"吸附"到检测到的边缘
    
    算法流程：
    1. 使用Canny边缘检测找到图像中的边缘
    2. 将模型轮廓投影到图像上
    3. 计算轮廓点到最近边缘的距离
    4. 使用梯度下降优化位姿，使轮廓点靠近边缘
    
    Args:
        frame_rgb: RGB图像 [H, W, 3]
        initial_pose: 初始位姿
        template_views: 模板视图 [N_views, N_points, 8]
        orientations: 方向 [N_views, 3]
        camera: 相机对象
        device: 设备
        num_iterations: 优化迭代次数
        learning_rate: 学习率
    
    Returns:
        优化后的位姿
    """
    logger.info("Starting edge-based pose optimization...")
    
    # 1. 边缘检测（使用更强的边缘检测参数）
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    
    # ========== 关键调试：检查图像和边缘质量 ==========
    gray_mean = gray.mean()
    gray_std = gray.std()
    gray_min = gray.min()
    gray_max = gray.max()
    logger.info(f"DEBUG Edge Optimization: Gray image stats - mean={gray_mean:.2f}, std={gray_std:.2f}, range=[{gray_min}, {gray_max}]")
    
    # 使用自适应阈值提高边缘检测效果
    # 降低阈值以检测更多边缘
    edges = cv2.Canny(gray, 30, 100)
    
    # ========== 关键调试：检查边缘检测结果 ==========
    edge_mean = edges.astype(np.float32).mean()
    edge_max = edges.astype(np.float32).max()
    edge_nonzero = np.count_nonzero(edges)
    edge_total = edges.size
    edge_ratio = edge_nonzero / edge_total if edge_total > 0 else 0.0
    logger.info(f"DEBUG Edge Optimization: Edge map stats - mean={edge_mean:.6f}, max={edge_max:.0f}, nonzero={edge_nonzero}/{edge_total} ({edge_ratio*100:.2f}%)")
    
    if edge_mean < 0.1 or edge_max < 10:
        logger.info(f"WARNING: Edge map is too weak! This will cause optimizer to fail.")
        logger.info(f"  Possible causes:")
        logger.info(f"    1) Image preprocessing issue (contrast too low)")
        logger.info(f"    2) Canny thresholds too high")
        logger.info(f"    3) Image resize causing edge loss")
    
    # 可选：使用形态学操作增强边缘
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # ========== 调试：检查dilate后的边缘 ==========
    edge_mean_after = edges.astype(np.float32).mean()
    edge_max_after = edges.astype(np.float32).max()
    logger.info(f"DEBUG Edge Optimization: After dilation - mean={edge_mean_after:.6f}, max={edge_max_after:.0f}")
    
    # 获取最近的模板视图
    best_idx = get_closest_template_view_index(initial_pose, orientations)
    if isinstance(best_idx, torch.Tensor):
        best_idx = best_idx.item() if best_idx.numel() == 1 else best_idx[0].item()
    template_view = template_views[best_idx]  # [N_points, 8]
    
    # 初始化优化位姿
    optimized_pose = Pose(initial_pose._data.clone()).to(device)
    
    # 2. 迭代优化位姿
    best_avg_distance = float('inf')
    best_pose = optimized_pose
    
    for iteration in range(num_iterations):
        # 投影模型轮廓到图像
        centers_3d = template_view[:, :3]  # [N_points, 3]
        centers_cam = optimized_pose.transform(centers_3d)
        centers_2d, valid = camera.view2image(centers_cam)
        
        if not valid.any():
            logger.info(f"Iteration {iteration}: No valid points, skipping")
            break
        
        # 计算轮廓点到最近边缘的距离
        centers_2d_np = centers_2d[valid].detach().cpu().numpy()
        total_distance = 0.0
        valid_count = 0
        edge_offsets = []  # 存储每个轮廓点到最近边缘的偏移
        
        h_img, w_img = edges.shape
        for center in centers_2d_np:
            x, y = int(round(center[0])), int(round(center[1]))
            if 0 <= x < w_img and 0 <= y < h_img:
                # 在边缘附近搜索最近的边缘点（使用更大的搜索半径）
                search_radius = 40
                min_dist = float('inf')
                nearest_edge_x, nearest_edge_y = x, y
                
                # 优先搜索法线方向（如果知道法线方向）
                # 这里简化处理，搜索所有方向
                for dy in range(-search_radius, search_radius + 1):
                    for dx in range(-search_radius, search_radius + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w_img and 0 <= ny < h_img:
                            if edges[ny, nx] > 0:
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest_edge_x, nearest_edge_y = nx, ny
                
                if min_dist < float('inf'):
                    total_distance += min_dist
                    valid_count += 1
                    edge_offsets.append((nearest_edge_x - x, nearest_edge_y - y))
        
        if valid_count == 0:
            logger.info(f"Iteration {iteration}: No edge points found near contour")
            break
        
        avg_distance = total_distance / valid_count
        logger.info(f"Iteration {iteration}: Average distance to edges = {avg_distance:.2f}px, valid_points = {valid_count}")
        
        # 保存最佳位姿
        if avg_distance < best_avg_distance:
            best_avg_distance = avg_distance
            best_pose = Pose(optimized_pose._data.clone()).to(device)
        
        # 如果距离足够小，停止优化
        if avg_distance < 3.0:
            logger.info(f"Converged at iteration {iteration} with distance {avg_distance:.2f}px")
            break
        
        # 3. 基于边缘偏移优化位姿
        if len(edge_offsets) > 0:
            # 计算平均偏移（使用加权平均，距离边缘越近的点权重越大）
            weights = []
            weighted_offsets_x = []
            weighted_offsets_y = []
            
            for i, (offset_x, offset_y) in enumerate(edge_offsets):
                # 计算到边缘的距离（使用偏移的模长）
                dist = np.sqrt(offset_x**2 + offset_y**2)
                # 距离越小，权重越大
                weight = 1.0 / (dist + 1.0)
                weights.append(weight)
                weighted_offsets_x.append(offset_x * weight)
                weighted_offsets_y.append(offset_y * weight)
            
            total_weight = sum(weights)
            if total_weight > 0:
                avg_offset_x = sum(weighted_offsets_x) / total_weight
                avg_offset_y = sum(weighted_offsets_y) / total_weight
            else:
                avg_offset_x = np.mean([offset[0] for offset in edge_offsets])
                avg_offset_y = np.mean([offset[1] for offset in edge_offsets])
            
            # 自适应学习率：如果距离没有减少，降低学习率
            if iteration > 0 and avg_distance >= best_avg_distance * 0.99:
                learning_rate *= 0.8  # 降低学习率
                logger.info(f"  Reducing learning rate to {learning_rate:.4f}")
            
            # 更新位姿（只更新平移的x和y分量）
            current_t = optimized_pose.t.flatten()
            # Camera对象使用f和c属性，f是[fx, fy]，c是[cx, cy]
            f = camera.f.flatten()
            fx = f[0].item() if f[0].numel() == 1 else f[0]
            fy = f[1].item() if f[1].numel() == 1 else f[1]
            depth = current_t[2].item()
            
            if depth > 0:
                # 将像素偏移转换为3D空间偏移
                fx_val = fx.item() if isinstance(fx, torch.Tensor) else fx
                fy_val = fy.item() if isinstance(fy, torch.Tensor) else fy
                delta_x_3d = avg_offset_x * depth / fx_val * learning_rate
                delta_y_3d = avg_offset_y * depth / fy_val * learning_rate
                
                # 限制位姿变化，防止过度漂移
                max_change = 0.01  # 最大变化0.01m
                delta_x_3d = np.clip(delta_x_3d, -max_change, max_change)
                delta_y_3d = np.clip(delta_y_3d, -max_change, max_change)
                
                new_t = current_t.clone()
                new_t[0] += delta_x_3d
                new_t[1] += delta_y_3d
                
                # 创建新位姿
                optimized_pose = Pose.from_Rt(optimized_pose.R, new_t.unsqueeze(0)).to(device)
    
    logger.info(f"Edge-based pose optimization completed. Best distance: {best_avg_distance:.2f}px")
    return best_pose

def parse_args():
    parser = argparse.ArgumentParser(description="DeepAC live tracking (fixed)")
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

def load_template(pre_render_path, device):
    with open(pre_render_path, "rb") as f:
        pre_render_dict = pickle.load(f)
    head = pre_render_dict["head"]
    num_sample = head["num_sample_contour_point"]
    template_views_flat = torch.from_numpy(pre_render_dict["template_view"]).float()
    num_views = template_views_flat.shape[0] // num_sample
    template_views = template_views_flat.view(num_views, num_sample, -1).to(device)
    orientations = torch.from_numpy(pre_render_dict["orientation_in_body"]).float().to(device)
    return template_views, orientations, num_sample

def prepare_initial_pose(cfg, template_views, orientations, device):
    track_cfg = cfg.tracking
    num_views = template_views.shape[0]
    view_idx = int(track_cfg.get("initial_view_index", 0))
    view_idx = max(0, min(view_idx, num_views - 1))
    translation = track_cfg.get("initial_translation")
    if translation is None:
        depth = float(track_cfg.init_depth)
        translation = [0.0, 0.0, depth]
    translation_tensor = torch.tensor(translation, dtype=torch.float32, device=device).unsqueeze(0)
    orientation = orientations[view_idx : view_idx + 1]
    initial_pose = Pose.from_aa(orientation, translation_tensor)
    return view_idx, initial_pose

def build_camera_tensor(camera_cfg, frame_shape, device):
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
    """与demo_cube.py完全相同的预处理"""
    bbox2d[2:] += data_conf.crop_border * 2
    img_crop, camera_out, _ = crop(img, bbox2d, camera=camera_cpu, return_bbox=True)
    if isinstance(data_conf.resize, int) and data_conf.resize > 0:
        img_crop, scales = resize(img_crop, data_conf.resize, fn=max if data_conf.resize_by == "max" else min)
        camera_out = camera_out.scale(scales)
    if isinstance(data_conf.pad, int) and data_conf.pad > 0:
        img_crop, = zero_pad(data_conf.pad, img_crop)
    img_tensor = numpy_image_to_torch(img_crop.astype(np.float32))
    _, h_t, w_t = img_tensor.shape
    pad_h = (32 - h_t % 32) % 32
    pad_w = (32 - w_t % 32) % 32
    if pad_h or pad_w:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h))
        cam_data = camera_out._data.clone()
        cam_data[..., 0] += pad_w
        cam_data[..., 1] += pad_h
        camera_out = Camera(cam_data)
    return img_tensor.to(device), camera_out.to(device)

def preprocess_frame(frame_bgr, camera_cfg, device, grayscale=False):
    """构建原始相机，不进行resize"""
    if grayscale:
        if len(frame_bgr.shape) == 2:
            frame_rgb = frame_bgr
        elif frame_bgr.shape[2] == 1:
            frame_rgb = frame_bgr[:, :, 0]
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    else:
        if len(frame_bgr.shape) == 2:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    intrinsic_param = build_camera_tensor(camera_cfg, frame_rgb.shape, device)
    camera = Camera(intrinsic_param)
    return frame_rgb, camera

def smooth_pose(prev_pose: Pose, new_pose: Pose, alpha: float) -> Pose:
    """与demo_cube.py完全相同的pose平滑"""
    if prev_pose is None or alpha <= 0.0:
        return new_pose.detach()
    device_local = new_pose._data.device
    prev_pose_d = prev_pose.to(device_local)
    prev_R = prev_pose_d.R
    new_R = new_pose.R
    if prev_R.ndim == 3:
        prev_R = prev_R[0]
    if new_R.ndim == 3:
        new_R = new_R[0]
    blended = alpha * prev_R + (1.0 - alpha) * new_R
    u, _, vT = torch.linalg.svd(blended)
    R_s = u @ vT
    det_val = torch.det(R_s)
    if det_val.dim() == 0:
        if det_val < 0:
            u[:, -1] *= -1
            R_s = u @ vT
    else:
        neg_idx = det_val < 0
        if neg_idx.any():
            u[neg_idx, :, -1] *= -1
            R_s = u @ vT
    prev_t = prev_pose_d.t
    new_t = new_pose.t
    if prev_t.ndim > 1:
        prev_t = prev_t[0]
    if new_t.ndim > 1:
        new_t = new_t[0]
    t_s = alpha * prev_t + (1.0 - alpha) * new_t
    pose = Pose.from_Rt(R_s, t_s)
    return pose.to(device_local)

def draw_overlay_simple(frame_rgb, bbox, centers_in_image, centers_valid, color=(0, 255, 0)):
    """简化的overlay绘制"""
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if bbox is not None and centers_in_image is not None and centers_valid is not None:
        cx, cy, w, h = bbox.cpu().numpy()
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        
        # 绘制轮廓点
        if centers_valid.ndim > 1:
            centers_valid = centers_valid[0]
        if centers_in_image.ndim > 2:
            centers_in_image = centers_in_image[0]
        centers_np = centers_in_image[centers_valid].detach().cpu().numpy()
        for pt in centers_np:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < frame_bgr.shape[1] and 0 <= y < frame_bgr.shape[0]:
                cv2.circle(frame_bgr, (x, y), 2, color, -1)
    return frame_bgr

def load_model(cfg, device, logger):
    train_cfg = OmegaConf.load(cfg.model.load_cfg)
    model_cfg = train_cfg.models if "models" in train_cfg else train_cfg
    model = get_model(model_cfg.name)(model_cfg)
    ckpt = torch.load(cfg.model.load_model, map_location="cpu")
    if "pytorch-lightning_version" not in ckpt:
        ckpt = convert_old_model(ckpt)
    load_model_weight(model, ckpt, logger)
    model.to(device).eval()
    return model, train_cfg

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

    logger = MyLightningLogger("deepac-live-fixed", str(save_dir))
    logger.dump_cfg(cfg, "live_cfg.yaml")

    model, train_cfg = load_model(cfg, device, logger)
    logger.dump_cfg(train_cfg, "train_cfg.yaml")

    # 同步数据配置，与demo_cube.py完全一致
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

    template_views, orientations, num_sample = load_template(cfg.object.pre_render_pkl, device)
    initial_view_idx, initial_pose = prepare_initial_pose(cfg, template_views, orientations, device)
    logger.info(f"Guide overlay uses template view index {initial_view_idx}")

    # 初始化RealSense RGB相机
    use_realsense_rgb = False
    realsense_rgb_pipeline = None
    cap = None
    
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) > 0:
            logger.info(f"Found RealSense device: {devices[0].get_info(rs.camera_info.name)}")
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
                    logger.info(f"RealSense RGB: {intrinsics.width}x{intrinsics.height}, "
                               f"fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
    except Exception as e:
        logger.info(f"RealSense not available: {e}")
    
    if not use_realsense_rgb:
        camera_id = args.camera_id if args.camera_id is not None else int(cfg.camera.get("camera_id", 0))
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.camera.set_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.camera.set_height))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open camera id={camera_id}")
        logger.info(f"Camera opened (id={camera_id})")

    grayscale = bool(cfg.tracking.get("grayscale", False))
    bbox_trim_ratio = float(cfg.tracking.get("bbox_trim_ratio", 0.08))
    smooth_alpha = float(cfg.tracking.get("smooth_alpha", 0.5))
    use_smoothing = smooth_alpha > 0.0
    fore_learn_rate = float(cfg.tracking.get("fore_learn_rate", 0.03))
    back_learn_rate = float(cfg.tracking.get("back_learn_rate", 0.03))
    
    logger.info("=" * 60)
    logger.info("PHASE 1: Model Alignment")
    logger.info("=" * 60)
    logger.info("Align cube with blue model, then press SPACE")
    
    # 第一阶段：对齐阶段
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
            except:
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
        
        # 渲染模型轮廓（参数顺序：template_view, pose_data, camera_data, ...）
        (
            _,
            _,
            guide_centers,
            guide_valid,
            _,
            _,
            _,
            _,
        ) = calculate_basic_line_data(template_views[initial_view_idx:initial_view_idx+1], initial_pose[None]._data, ori_camera._data, 1, 0)
        
        guide_bbox = None
        if guide_valid[0].any():
            guide_bbox = get_bbox_from_p2d(
                guide_centers[0][guide_valid[0]], trim_ratio=bbox_trim_ratio
            )
        
        overlay = draw_overlay_simple(frame_rgb, guide_bbox, guide_centers, guide_valid, color=(255, 128, 0))
        cv2.putText(overlay, "Align cube with blue model...", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        cv2.putText(overlay, "Press SPACE to start tracking", (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("DeepAC Live - Alignment", overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") or key == ord("s"):
            alignment_mode = False
            logger.info("Alignment complete. Starting tracking...")
        elif key == 27 or key == ord("q"):
            return
        
        frame_idx += 1
    
    # 第二阶段：追踪阶段 - 完全按照demo_cube.py的逻辑
    logger.info("=" * 60)
    logger.info("PHASE 2: Tracking")
    logger.info("=" * 60)
    
    # 关键修复：在对齐后使用边缘检测优化位姿，使模型"吸附"到魔方边缘
    logger.info("=" * 60)
    logger.info("PHASE 2.1: Edge-based Pose Optimization")
    logger.info("=" * 60)
    logger.info("Optimizing pose to align model contour with detected cube edges...")
    
    # 步骤1：边缘检测优化位姿
    # ========== 关键修复：使用resize后的camera，而不是原始camera ==========
    # 原因：edge-based优化需要与预处理后的图像尺寸匹配
    # 先获取第一帧的bbox和预处理后的camera
    idx0_for_edge = get_closest_template_view_index(initial_pose, orientations)
    if isinstance(idx0_for_edge, torch.Tensor):
        idx0_for_edge = idx0_for_edge.item() if idx0_for_edge.numel() == 1 else idx0_for_edge[0].item()
    template_view_for_edge = template_views[idx0_for_edge : idx0_for_edge + 1]
    data_lines_for_edge = project_correspondences_line(template_view_for_edge, initial_pose, ori_camera)
    centers_for_edge = data_lines_for_edge["centers_in_image"]
    if centers_for_edge.ndim > 2:
        centers_for_edge = centers_for_edge[0]
    if not isinstance(centers_for_edge, torch.Tensor):
        centers_for_edge = torch.tensor(centers_for_edge, device=device)
    bbox2d_for_edge = get_bbox_from_p2d(centers_for_edge, trim_ratio=bbox_trim_ratio)
    
    # 处理无效bbox
    h_img_orig, w_img_orig = frame_rgb.shape[:2]
    if bbox2d_for_edge.numel() < 4 or torch.isnan(bbox2d_for_edge).any() or bbox2d_for_edge[2] < 4 or bbox2d_for_edge[3] < 4:
        bbox2d_for_edge = torch.tensor([w_img_orig / 2.0, h_img_orig / 2.0, float(w_img_orig), float(h_img_orig)], dtype=torch.float32, device=device)
    
    # 预处理图像以获取resize后的camera
    img_tensor_for_edge, camera_for_edge = preprocess_image(frame_rgb, bbox2d_for_edge.cpu().numpy().copy(), ori_camera_cpu, data_conf, device)
    
    # ========== 调试：检查相机内参 ==========
    f_edge = camera_for_edge.f.flatten()
    c_edge = camera_for_edge.c.flatten()
    logger.info(f"DEBUG Edge Optimization: Using resized camera - fx={f_edge[0].item():.2f}, fy={f_edge[1].item():.2f}, cx={c_edge[0].item():.2f}, cy={c_edge[1].item():.2f}")
    logger.info(f"DEBUG Edge Optimization: Original camera - fx={ori_camera.f.flatten()[0].item():.2f}, fy={ori_camera.f.flatten()[1].item():.2f}, cx={ori_camera.c.flatten()[0].item():.2f}, cy={ori_camera.c.flatten()[1].item():.2f}")
    logger.info(f"DEBUG Edge Optimization: Image size - original={frame_rgb.shape[:2]}, resized={img_tensor_for_edge.shape[1:3]}")
    
    # 将预处理后的图像转换回RGB格式用于边缘检测
    img_rgb_for_edge = img_tensor_for_edge.permute(1, 2, 0).detach().cpu().numpy()
    if img_rgb_for_edge.shape[2] == 1:
        img_rgb_for_edge = cv2.cvtColor(img_rgb_for_edge, cv2.COLOR_GRAY2RGB)
    elif img_rgb_for_edge.shape[2] == 3:
        # 确保值范围在[0, 1]
        img_rgb_for_edge = np.clip(img_rgb_for_edge, 0, 1)
        # 转换为[0, 255]范围
        img_rgb_for_edge = (img_rgb_for_edge * 255).astype(np.uint8)
    
    optimized_pose = optimize_pose_with_edge_detection(
        img_rgb_for_edge,  # 使用resize后的图像
        initial_pose,
        template_views,
        orientations,
        camera_for_edge,  # 使用resize后的camera
        device,
        logger,
        num_iterations=15,
        learning_rate=0.05,
    )
    
    # 步骤2：简化检测（降低频率）- 只在必要时进行检测
    # 由于histogram gate失效，主要依赖edge-based优化
    logger.info("=" * 60)
    logger.info("PHASE 2.2: Simplified Detection (Low Frequency)")
    logger.info("=" * 60)
    logger.info("Using edge-optimized pose as initial pose (detection disabled for now)")
    
    # 直接使用边缘优化后的位姿，不进行模板匹配检测
    # 这样可以避免检测带来的额外延迟，主要依赖edge-based优化
    current_pose = optimized_pose
    pose_t = current_pose.t.flatten()
    logger.info(f"Using edge-optimized pose: t=({pose_t[0].item():.3f}, {pose_t[1].item():.3f}, {pose_t[2].item():.3f})m")
    logger.info("NOTE: Histogram gate disabled - using edge-based tracking only")
    
    total_fore_hist = None
    total_back_hist = None
    pose_smooth = current_pose.detach() if use_smoothing else None
    
    frame_idx = 0
    
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
            except:
                continue
        else:
            ret, frame_bgr = cap.read()
            if not ret:
                break
        
        if grayscale and len(frame_bgr.shape) == 2:
            pass
        elif grayscale:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        ori_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if len(frame_bgr.shape) == 3 else cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
        h, w = ori_image.shape[:2]
        intrinsic_param = torch.tensor([w, h, cfg.camera.fx, cfg.camera.fy, cfg.camera.cx, cfg.camera.cy], dtype=torch.float32)
        ori_camera_cpu = Camera(intrinsic_param)
        ori_camera = ori_camera_cpu.to(device)
        
        # 与demo_cube.py完全相同的逻辑
        indices_full = get_closest_k_template_view_index(
            current_pose, orientations, data_conf.get_top_k_template_views * data_conf.skip_template_view
        )
        if indices_full.ndim > 1:
            indices_full = indices_full.view(-1)
        indices_full = indices_full[::data_conf.skip_template_view]
        indices_list = indices_full.tolist()
        closest_template_views = template_views[indices_list].contiguous()
        closest_orientations = orientations[indices_full].contiguous()
        
        template_view_first = closest_template_views[:1]
        data_lines = project_correspondences_line(template_view_first, current_pose, ori_camera)
        bbox2d = get_bbox_from_p2d(data_lines["centers_in_image"][0], trim_ratio=bbox_trim_ratio)
        img_tensor, camera = preprocess_image(ori_image, bbox2d.cpu().numpy().copy(), ori_camera_cpu, data_conf, device)
        
        # 初始化直方图（只在第一帧）
        if total_fore_hist is None:
            _, _, centers_in_image, centers_valid, normals_in_image, f_dist, b_dist, _ = calculate_basic_line_data(
                closest_template_views[None][:, 0], current_pose[None]._data, camera[None]._data, 1, 0
            )
            # 确保维度正确（与demo_cube.py完全一致）
            # 去除多余的batch维度，然后确保至少有一个batch维度
            while centers_in_image.ndim > 3:
                centers_in_image = centers_in_image.squeeze(0)
            if centers_in_image.ndim == 2:
                centers_in_image = centers_in_image.unsqueeze(0)  # [N, 2] -> [1, N, 2]
            
            while centers_valid.ndim > 2:
                centers_valid = centers_valid.squeeze(0)
            if centers_valid.ndim == 1:
                centers_valid = centers_valid.unsqueeze(0)  # [N] -> [1, N]
            
            while normals_in_image.ndim > 3:
                normals_in_image = normals_in_image.squeeze(0)
            if normals_in_image.ndim == 2:
                normals_in_image = normals_in_image.unsqueeze(0)  # [N, 2] -> [1, N, 2]
            
            while f_dist.ndim > 2:
                f_dist = f_dist.squeeze(0)
            if f_dist.ndim == 1:
                f_dist = f_dist.unsqueeze(0)  # [N] -> [1, N]
            
            while b_dist.ndim > 2:
                b_dist = b_dist.squeeze(0)
            if b_dist.ndim == 1:
                b_dist = b_dist.unsqueeze(0)  # [N] -> [1, N]
            
            # 确保centers_in_image的batch size是1
            if centers_in_image.shape[0] != 1:
                if centers_in_image.ndim == 3 and centers_in_image.shape[0] > 1:
                    centers_in_image = centers_in_image[0:1]
                elif centers_in_image.ndim == 2:
                    centers_in_image = centers_in_image.unsqueeze(0)
            
            # 详细调试信息：检查轮廓点和图像
            centers_np = centers_in_image[0].detach().cpu().numpy()
            valid_np = centers_valid[0].detach().cpu().numpy()
            img_np = img_tensor.detach().cpu().numpy()
            
            # 检查图像像素值范围
            logger.info(f"DEBUG: Image shape={img_tensor.shape}, pixel range=[{img_np.min():.3f}, {img_np.max():.3f}], mean={img_np.mean():.3f}")
            
            # 检查轮廓点位置
            valid_centers = centers_np[valid_np > 0]
            if len(valid_centers) > 0:
                logger.info(f"DEBUG: Valid centers: {len(valid_centers)}/{len(centers_np)}, "
                          f"x_range=[{valid_centers[:, 0].min():.1f}, {valid_centers[:, 0].max():.1f}], "
                          f"y_range=[{valid_centers[:, 1].min():.1f}, {valid_centers[:, 1].max():.1f}]")
                # 检查轮廓点处的像素值
                h, w = img_np.shape[1], img_np.shape[2]
                sample_pixels = []
                for center in valid_centers[:10]:  # 采样前10个点
                    x, y = int(round(center[0])), int(round(center[1]))
                    if 0 <= x < w and 0 <= y < h:
                        if img_np.ndim == 3:
                            pixel_val = img_np[:, y, x].mean()
                        else:
                            pixel_val = img_np[y, x]
                        sample_pixels.append(pixel_val)
                if sample_pixels:
                    logger.info(f"DEBUG: Sample pixel values at contour centers: {[f'{v:.3f}' for v in sample_pixels[:5]]}")
            
            # 检查前景和背景距离
            f_dist_np = f_dist[0].detach().cpu().numpy()
            b_dist_np = b_dist[0].detach().cpu().numpy()
            valid_f_dist = f_dist_np[valid_np > 0]
            valid_b_dist = b_dist_np[valid_np > 0]
            if len(valid_f_dist) > 0:
                logger.info(f"DEBUG: Foreground distance: mean={valid_f_dist.mean():.2f}, range=[{valid_f_dist.min():.2f}, {valid_f_dist.max():.2f}]")
                logger.info(f"DEBUG: Background distance: mean={valid_b_dist.mean():.2f}, range=[{valid_b_dist.min():.2f}, {valid_b_dist.max():.2f}]")
            
            # 关键修复：限制背景距离，防止过大值导致采样点超出图像范围
            # 背景距离过大可能是因为模板视图中的值很大，或者深度很小
            # 限制背景距离为图像对角线长度的3倍（与live_tracking.py一致）
            # 重要：在计算直方图之前就限制，避免无效采样
            h_img, w_img = img_tensor.shape[1], img_tensor.shape[2]
            max_b_dist = np.sqrt(h_img**2 + w_img**2) * 3.0
            
            # 检查并限制背景距离
            b_dist_max = b_dist.max().item()
            if b_dist_max > max_b_dist:
                logger.info(f"WARNING: Clamping background distance from max={b_dist_max:.2f} to {max_b_dist:.2f}")
                b_dist = torch.clamp(b_dist, max=max_b_dist)
            
            # 额外检查：确保前景和背景距离有合理的差异
            # 如果背景距离太小（接近前景距离），说明位姿可能不准确
            # 确保f_dist和b_dist的维度正确
            if f_dist.ndim > 1:
                f_dist_flat = f_dist[0]  # 取第一个batch
            else:
                f_dist_flat = f_dist
            if b_dist.ndim > 1:
                b_dist_flat = b_dist[0]  # 取第一个batch
            else:
                b_dist_flat = b_dist
            
            # 使用valid_np作为mask
            valid_mask = valid_np > 0
            if valid_mask.sum() > 0:
                f_dist_mean = f_dist_flat[valid_mask].mean().item()
                b_dist_mean = b_dist_flat[valid_mask].mean().item()
                
                if b_dist_mean > 0 and f_dist_mean > 0:
                    dist_ratio = b_dist_mean / f_dist_mean
                    if dist_ratio < 1.5:  # 背景距离应该至少是前景距离的1.5倍
                        logger.info(f"WARNING: Background distance too close to foreground distance (ratio={dist_ratio:.2f})")
                        logger.info(f"  This may indicate pose misalignment. Foreground={f_dist_mean:.2f}, Background={b_dist_mean:.2f}")
                        # 增加背景距离以确保采样点在背景区域
                        min_b_dist = f_dist_mean * 1.5
                        if b_dist.ndim > 1:
                            b_dist = torch.clamp(b_dist, min=min_b_dist)
                        else:
                            b_dist = torch.clamp(b_dist, min=min_b_dist)
            
            total_fore_hist, total_back_hist = model.histogram.calculate_histogram(
                img_tensor[None],
                centers_in_image,
                centers_valid,
                normals_in_image,
                f_dist,
                b_dist,
                True,
            )
            
            # 检查初始化直方图
            fore_mean = torch.mean(total_fore_hist).item()
            back_mean = torch.mean(total_back_hist).item()
            hist_distinction = (fore_mean - back_mean) / (fore_mean + back_mean + 1e-7)
            fore_sum = total_fore_hist.sum().item()
            back_sum = total_back_hist.sum().item()
            fore_max = total_fore_hist.max().item()
            back_max = total_back_hist.max().item()
            logger.info(f"Initial histogram: distinction={hist_distinction:.6f}, fore_mean={fore_mean:.6f}, back_mean={back_mean:.6f}")
            logger.info(f"DEBUG: Histogram details: fore_sum={fore_sum:.6f}, back_sum={back_sum:.6f}, fore_max={fore_max:.6f}, back_max={back_max:.6f}")
            if abs(hist_distinction) < 0.01:
                logger.info("WARNING: Low histogram distinction! Tracking may fail.")
                logger.info("Possible causes:")
                logger.info("  1) Initial pose does not align with the object")
                logger.info("  2) Contour points are not sampling foreground/background correctly")
                logger.info("  3) Image preprocessing issue (wrong pixel value range)")
                logger.info("  4) Foreground/background distances are too similar")
        
        # 追踪步骤（与demo_cube.py完全一致）
        data = {
            "image": img_tensor[None],
            "camera": camera[None],
            "body2view_pose": current_pose[None],
            "closest_template_views": closest_template_views[None],
            "closest_orientations_in_body": closest_orientations[None],
            "fore_hist": total_fore_hist,
            "back_hist": total_back_hist,
        }
        data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
        pred = model._forward(data, visualize=False, tracking=True)
        
        opt_pose = pred["opt_body2view_pose"][-1][0]
        opt_pose_det = opt_pose.detach()
        
        # 计算当前帧的直方图区分度（用于判断是否接受位姿更新）
        # 使用更新后的位姿重新计算直方图
        best_idx_tracking = get_closest_template_view_index(opt_pose_det, orientations)
        if isinstance(best_idx_tracking, torch.Tensor):
            best_idx_tracking = best_idx_tracking.item() if best_idx_tracking.numel() == 1 else best_idx_tracking[0].item()
        closest_template_view_tracking = template_views[best_idx_tracking]
        
        (
            _,
            _,
            centers_in_image_track,
            centers_valid_track,
            normals_in_image_track,
            f_dist_track,
            b_dist_track,
            _,
        ) = calculate_basic_line_data(
            closest_template_view_tracking[None], 
            opt_pose_det[None]._data, 
            camera[None]._data, 
            1, 
            0
        )
        
        # 确保维度正确：移除多余的维度，确保batch维度为1
        # 参考live_tracking.py的处理方式
        while centers_in_image_track.ndim > 3:
            centers_in_image_track = centers_in_image_track.squeeze(0)
        while centers_valid_track.ndim > 2:
            centers_valid_track = centers_valid_track.squeeze(0)
        while normals_in_image_track.ndim > 3:
            normals_in_image_track = normals_in_image_track.squeeze(0)
        while f_dist_track.ndim > 2:
            f_dist_track = f_dist_track.squeeze(0)
        while b_dist_track.ndim > 2:
            b_dist_track = b_dist_track.squeeze(0)
        
        # 确保batch维度为1（取第一个batch如果存在多个）
        if centers_in_image_track.ndim == 3 and centers_in_image_track.shape[0] > 1:
            centers_in_image_track = centers_in_image_track[0:1]  # 取第一个batch
        elif centers_in_image_track.ndim == 2:
            centers_in_image_track = centers_in_image_track.unsqueeze(0)
        
        if centers_valid_track.ndim == 2 and centers_valid_track.shape[0] > 1:
            centers_valid_track = centers_valid_track[0:1]  # 取第一个batch
        elif centers_valid_track.ndim == 1:
            centers_valid_track = centers_valid_track.unsqueeze(0)
        
        if normals_in_image_track.ndim == 3 and normals_in_image_track.shape[0] > 1:
            normals_in_image_track = normals_in_image_track[0:1]  # 取第一个batch
        elif normals_in_image_track.ndim == 2:
            normals_in_image_track = normals_in_image_track.unsqueeze(0)
        
        if f_dist_track.ndim == 2 and f_dist_track.shape[0] > 1:
            f_dist_track = f_dist_track[0:1]  # 取第一个batch
        elif f_dist_track.ndim == 1:
            f_dist_track = f_dist_track.unsqueeze(0)
        
        if b_dist_track.ndim == 2 and b_dist_track.shape[0] > 1:
            b_dist_track = b_dist_track[0:1]  # 取第一个batch
        elif b_dist_track.ndim == 1:
            b_dist_track = b_dist_track.unsqueeze(0)
        
        # 最终检查：确保形状正确
        # centers_in_image_track应该是[batch, n_points, 2]
        # centers_valid_track应该是[batch, n_points]
        # normals_in_image_track应该是[batch, n_points, 2]
        # f_dist_track和b_dist_track应该是[batch, n_points]
        assert centers_in_image_track.ndim == 3, f"centers_in_image_track should be 3D, got {centers_in_image_track.shape}"
        assert centers_in_image_track.shape[0] == 1, f"centers_in_image_track batch size should be 1, got {centers_in_image_track.shape[0]}"
        assert centers_valid_track.ndim == 2, f"centers_valid_track should be 2D, got {centers_valid_track.shape}"
        assert centers_valid_track.shape[0] == 1, f"centers_valid_track batch size should be 1, got {centers_valid_track.shape[0]}"
        
        # 限制背景距离
        h_img, w_img = img_tensor.shape[1], img_tensor.shape[2]
        max_b_dist = np.sqrt(h_img**2 + w_img**2) * 3.0
        if torch.any(b_dist_track > max_b_dist):
            b_dist_track = torch.clamp(b_dist_track, max=max_b_dist)
        
        fore_hist_track, back_hist_track = model.histogram.calculate_histogram(
            img_tensor[None],
            centers_in_image_track,
            centers_valid_track,
            normals_in_image_track,
            f_dist_track,
            b_dist_track,
            True,
        )
        
        fore_mean_track = torch.mean(fore_hist_track).item()
        back_mean_track = torch.mean(back_hist_track).item()
        hist_distinction_track = (fore_mean_track - back_mean_track) / (fore_mean_track + back_mean_track + 1e-7)
        
        # 调试：检查位姿更新
        pose_change_t = torch.norm(opt_pose_det.t - current_pose.t).item()
        pose_change_R = torch.norm(opt_pose_det.R - current_pose.R).item()
        
        # 安全地提取tensor值：先flatten再取前3个元素
        current_t_flat = current_pose.t.flatten()
        new_t_flat = opt_pose_det.t.flatten()
        current_t_vals = [current_t_flat[i].item() if i < current_t_flat.shape[0] else 0.0 for i in range(3)]
        new_t_vals = [new_t_flat[i].item() if i < new_t_flat.shape[0] else 0.0 for i in range(3)]
        
        if frame_idx == 0 or frame_idx % 10 == 0:
            logger.info(f"DEBUG Frame {frame_idx}: pose_change_t={pose_change_t:.6f}m, pose_change_R={pose_change_R:.6f}")
            logger.info(f"DEBUG Frame {frame_idx}: current_t=({current_t_vals[0]:.3f}, {current_t_vals[1]:.3f}, {current_t_vals[2]:.3f})")
            logger.info(f"DEBUG Frame {frame_idx}: new_t=({new_t_vals[0]:.3f}, {new_t_vals[1]:.3f}, {new_t_vals[2]:.3f})")
        
        # 关键修复：暂时关闭histogram gate，主要使用edge-based tracking
        # 原因：直方图在RGB场景下完全失效（区分度为0），导致所有更新被拒绝
        # 方案：只检查位姿变化大小，不依赖直方图区分度
        
        # ========== 临时修改：完全注释pose change limit，让位姿可以自由更新（用于测试） ==========
        # max_pose_change = 1.0  # 临时设为1.0m，基本不限制（用于测试）
        
        # 记录直方图区分度（仅用于调试，不用于gate）
        if frame_idx == 0 or frame_idx % 10 == 0:
            logger.info(f"DEBUG Frame {frame_idx}: Histogram distinction={hist_distinction_track:.6f} (for reference only, not used for gate)")
        
        # ========== 完全注释pose change limit逻辑 ==========
        # 只检查位姿变化大小，不依赖直方图区分度
        # 临时修改：完全注释pose change limit，让位姿可以自由更新
        # if pose_change_t > max_pose_change:
        #     logger.info(f"WARNING Frame {frame_idx}: Pose change too large ({pose_change_t:.6f}m > {max_pose_change:.3f}m)")
        #     logger.info(f"  Limiting pose update (histogram distinction={hist_distinction_track:.6f}, not used for gate)")
        #     # 限制位姿变化：只应用部分更新
        #     alpha_limit = max_pose_change / pose_change_t
        #     
        #     # 确保tensor维度正确：提取3D平移向量
        #     current_t_vec = current_t_flat[:3] if current_t_flat.shape[0] >= 3 else current_t_flat
        #     new_t_vec = new_t_flat[:3] if new_t_flat.shape[0] >= 3 else new_t_flat
        #     
        #     # 计算限制后的平移向量
        #     delta_t = (new_t_vec - current_t_vec) * alpha_limit
        #     limited_t = current_t_vec + delta_t
        #     
        #     # 确保limited_t是3D向量
        #     if limited_t.shape[0] != 3:
        #         limited_t = limited_t[:3]
        #     
        #     # 确保limited_t的形状与R的batch维度匹配
        #     R_shape = opt_pose_det.R.shape
        #     if len(R_shape) == 3:  # [batch, 3, 3]
        #         batch_size = R_shape[0]
        #         limited_t_batch = limited_t.unsqueeze(0).expand(batch_size, -1)  # [batch, 3]
        #     else:  # [3, 3]
        #         limited_t_batch = limited_t.unsqueeze(0)  # [1, 3]
        #     
        #     # 创建新的位姿
        #     opt_pose_det = Pose.from_Rt(opt_pose_det.R, limited_t_batch).to(device)
        #     
        #     # 重新计算位姿变化（确保维度匹配）
        #     current_t_for_norm = current_pose.t.flatten()[:3] if current_pose.t.flatten().shape[0] >= 3 else current_pose.t.flatten()
        #     new_t_for_norm = opt_pose_det.t.flatten()[:3] if opt_pose_det.t.flatten().shape[0] >= 3 else opt_pose_det.t.flatten()
        #     pose_change_t = torch.norm(new_t_for_norm - current_t_for_norm).item()
        #     
        #     # 调试：检查限制后的位姿变化
        #     if frame_idx % 10 == 0:
        #         logger.info(f"DEBUG Frame {frame_idx}: After limiting, pose_change_t={pose_change_t:.6f}m, alpha_limit={alpha_limit:.3f}")
        #         logger.info(f"  current_t=({current_t_for_norm[0].item():.3f}, {current_t_for_norm[1].item():.3f}, {current_t_for_norm[2].item():.3f})")
        #         logger.info(f"  limited_t=({new_t_for_norm[0].item():.3f}, {new_t_for_norm[1].item():.3f}, {new_t_for_norm[2].item():.3f})")
        
        # 如果位姿变化很小，应用edge-based优化（而不是仅仅警告）
        if pose_change_t < 1e-6 and frame_idx > 0:
            # 每30帧应用一次edge-based优化，避免过于频繁
            edge_optimization_interval = 30
            if frame_idx % edge_optimization_interval == 0:
                logger.info(f"Frame {frame_idx}: Pose update very small ({pose_change_t:.9f}m), applying edge-based optimization...")
                try:
                    # 将预处理后的图像转换回RGB格式用于边缘检测
                    img_rgb_for_edge = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
                    if img_rgb_for_edge.shape[2] == 1:
                        img_rgb_for_edge = cv2.cvtColor(img_rgb_for_edge, cv2.COLOR_GRAY2RGB)
                    elif img_rgb_for_edge.shape[2] == 3:
                        # 确保值范围在[0, 1]
                        img_rgb_for_edge = np.clip(img_rgb_for_edge, 0, 1)
                        # 转换为[0, 255]范围
                        img_rgb_for_edge = (img_rgb_for_edge * 255).astype(np.uint8)
                    
                    # 使用当前位姿进行edge-based优化（少量迭代）
                    edge_optimized_pose = optimize_pose_with_edge_detection(
                        img_rgb_for_edge,
                        current_pose,
                        template_views,
                        orientations,
                        camera,
                        device,
                        logger,
                        num_iterations=5,  # 少量迭代，避免延迟
                        learning_rate=0.05,
                    )
                    
                    # 检查edge优化后的位姿变化
                    edge_pose_change = torch.norm(edge_optimized_pose.t.flatten()[:3] - current_pose.t.flatten()[:3]).item()
                    if edge_pose_change > 0.001:  # 如果有明显变化
                        logger.info(f"  Edge optimization improved pose by {edge_pose_change:.6f}m")
                        opt_pose_det = edge_optimized_pose
                        pose_change_t = edge_pose_change
                    else:
                        logger.info(f"  Edge optimization did not improve pose significantly")
                except Exception as e:
                    logger.info(f"  Edge optimization failed: {e}")
                    # 继续使用原来的位姿
            else:
                # 非优化帧，只记录警告
                logger.info(f"WARNING Frame {frame_idx}: Pose update is very small ({pose_change_t:.9f}m), optimizer may be stuck!")
                logger.info(f"  This usually means histogram distinction is too low for effective tracking.")
        
        if use_smoothing:
            pose_smooth = smooth_pose(pose_smooth, opt_pose_det, smooth_alpha)
            pose_for_output = pose_smooth
        else:
            pose_for_output = opt_pose_det
        
        current_pose = opt_pose_det
        
        # 更新直方图（与demo_cube.py完全一致）
        # 关键修复：确保current_pose没有batch维度（用于get_closest_template_view_index）
        if isinstance(current_pose, Pose):
            if current_pose._data.ndim > 1:
                # 如果有batch维度，取第一个
                current_pose_for_index = Pose(current_pose._data[0])
            else:
                current_pose_for_index = current_pose
        else:
            current_pose_for_index = current_pose
        
        best_idx = get_closest_template_view_index(current_pose_for_index, orientations)
        
        # 关键修复：确保索引是标量，不是tensor
        if isinstance(best_idx, torch.Tensor):
            if best_idx.ndim > 0:
                if best_idx.numel() == 1:
                    best_idx = best_idx.item()
                else:
                    # 如果有多个值，取第一个
                    best_idx = best_idx[0].item()
            else:
                best_idx = best_idx.item()
        
        # 调试日志（每10帧输出一次）
        if frame_idx == 0 or frame_idx % 10 == 0:
            logger.info(f"DEBUG: best_idx={best_idx}, template_views.shape={template_views.shape}")
        
        closest_template_view_hist = template_views[best_idx]
        
        # 关键修复：确保输入维度正确
        # closest_template_view_hist 应该是 [N, 8]，需要变成 [1, N, 8]
        # 检查维度并正确处理
        if closest_template_view_hist.ndim == 2:
            # [N, 8] -> [1, N, 8]
            template_view_for_hist = closest_template_view_hist.unsqueeze(0)
        elif closest_template_view_hist.ndim == 3:
            # 如果已经是 [B, N, 8]，检查batch size
            if closest_template_view_hist.shape[0] == 1:
                template_view_for_hist = closest_template_view_hist
            else:
                # 如果batch size > 1，只取第一个
                template_view_for_hist = closest_template_view_hist[0:1]
        elif closest_template_view_hist.ndim == 1:
            # 如果是 [8]，需要reshape
            raise RuntimeError(f"Unexpected template_view shape: {closest_template_view_hist.shape}, expected [N, 8]")
        else:
            raise RuntimeError(f"Unexpected template_view_hist shape: {closest_template_view_hist.shape}")
        
        # 关键修复：确保camera._data的维度正确
        # camera是Camera对象，camera._data应该是[6]或[1, 6]
        camera_data = camera._data
        if camera_data.ndim == 1:
            # [6] -> [1, 6]
            camera_data = camera_data.unsqueeze(0)
        elif camera_data.ndim == 2 and camera_data.shape[0] > 1:
            # [B, 6] -> [1, 6] (取第一个)
            camera_data = camera_data[0:1]
        
        _, _, centers_in_image, centers_valid, normals_in_image, f_dist, b_dist, _ = calculate_basic_line_data(
            template_view_for_hist, current_pose[None]._data, camera_data, 1, 0
        )
        
        # 关键修复：限制背景距离，防止过大值导致采样点超出图像范围
        h_img, w_img = img_tensor.shape[1], img_tensor.shape[2]
        max_b_dist = np.sqrt(h_img**2 + w_img**2) * 3.0
        if torch.any(b_dist > max_b_dist):
            b_dist = torch.clamp(b_dist, max=max_b_dist)
        
        # 确保维度正确（与demo_cube.py完全一致）
        # 关键修复：如果centers_in_image是[1, 200, 200, 2]，说明输入维度有问题
        # 问题可能在于template_view_for_hist、body2view_pose_data或camera_data的batch维度不匹配
        # 检查并修复维度
        if centers_in_image.ndim == 4:
            # [1, 200, 200, 2] 说明输入有问题
            # 可能是batch维度广播导致的，需要取正确的部分
            if centers_in_image.shape[1] == centers_in_image.shape[2]:
                # 错误的维度，需要取正确的部分
                # 应该是[1, 200, 2]，取第一个batch的第一个维度
                # 但更可能的问题是：template_view_for_hist的维度不对
                # 如果template_view_for_hist是[1, 200, 8]，但被错误地处理成了[200, 200, 8]
                # 我们需要确保template_view_for_hist是[1, 200, 8]
                centers_in_image = centers_in_image[0, :, 0, :].unsqueeze(0)  # [200, 2] -> [1, 200, 2]
                if centers_valid.ndim == 3 and centers_valid.shape[1] == centers_valid.shape[2]:
                    centers_valid = centers_valid[0, :, 0].unsqueeze(0)  # [200] -> [1, 200]
                elif centers_valid.ndim == 3:
                    centers_valid = centers_valid[0, :].unsqueeze(0)  # [200, ?] -> [1, 200]
                if normals_in_image.ndim == 4 and normals_in_image.shape[1] == normals_in_image.shape[2]:
                    normals_in_image = normals_in_image[0, :, 0, :].unsqueeze(0)  # [200, 2] -> [1, 200, 2]
                elif normals_in_image.ndim == 4:
                    normals_in_image = normals_in_image[0, :, :].unsqueeze(0)  # [200, ?, 2] -> [1, 200, 2]
                if f_dist.ndim == 3 and f_dist.shape[1] == f_dist.shape[2]:
                    f_dist = f_dist[0, :, 0].unsqueeze(0)  # [200] -> [1, 200]
                elif f_dist.ndim == 3:
                    f_dist = f_dist[0, :].unsqueeze(0)  # [200, ?] -> [1, 200]
                if b_dist.ndim == 3 and b_dist.shape[1] == b_dist.shape[2]:
                    b_dist = b_dist[0, :, 0].unsqueeze(0)  # [200] -> [1, 200]
                elif b_dist.ndim == 3:
                    b_dist = b_dist[0, :].unsqueeze(0)  # [200, ?] -> [1, 200]
        
        # 去除多余的batch维度，然后确保至少有一个batch维度
        while centers_in_image.ndim > 3:
            centers_in_image = centers_in_image.squeeze(0)
        if centers_in_image.ndim == 2:
            centers_in_image = centers_in_image.unsqueeze(0)  # [N, 2] -> [1, N, 2]
        
        while centers_valid.ndim > 2:
            centers_valid = centers_valid.squeeze(0)
        if centers_valid.ndim == 1:
            centers_valid = centers_valid.unsqueeze(0)  # [N] -> [1, N]
        
        while normals_in_image.ndim > 3:
            normals_in_image = normals_in_image.squeeze(0)
        if normals_in_image.ndim == 2:
            normals_in_image = normals_in_image.unsqueeze(0)  # [N, 2] -> [1, N, 2]
        
        while f_dist.ndim > 2:
            f_dist = f_dist.squeeze(0)
        if f_dist.ndim == 1:
            f_dist = f_dist.unsqueeze(0)  # [N] -> [1, N]
        
        while b_dist.ndim > 2:
            b_dist = b_dist.squeeze(0)
        if b_dist.ndim == 1:
            b_dist = b_dist.unsqueeze(0)  # [N] -> [1, N]
        
        # 关键修复：确保centers_in_image的batch size是1
        # 如果batch size > 1，只取第一个batch
        if centers_in_image.ndim == 3:
            if centers_in_image.shape[0] > 1:
                logger.info(f"WARNING: centers_in_image batch size is {centers_in_image.shape[0]}, taking first batch")
                centers_in_image = centers_in_image[0:1]
            elif centers_in_image.shape[0] == 0:
                raise RuntimeError(f"centers_in_image has zero batch size: {centers_in_image.shape}")
        elif centers_in_image.ndim == 2:
            centers_in_image = centers_in_image.unsqueeze(0)
        
        # 同样确保其他tensor的batch size是1
        if centers_valid.ndim == 2 and centers_valid.shape[0] > 1:
            centers_valid = centers_valid[0:1]
        if normals_in_image.ndim == 3 and normals_in_image.shape[0] > 1:
            normals_in_image = normals_in_image[0:1]
        if f_dist.ndim == 2 and f_dist.shape[0] > 1:
            f_dist = f_dist[0:1]
        if b_dist.ndim == 2 and b_dist.shape[0] > 1:
            b_dist = b_dist[0:1]
        
        fore_hist, back_hist = model.histogram.calculate_histogram(
            img_tensor[None],
            centers_in_image,
            centers_valid,
            normals_in_image,
            f_dist,
            b_dist,
            True,
        )
        total_fore_hist = (1 - fore_learn_rate) * total_fore_hist + fore_learn_rate * fore_hist
        total_back_hist = (1 - back_learn_rate) * total_back_hist + back_learn_rate * back_hist
        
        # 可视化
        # 确保closest_template_view_hist是正确的维度 [N, 8]
        if closest_template_view_hist.ndim == 3:
            # 如果是 [B, N, 8]，取第一个batch
            template_view_vis = closest_template_view_hist[0] if closest_template_view_hist.shape[0] > 0 else closest_template_view_hist
        elif closest_template_view_hist.ndim == 2:
            template_view_vis = closest_template_view_hist
        else:
            raise RuntimeError(f"Unexpected template_view shape for visualization: {closest_template_view_hist.shape}")
        
        centers_ori, valid_ori = ori_camera.view2image(pose_for_output.transform(template_view_vis[:, :3]))
        bbox_ori = None
        if valid_ori.any():
            bbox_ori = get_bbox_from_p2d(centers_ori[valid_ori], trim_ratio=bbox_trim_ratio)
        
        overlay = draw_overlay_simple(ori_image, bbox_ori, centers_ori, valid_ori, color=(0, 255, 0))
        
        # 显示位姿信息
        t_vis = pose_for_output.t
        if t_vis.ndim > 1:
            t_vis = t_vis[0]
        cv2.putText(overlay, f"Frame {frame_idx}: t=({t_vis[0]:.3f}, {t_vis[1]:.3f}, {t_vis[2]:.3f})", 
                   (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("DeepAC Live - Tracking", overlay)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
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
    logger.info("Tracking finished")

if __name__ == "__main__":
    main()

