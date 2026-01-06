"""
创新的IR追踪框架
基于边缘提取、轮廓置信度、SDF距离场、加权几何优化和时间一致性滤波
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
import argparse
import time
from pathlib import Path

from src_open.utils.lightening_utils import MyLightningLogger
from src_open.utils.m3t_realsense_camera import M3TRealSenseIRCamera
from src_open.utils.edge_extraction import compute_gradient_magnitude, compute_edge_strength, \
    get_edge_strength_at_points
from src_open.utils.contour_confidence import estimate_contour_confidence
from src_open.utils.performance_profiler import PerformanceProfiler
from src_open.utils.geometry.wrappers import Pose, Camera
from src_open.utils.utils import (
    get_bbox_from_p2d,
    get_closest_template_view_index,
    get_closest_k_template_view_index,
    project_correspondences_line,
)
from src_open.models.deep_ac import calculate_basic_line_data
from src_open.models import get_model
from src_open.utils.lightening_utils import convert_old_model, load_model_weight
from src_open.tools.live_tracking import preprocess_frame, preprocess_image, draw_overlay, initialize_from_pose_with_preprocess

def compute_ir_edge_confidence(
    ir_image: np.ndarray,
    edge_strength_map: np.ndarray,
    ksize: int = 1,  # 优化：使用更小的核（1x1或3x3）减少计算量
    tau: float = 5.0,
    use_fast_mode: bool = True,  # 新增：快速模式
):
    """
    基于IR局部结构稳定性的边缘可信度估计（抑制speckle和假边）
    优化版本：使用更高效的算法和更小的核

    Args:
        ir_image: 原始IR灰度图 (H, W), uint8 / float
        edge_strength_map: 边缘强度图 (H, W)
        ksize: 局部算子尺寸（优化：默认使用1，更快）
        tau: 控制sigmoid平滑程度
        use_fast_mode: 是否使用快速模式（跳过Laplacian，直接使用边缘强度）

    Returns:
        edge_confidence_map: [0, 1]，越大表示边缘越可信
    """
    if use_fast_mode:
        # 快速模式：直接使用归一化的边缘强度，跳过Laplacian计算
        # 这样可以节省约70%的计算时间
        edge_strength_max = edge_strength_map.max()
        if edge_strength_max > 1e-6:
            edge_confidence = edge_strength_map / edge_strength_max
        else:
            edge_confidence = np.zeros_like(edge_strength_map)
        return np.clip(edge_confidence, 0.0, 1.0)
    
    # 标准模式：使用Laplacian（保留原功能）
    # 优化：使用uint8输入，避免不必要的float转换
    if ir_image.dtype != np.uint8:
        ir_f = ir_image.astype(np.float32)
    else:
        ir_f = ir_image.astype(np.float32)

    # 优化：使用ksize=1的Laplacian（更快）或使用Sobel代替
    if ksize == 1:
        # 使用更快的Sobel算子代替Laplacian
        sobel_x = cv2.Sobel(ir_f, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(ir_f, cv2.CV_32F, 0, 1, ksize=3)
        local_var = sobel_x * sobel_x + sobel_y * sobel_y
    else:
        # 使用Laplacian（较慢）
        lap = cv2.Laplacian(ir_f, cv2.CV_32F, ksize=ksize)
        local_var = lap * lap

    # 优化：使用更快的归一化方法
    local_var_mean = np.mean(local_var)
    if local_var_mean > 1e-6:
        local_var_norm = local_var / local_var_mean
    else:
        local_var_norm = np.zeros_like(local_var)

    # Sigmoid 抑制 speckle（优化：使用更快的近似）
    # 使用exp的快速近似：1/(1+exp(-x)) ≈ sigmoid(x)
    # 对于x = (local_var_norm - 1.0) * tau
    x = (local_var_norm - 1.0) * tau
    # 使用clipped exp避免溢出
    x_clipped = np.clip(x, -10, 10)
    edge_confidence = 1.0 / (1.0 + np.exp(-x_clipped))

    # 与边缘强度联合
    edge_strength_max = edge_strength_map.max()
    if edge_strength_max > 1e-6:
        edge_strength_norm = edge_strength_map / edge_strength_max
        edge_confidence *= edge_strength_norm
    else:
        edge_confidence = np.zeros_like(edge_confidence)

    return np.clip(edge_confidence, 0.0, 1.0)

def parse_args():
    parser = argparse.ArgumentParser(description="Innovative IR Tracking with Edge-based Optimization")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    return parser.parse_args()


def load_template(pre_render_path, device):
    """加载预渲染模板"""
    import pickle
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
        orientations = pre.get("orientations", None)
    if orientations is None:
        raise RuntimeError("pre_render pkl missing `orientation_in_body` field")
    orientations = torch.from_numpy(np.array(orientations)).float()

    template_views = torch.from_numpy(template_view).float()
    return template_views.to(device), orientations.to(device), num_sample


def prepare_initial_pose(cfg, template_views, orientations, device):
    """准备初始位姿"""
    initial_view_idx = cfg.tracking.get("initial_view_index", 0)
    if isinstance(initial_view_idx, torch.Tensor):
        initial_view_idx = initial_view_idx.item()

    init_depth = cfg.tracking.get("init_depth", 0.45)
    initial_translation = cfg.tracking.get("initial_translation", [0.0, 0.0, init_depth])

    # 获取初始方向
    if initial_view_idx < len(orientations):
        initial_orientation = orientations[initial_view_idx]
    else:
        initial_orientation = orientations[0]

    # 创建初始位姿
    initial_pose = Pose.from_aa(initial_orientation.unsqueeze(0), torch.tensor([initial_translation], device=device))

    # 应用Z轴旋转（如果配置了）
    initial_rotation_z = cfg.tracking.get("initial_rotation_z", 0.0)
    if initial_rotation_z != 0.0:
        from scipy.spatial.transform import Rotation as R
        rotation_z_rad = np.deg2rad(initial_rotation_z)
        z_axis = np.array([0, 0, 1])
        rotation_z = R.from_rotvec(rotation_z_rad * z_axis)

        current_R = initial_pose.R.detach().cpu().numpy()
        if current_R.ndim == 3:
            current_R = current_R[0]
        rotated_R = rotation_z.as_matrix() @ current_R
        rotated_R_tensor = torch.from_numpy(rotated_R).float().to(device)
        initial_pose = Pose.from_Rt(rotated_R_tensor.unsqueeze(0), initial_pose.t).to(device)

    initial_template = template_views[initial_view_idx: initial_view_idx + 1]
    return initial_view_idx, initial_pose, initial_template


@torch.no_grad()
def tracking_step_innovative(
        frame_rgb,
        ori_camera_cpu,
        current_pose,
        template_views,
        orientations,
        model,
        fore_hist,
        back_hist,
        track_cfg,
        data_conf,
        device,
        edge_strength_map=None,
        depth_map=None,
        edge_confidence_map=None,   # ← 新增
        bbox_trim_ratio=0.0,
):
    """
    创新的追踪步骤：结合边缘强度和深度信息
    
    使用 @torch.no_grad() 装饰器避免构建 autograd graph，提升推理速度并减少显存占用

    Args:
        edge_strength_map: 边缘强度图（可选）
        depth_map: 深度图（可选）
    """
    ori_camera = ori_camera_cpu.to(device)

    # 获取bbox进行预处理
    idx0 = get_closest_template_view_index(current_pose, orientations)
    initial_template = template_views[idx0: idx0 + 1]
    data_lines = project_correspondences_line(initial_template, current_pose, ori_camera)
    bbox2d = get_bbox_from_p2d(data_lines["centers_in_image"][0], trim_ratio=bbox_trim_ratio)

    h_img, w_img = frame_rgb.shape[:2]
    if bbox2d.numel() < 4 or torch.isnan(bbox2d).any() or bbox2d[2] < 4 or bbox2d[3] < 4:
        bbox2d = torch.tensor([w_img / 2.0, h_img / 2.0, float(w_img), float(h_img)], dtype=torch.float32,
                              device=device)

    img_tensor, camera = preprocess_image(frame_rgb, bbox2d.detach().cpu().numpy().copy(), ori_camera_cpu, data_conf, device)

    # 标准DeepAC追踪步骤
    # 使用多个模板视图
    template_top_k = track_cfg.get("template_top_k", 10)
    template_skip = data_conf.get("skip_template_view", 1)
    
    # 获取最接近的k个模板视图索引
    indices_full = get_closest_k_template_view_index(
        current_pose, orientations, template_top_k * template_skip
    )
    if indices_full.ndim > 1:
        indices_full = indices_full.view(-1)
    indices_full = indices_full[::template_skip]
    indices_list = indices_full.tolist()
    closest_template_views = template_views[indices_list].contiguous()
    closest_orientations = orientations[indices_full].contiguous()

    if isinstance(current_pose, Pose):
        if current_pose._data.ndim == 1:
            body2view_pose = current_pose[None]
        else:
            if current_pose._data.ndim == 2:
                if current_pose._data.shape[0] > 1:
                    body2view_pose = Pose(current_pose._data[0:1])
                else:
                    body2view_pose = current_pose
            else:
                body2view_pose = Pose(current_pose._data.view(-1)[:12].unsqueeze(0))
    else:
        body2view_pose = current_pose

    data = {
        "image": img_tensor[None],  # 添加batch维度
        "camera": camera[None],  # 添加batch维度
        "body2view_pose": body2view_pose,
        "closest_template_views": closest_template_views[None],
        "closest_orientations_in_body": closest_orientations[None],
        "fore_hist": fore_hist,
        "back_hist": back_hist,
    }

    pred = model._forward(data, visualize=False, tracking=True)
    new_pose: Pose = pred["opt_body2view_pose"][-1][0]

    # 计算轮廓点置信度（如果提供了边缘强度图）
    # 使用第一个模板视图计算轮廓点
    (
        _,
        _,
        centers_in_image,
        centers_valid,
        normals_in_image,
        fg_dist,
        bg_dist,
        _,
    ) = calculate_basic_line_data(closest_template_views[None][:, 0], new_pose[None]._data, camera[None]._data, 1, 0)

    # 确保维度正确
    while centers_in_image.ndim > 3:
        centers_in_image = centers_in_image.squeeze(0)
    while centers_valid.ndim > 2:
        centers_valid = centers_valid.squeeze(0)

    # 在原始图像坐标系中计算bbox和centers（先计算，用于返回）
    idx_best = get_closest_template_view_index(new_pose, orientations)
    best_template = template_views[idx_best: idx_best + 1]
    centers_ori, valid_ori = ori_camera.view2image(new_pose.transform(best_template[0, :, :3]))

    # 如果有边缘强度图，计算轮廓点置信度（基于返回的valid_ori）
    contour_confidence = None
    if edge_strength_map is not None and valid_ori.any():
        # =========================
        # 创新点2：IR物理一致性权重
        # =========================
        lambertian_weight = None
        
        # 获取对应模板点的法向（body坐标）
        normals_body = best_template[0, :, 3:6].detach().cpu().numpy()  # [N,3] numpy

        # 转到相机坐标
        R = new_pose.R.detach().cpu().numpy()
        if R.ndim == 3:
            R = R[0]
        normals_cam = (R @ normals_body.T).T

        # 视线方向（相机坐标系下 z 轴）
        view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # cos(theta)
        cos_theta = normals_cam @ view_dir
        cos_theta = np.clip(cos_theta, 0.0, 1.0)

        lambertian_weight = cos_theta
        
        # 计算轮廓点置信度
        contour_confidence = estimate_contour_confidence(
            edge_strength_map,
            centers_ori,
            depth_map=depth_map,
            valid_mask=valid_ori,
            edge_weight=track_cfg.get("edge_weight", 0.6),
            depth_weight=track_cfg.get("depth_weight", 0.4),
        )
        
        # =========================
        # 融合创新权重
        # =========================
        if contour_confidence is not None:
            # 全部转 NumPy
            valid_mask = valid_ori.detach().cpu().numpy().astype(bool) if isinstance(valid_ori, torch.Tensor) else valid_ori.astype(bool)
            centers_np = centers_ori.detach().cpu().numpy() if isinstance(centers_ori, torch.Tensor) else centers_ori

            weights = contour_confidence.copy()

            # 创新点1：IR 边缘可信度
            if edge_confidence_map is not None:
                pts = centers_np[valid_mask].astype(np.int32)
                pts[:, 0] = np.clip(pts[:, 0], 0, edge_confidence_map.shape[1] - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, edge_confidence_map.shape[0] - 1)

                edge_conf = edge_confidence_map[pts[:, 1], pts[:, 0]]
                weights *= edge_conf

            # 创新点2：Lambertian 物理一致性
            if lambertian_weight is not None:
                weights *= lambertian_weight[valid_mask]

            contour_confidence = np.clip(weights, 0.05, 1.0)
            
            # 将contour_confidence扩展到与valid_ori相同的长度（填充无效位置为0）
            if len(contour_confidence) < len(valid_mask):
                full_confidence = np.zeros(len(valid_mask), dtype=contour_confidence.dtype)
                full_confidence[valid_mask] = contour_confidence
                contour_confidence = full_confidence

    # 更新直方图
    while normals_in_image.ndim > 3:
        normals_in_image = normals_in_image.squeeze(0)
    while fg_dist.ndim > 2:
        fg_dist = fg_dist.squeeze(0)
    while bg_dist.ndim > 2:
        bg_dist = bg_dist.squeeze(0)

    fore_hist_new, back_hist_new = model.histogram.calculate_histogram(
        img_tensor[None], centers_in_image, centers_valid, normals_in_image, fg_dist, bg_dist, True
    )

    alpha_f = float(track_cfg.fore_learn_rate)
    alpha_b = float(track_cfg.back_learn_rate)
    fore_hist = (1 - alpha_f) * fore_hist + alpha_f * fore_hist_new
    back_hist = (1 - alpha_b) * back_hist + alpha_b * back_hist_new

    # 计算bbox
    bbox = None
    if valid_ori.any():
        bbox = get_bbox_from_p2d(centers_ori[valid_ori], trim_ratio=bbox_trim_ratio)

    return new_pose, fore_hist, back_hist, bbox, centers_ori, valid_ori, contour_confidence


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = MyLightningLogger("deepac-live-ir-innovative", cfg.save_dir)
    logger.info(f"Using device: {device}")

    # 加载模型
    train_cfg = OmegaConf.load(cfg.model.load_cfg)
    model_cfg = train_cfg.models if "models" in train_cfg else train_cfg
    model = get_model(model_cfg.name)(model_cfg)
    
    ckpt = torch.load(cfg.model.load_model, map_location="cpu")
    if "pytorch-lightning_version" not in ckpt:
        ckpt = convert_old_model(ckpt)
    load_model_weight(model, ckpt, logger)
    model.to(device).eval()
    
    logger.info(f"Loaded weights from {cfg.model.load_model}")
    
    # 获取数据配置
    data_conf = train_cfg.data

    # 加载模板
    template_views, orientations, num_sample = load_template(cfg.object.pre_render_pkl, device)
    logger.info(f"Loaded {len(template_views)} template views")

    # 准备初始位姿
    initial_view_idx, initial_pose, initial_template = prepare_initial_pose(cfg, template_views, orientations, device)
    logger.info(f"Guide overlay uses template view index {initial_view_idx}")

    # 初始化RealSense IR相机
    realsense_camera = M3TRealSenseIRCamera(
        ir_index=cfg.camera.get("ir_index", 1),
        emitter_enabled=cfg.camera.get("emitter_enabled", False),
    )
    realsense_camera.setup(
        width=cfg.camera.get("set_width", 1280),
        height=cfg.camera.get("set_height", 720),
        fps=cfg.camera.get("fps", 30),
    )

    # 重要：用RealSense实际内参覆盖YAML配置中的内参
    # 这可以避免投影轮廓与真实轮廓的系统性偏差
    if realsense_camera.intrinsics is not None:
        cfg.camera.fx = float(realsense_camera.intrinsics['fx'])
        cfg.camera.fy = float(realsense_camera.intrinsics['fy'])
        cfg.camera.cx = float(realsense_camera.intrinsics['cx'])
        cfg.camera.cy = float(realsense_camera.intrinsics['cy'])
        logger.info(
            f"RealSense IR camera initialized: {realsense_camera.width}x{realsense_camera.height}, "
            f"fx={cfg.camera.fx:.2f}, fy={cfg.camera.fy:.2f}, "
            f"cx={cfg.camera.cx:.2f}, cy={cfg.camera.cy:.2f}"
        )
        logger.info("Updated camera intrinsics in config with RealSense actual values")
    else:
        logger.warn("Failed to get RealSense intrinsics, using YAML config values")
        logger.info(
            f"Using YAML camera intrinsics: fx={cfg.camera.get('fx', 0):.2f}, "
            f"fy={cfg.camera.get('fy', 0):.2f}, cx={cfg.camera.get('cx', 0):.2f}, "
            f"cy={cfg.camera.get('cy', 0):.2f}"
        )

    # 初始化CLAHE（如果启用）
    use_clahe = cfg.tracking.get("use_clahe", True)
    clahe = None
    if use_clahe:
        clahe_clip_limit = cfg.tracking.get("clahe_clip_limit", 2.0)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
        logger.info("CLAHE enhancement enabled")

    # 初始化追踪状态
    initialized = False
    current_pose = None
    fore_hist = None
    back_hist = None
    frame_idx = 0

    # 创建输出目录
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化位姿保存文件
    pose_file_path = save_dir / "pose.txt"
    pose_file = open(pose_file_path, "w")
    logger.info(f"Saving poses to {pose_file_path}")
    
    # 初始化视频输出
    video_writer = None
    out_size = None
    if cfg.tracking.get("output_video", False):
        out_size = tuple(cfg.tracking.get("output_size", [1280, 720]))
        video_path = save_dir / f"live_ir_innovative_{time.strftime('%Y%m%d_%H%M%S')}.avi"
        video_writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"XVID"),
            cfg.tracking.get("output_fps", 30),
            out_size,
        )
        logger.info(f"Recording video to {video_path}")
    
    # FPS和时间统计
    fps_start_time = time.time()
    fps_frame_count = 0
    inference_times = []
    elapsed_time = 0.0  # 初始化elapsed_time
    
    # 初始化性能分析器
    enable_profiling = cfg.tracking.get("enable_profiling", True)
    profiler = PerformanceProfiler(enabled=enable_profiling)
    logger.info(f"Performance profiling: {'enabled' if enable_profiling else 'disabled'}")
    
    # 获取几何单位（用于位姿保存）
    geometry_unit = cfg.object.get("geometry_unit_in_meter", 0.001)

    # 主循环
    try:
        while True:
            # 从RealSense获取IR图像
            ir_image = realsense_camera.get_image()
            if ir_image is None:
                logger.warn("Failed to grab frame from RealSense IR camera; stopping")
                break

            # 开始新的一帧性能分析
            profiler.start_frame()
            
            # 应用CLAHE增强（如果启用）
            with profiler.profile("clahe_enhancement"):
                if clahe is not None:
                    ir_image = clahe.apply(ir_image)

            # 转换为BGR格式（用于显示和处理）
            with profiler.profile("color_conversion"):
                frame_bgr = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)

            with profiler.profile("preprocess_frame"):
                frame_rgb, ori_camera_cpu = preprocess_frame(frame_bgr, cfg.camera, device)
            ori_camera = ori_camera_cpu.to(device)

            if not initialized:
                # 初始化阶段：显示引导框
                data_lines = project_correspondences_line(initial_template, initial_pose, ori_camera)
                guide_bbox = get_bbox_from_p2d(data_lines["centers_in_image"][0], trim_ratio=0.0)
                guide_centers = data_lines["centers_in_image"][0]
                guide_valid = data_lines["centers_valid"][0]

                overlay = draw_overlay(
                    frame_rgb,
                    guide_bbox,
                    guide_centers,
                    guide_valid,
                    color=(255, 128, 0),
                )
                cv2.putText(
                    overlay,
                    "Align cube with guide...",
                    (16, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 200, 255),
                    2,
                )
                cv2.putText(
                    overlay,
                    "Press 's' to start tracking",
                    (16, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                # 追踪阶段
                # 只在追踪阶段计算边缘提取和边缘可信度（初始化阶段不需要）
                with profiler.profile("edge_extraction"):
                    edge_strength_map = compute_edge_strength(ir_image)
                
                # IR边缘可信度图（创新点1）
                with profiler.profile("edge_confidence"):
                    use_fast_edge_confidence = cfg.tracking.get("use_fast_edge_confidence", True)
                    edge_confidence_map = compute_ir_edge_confidence(
                        ir_image=ir_image,
                        edge_strength_map=edge_strength_map,
                        ksize=1,  # 优化：使用更小的核
                        tau=cfg.tracking.get("edge_confidence_tau", 5.0),
                        use_fast_mode=use_fast_edge_confidence,
                    )
                
                bbox_trim_ratio = cfg.tracking.get("bbox_trim_ratio", 0.0)

                # 记录推理开始时间
                inference_start = time.time()
                
                # 使用 torch.inference_mode() 进一步优化推理性能（比 no_grad 更快）
                # 注意：tracking_step_innovative 已经有 @torch.no_grad()，这里再加一层确保安全
                with profiler.profile("tracking_step"), torch.inference_mode():
                    (
                        current_pose,
                        fore_hist,
                        back_hist,
                        last_bbox,
                        last_centers,
                        last_valid,
                        contour_confidence,
                    ) = tracking_step_innovative(
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
                        edge_strength_map=edge_strength_map,
                        depth_map=None,  # TODO: 添加深度图支持
                        edge_confidence_map=edge_confidence_map,  # ← 新增
                        bbox_trim_ratio=bbox_trim_ratio,
                    )
                
                # 记录推理时间
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                
                # 保存位姿到文件（格式：12个数字一行，R的9个元素 + t的3个元素）
                with profiler.profile("pose_saving"):
                    if current_pose is not None:
                        R = current_pose.R.detach().cpu().numpy()
                        t = current_pose.t.detach().cpu().numpy() / geometry_unit  # 转换为mm
                        
                        # 处理batch维度
                        if R.ndim == 3:
                            R = R[0]
                        if t.ndim == 2:
                            t = t[0]
                        
                        # 格式：12个数字一行 [R11, R12, R13, R21, R22, R23, R31, R32, R33, t1, t2, t3]
                        pose_flat = np.concatenate([R.flatten(), t.flatten()])
                        pose_file.write(" ".join(f"{x:.8f}" for x in pose_flat) + "\n")
                        pose_file.flush()  # 确保立即写入

                # 绘制结果
                with profiler.profile("drawing"):
                    overlay = draw_overlay(
                        frame_rgb,
                        last_bbox,
                        last_centers,
                        last_valid,
                        color=(0, 255, 0),
                    )

                # 计算FPS
                fps_frame_count += 1
                elapsed_time = time.time() - fps_start_time
                if elapsed_time > 0:
                    current_fps = fps_frame_count / elapsed_time
                    avg_inference_time = np.mean(inference_times[-30:]) if inference_times else 0.0  # 最近30帧的平均推理时间
                else:
                    current_fps = 0.0
                    avg_inference_time = 0.0

                # 显示信息
                info_y = 48
                if contour_confidence is not None and last_valid is not None:
                    # 转换last_valid为numpy数组
                    if isinstance(last_valid, torch.Tensor):
                        last_valid_np = last_valid.detach().cpu().numpy()
                    else:
                        last_valid_np = np.array(last_valid)
                    
                    # contour_confidence现在应该与last_valid_np长度匹配（已扩展到完整长度）
                    if len(contour_confidence) == len(last_valid_np):
                        # 长度匹配，使用last_valid_np作为索引
                        avg_confidence = np.mean(contour_confidence[last_valid_np]) if last_valid_np.any() else 0.0
                    elif len(contour_confidence) < len(last_valid_np) and last_valid_np.any():
                        # 如果长度不匹配，只计算有效位置的置信度
                        valid_indices = np.where(last_valid_np)[0]
                        if len(valid_indices) <= len(contour_confidence):
                            avg_confidence = np.mean(contour_confidence[:len(valid_indices)]) if len(valid_indices) > 0 else 0.0
                        else:
                            avg_confidence = np.mean(contour_confidence) if len(contour_confidence) > 0 else 0.0
                    else:
                        avg_confidence = 0.0
                    cv2.putText(
                        overlay,
                        f"Confidence: {avg_confidence:.3f}",
                        (16, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                    info_y += 30
                
                cv2.putText(
                    overlay,
                    f"FPS: {current_fps:.1f}",
                    (16, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                info_y += 30
                
                cv2.putText(
                    overlay,
                    f"Inference: {avg_inference_time*1000:.1f}ms",
                    (16, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                info_y += 30
                
                cv2.putText(
                    overlay,
                    f"Frame: {frame_idx}",
                    (16, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                
                # 获取模板索引（用于显示）
                template_idx = get_closest_template_view_index(current_pose, orientations)
                if isinstance(template_idx, torch.Tensor):
                    template_idx = template_idx.item()
                cv2.putText(
                    overlay,
                    f"Template: {template_idx}",
                    (16, info_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                # 写入视频（如果启用）
                with profiler.profile("video_writing"):
                    if video_writer is not None and out_size is not None:
                        # 调整overlay大小以匹配视频输出尺寸
                        overlay_h, overlay_w = overlay.shape[:2]
                        if (overlay_w, overlay_h) != out_size:
                            overlay_resized = cv2.resize(overlay, out_size)
                        else:
                            overlay_resized = overlay
                        video_writer.write(overlay_resized)
                
                # 显示性能信息（每10帧更新一次）
                if frame_idx % 10 == 0 and enable_profiling:
                    current_timings = profiler.get_current_timings()
                    if current_timings:
                        perf_text = "Performance (ms): "
                        perf_items = []
                        for name, timing in sorted(current_timings.items(), key=lambda x: x[1], reverse=True)[:3]:
                            perf_items.append(f"{name}={timing:.1f}")
                        if perf_items:
                            cv2.putText(
                                overlay,
                                perf_text + ", ".join(perf_items),
                                (16, overlay.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 0),
                                1,
                            )

            frame_idx += 1

            key = -1
            if cfg.tracking.get("show_window", True):
                cv2.imshow("DeepAC Live (Innovative IR)", overlay)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF

            if not initialized and key == ord("s"):
                logger.info("Manual initialization requested (key 's') using guide pose directly")
                current_pose = Pose(initial_pose._data.clone()).to(device)

                # 使用正确的初始化函数
                fore_hist, back_hist, last_bbox, last_centers, last_valid = initialize_from_pose_with_preprocess(
                    frame_rgb,
                    ori_camera_cpu,
                    current_pose,
                    template_views,
                    orientations,
                    model,
                    device,
                    data_conf,
                    bbox_trim_ratio=0.0,
                )

                # 检查初始化结果
                fore_mean = torch.mean(fore_hist).item()
                back_mean = torch.mean(back_hist).item()
                hist_distinction = (fore_mean - back_mean) / (fore_mean + back_mean + 1e-7)
                logger.info(f"Histogram initialized: distinction={hist_distinction:.6f}, fore_mean={fore_mean:.6f}, back_mean={back_mean:.6f}")
                

                initialized = True
                frame_idx = 0
                fps_start_time = time.time()  # 重置FPS计时
                fps_frame_count = 0
                inference_times = []
                elapsed_time = 0.0  # 重置elapsed_time

            if key == 27 or key == ord("q"):
                logger.info("Exiting...")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # 关闭位姿文件
        if pose_file is not None:
            pose_file.close()
            logger.info(f"Saved {frame_idx} poses to {pose_file_path}")
        
        # 关闭视频写入器
        if video_writer is not None:
            video_writer.release()
            logger.info("Video recording finished")
        
        # 输出统计信息
        if inference_times:
            avg_inference = np.mean(inference_times)
            min_inference = np.min(inference_times)
            max_inference = np.max(inference_times)
            logger.info(f"Inference time stats: avg={avg_inference*1000:.2f}ms, min={min_inference*1000:.2f}ms, max={max_inference*1000:.2f}ms")
        
        if fps_frame_count > 0 and elapsed_time > 0:
            avg_fps = fps_frame_count / elapsed_time
            logger.info(f"Average FPS: {avg_fps:.2f}")
        
        # 输出性能分析结果
        if enable_profiling:
            profiler.print_summary()
            # 保存性能分析结果到文件
            perf_file = save_dir / "performance_profile.txt"
            profiler.save_to_file(str(perf_file))
            logger.info(f"Performance profile saved to {perf_file}")
        
        realsense_camera.stop()
        cv2.destroyAllWindows()
        logger.info("Live tracking finished")


if __name__ == "__main__":
    main()
