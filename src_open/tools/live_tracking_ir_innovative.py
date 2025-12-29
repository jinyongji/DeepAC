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
        bbox_trim_ratio=0.0,
):
    """
    创新的追踪步骤：结合边缘强度和深度信息

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

    # 如果有边缘强度图，计算轮廓点置信度
    contour_confidence = None
    if edge_strength_map is not None:
        # 在原始图像坐标系中计算轮廓点
        idx_best = get_closest_template_view_index(new_pose, orientations)
        best_template = template_views[idx_best: idx_best + 1]
        centers_ori, valid_ori = ori_camera.view2image(new_pose.transform(best_template[0, :, :3]))

        if valid_ori.any():
            contour_confidence = estimate_contour_confidence(
                edge_strength_map,
                centers_ori,
                depth_map=depth_map,
                valid_mask=valid_ori,
                edge_weight=track_cfg.get("edge_weight", 0.6),
                depth_weight=track_cfg.get("depth_weight", 0.4),
            )

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

    # 在原始图像坐标系中计算bbox和centers
    idx_best = get_closest_template_view_index(new_pose, orientations)
    best_template = template_views[idx_best: idx_best + 1]
    centers_ori, valid_ori = ori_camera.view2image(new_pose.transform(best_template[0, :, :3]))

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

    logger.info(
        f"RealSense IR camera initialized: {realsense_camera.width}x{realsense_camera.height}, "
        f"fx={realsense_camera.intrinsics['fx']:.2f}, fy={realsense_camera.intrinsics['fy']:.2f}, "
        f"cx={realsense_camera.intrinsics['cx']:.2f}, cy={realsense_camera.intrinsics['cy']:.2f}"
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

            # 应用CLAHE增强（如果启用）
            if clahe is not None:
                ir_image = clahe.apply(ir_image)

            # 转换为BGR格式（用于显示和处理）
            frame_bgr = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)

            # 计算边缘强度图
            edge_strength_map = compute_edge_strength(ir_image)

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
                bbox_trim_ratio = cfg.tracking.get("bbox_trim_ratio", 0.0)

                # 记录推理开始时间
                inference_start = time.time()
                
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
                    bbox_trim_ratio=bbox_trim_ratio,
                )
                
                # 记录推理时间
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                
                # 保存位姿到文件（格式：12个数字一行，R的9个元素 + t的3个元素）
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
                    avg_confidence = np.mean(contour_confidence[last_valid_np]) if last_valid_np.any() else 0.0
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
                if video_writer is not None and out_size is not None:
                    # 调整overlay大小以匹配视频输出尺寸
                    overlay_h, overlay_w = overlay.shape[:2]
                    if (overlay_w, overlay_h) != out_size:
                        overlay_resized = cv2.resize(overlay, out_size)
                    else:
                        overlay_resized = overlay
                    video_writer.write(overlay_resized)

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
        
        realsense_camera.stop()
        cv2.destroyAllWindows()
        logger.info("Live tracking finished")


if __name__ == "__main__":
    main()
