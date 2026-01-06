"""
评估IR追踪算法在5cm/5度阈值下的准确率
对IR图像序列运行追踪算法，并与GT位姿对比
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import argparse
from omegaconf import OmegaConf
import json
import cv2
import torch
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src_open.utils.lightening_utils import MyLightningLogger
from src_open.utils.edge_extraction import compute_edge_strength
from src_open.utils.geometry.wrappers import Pose
from src_open.models import get_model
from src_open.utils.lightening_utils import convert_old_model, load_model_weight
from src_open.tools.live_tracking import preprocess_frame, initialize_from_pose_with_preprocess, draw_overlay, tensor_bbox_to_xyxy
from src_open.tools.live_tracking_realsense import (
    tracking_step_innovative,
    compute_ir_edge_confidence,
    load_template,
    prepare_initial_pose,
)
from src_open.utils.utils import project_correspondences_line, get_bbox_from_p2d, get_closest_template_view_index
from src_open.utils.m3t_realsense_camera import M3TRealSenseIRCamera


def parse_gt_pose_file(gt_pose_file: Path) -> Dict[int, Dict[str, np.ndarray]]:
    """
    解析GT位姿文件
    
    格式：frame_index timestamp tx ty tz r00 r01 r02 r10 r11 r12 r20 r21 r22
    
    Args:
        gt_pose_file: GT位姿文件路径
    
    Returns:
        字典，key为帧索引，value为包含'R'和't'的字典
    """
    gt_poses = {}
    
    with open(gt_pose_file, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 13:  # frame_idx + timestamp + 3 translation + 9 rotation
                continue
            
            try:
                frame_idx = int(parts[0])
                # 跳过timestamp (parts[1])
                tx, ty, tz = float(parts[2]), float(parts[3]), float(parts[4])
                # 旋转矩阵的9个元素
                R_flat = [float(x) for x in parts[5:14]]
                
                # 构建旋转矩阵 (3x3)
                R = np.array(R_flat).reshape(3, 3)
                # 平移向量
                t = np.array([tx, ty, tz])
                
                # 检查旋转矩阵是否有效（对于第一帧）
                if frame_idx == 0:
                    print(f"DEBUG: Frame 0 GT pose - Raw rotation matrix values:")
                    print(f"  R_flat: {R_flat}")
                    print(f"  R matrix:\n{R}")
                    print(f"  R determinant: {np.linalg.det(R):.6f}")
                    print(f"  R trace: {np.trace(R):.6f}")
                    # 检查是否是有效的旋转矩阵（行列式应该接近1，trace应该接近3）
                    if abs(np.linalg.det(R)) < 0.1:
                        print(f"  WARNING: Rotation matrix determinant is too small ({np.linalg.det(R):.6f})!")
                        print(f"  This suggests the rotation matrix format may be incorrect.")
                        print(f"  Expected: valid 3x3 rotation matrix (determinant ≈ 1)")
                        print(f"  Got: matrix with mostly zeros or very small values")
                
                gt_poses[frame_idx] = {'R': R, 't': t}
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse line: {line[:50]}... Error: {e}")
                continue
    
    return gt_poses


def compute_add_error(
    R_pred: np.ndarray,
    t_pred: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    model_points: np.ndarray,
) -> float:
    """
    计算ADD误差（Average Distance）
    
    Args:
        R_pred: 预测的旋转矩阵 (3x3)
        t_pred: 预测的平移向量 (3,) 或 (1, 3)
        R_gt: GT旋转矩阵 (3x3)
        t_gt: GT平移向量 (3,) 或 (1, 3)
        model_points: 模型点云 (N, 3)，在物体坐标系中
    
    Returns:
        ADD误差（米）
    """
    # 确保平移向量是1D数组
    t_pred = np.asarray(t_pred).flatten()
    t_gt = np.asarray(t_gt).flatten()
    
    # 确保形状正确
    assert t_pred.shape == (3,), f"t_pred shape should be (3,), got {t_pred.shape}"
    assert t_gt.shape == (3,), f"t_gt shape should be (3,), got {t_gt.shape}"
    
    # 将模型点转换到相机坐标系
    points_pred = (R_pred @ model_points.T).T + t_pred.reshape(1, 3)
    points_gt = (R_gt @ model_points.T).T + t_gt.reshape(1, 3)
    
    # 计算每个点的距离
    distances = np.linalg.norm(points_pred - points_gt, axis=1)
    
    # 返回平均距离
    return np.mean(distances)


def compute_rotation_error(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """
    计算旋转误差（角度）
    
    Args:
        R_pred: 预测的旋转矩阵 (3x3)
        R_gt: GT旋转矩阵 (3x3)
    
    Returns:
        旋转误差（度）
    """
    # 计算相对旋转
    R_rel = R_pred @ R_gt.T
    
    # 从旋转矩阵提取角度
    trace = np.trace(R_rel)
    trace = np.clip(trace, -1.0, 3.0)
    angle_rad = np.arccos((trace - 1.0) / 2.0)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def compute_translation_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """
    计算平移误差（米）
    
    Args:
        t_pred: 预测的平移向量 (3,)
        t_gt: GT平移向量 (3,)
    
    Returns:
        平移误差（米）
    """
    return np.linalg.norm(t_pred - t_gt)


def pose_to_numpy(pose: Pose) -> Tuple[np.ndarray, np.ndarray]:
    """
    将Pose对象转换为numpy数组
    
    Args:
        pose: Pose对象
    
    Returns:
        (R, t): 旋转矩阵和平移向量
    """
    # Pose对象有.R和.t属性
    R_tensor = pose.R
    t_tensor = pose.t
    
    # 处理batch维度
    if R_tensor.ndim == 3:  # (batch, 3, 3)
        R = R_tensor[0].cpu().numpy()  # (3, 3)
    else:  # (3, 3)
        R = R_tensor.cpu().numpy()
    
    if t_tensor.ndim == 2:  # (batch, 3)
        t = t_tensor[0].cpu().numpy()  # (3,)
    else:  # (3,)
        t = t_tensor.cpu().numpy()
    
    # 确保t是1D数组
    t = t.flatten()
    
    return R, t


def evaluate_ir_tracking(
    image_dir: Path,
    gt_pose_file: Path,
    cfg_path: Path,
    output_dir: Optional[Path] = None,
    add_threshold: float = 0.05,  # 5cm
    rotation_threshold: float = 5.0,  # 5度
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    use_first_gt_as_init: bool = True,  # 使用第一帧GT位姿作为初始化
    gt_camera_intrinsics: Optional[Dict[str, float]] = None,  # GT位姿使用的相机内参
) -> Dict:
    """
    评估IR追踪算法
    
    Args:
        image_dir: IR图像目录
        gt_pose_file: GT位姿文件路径
        cfg_path: 配置文件路径
        output_dir: 输出目录（可选）
        add_threshold: ADD阈值（米），默认5cm
        rotation_threshold: 旋转阈值（度），默认5度
        start_frame: 起始帧索引
        end_frame: 结束帧索引（None表示处理所有帧）
        use_first_gt_as_init: 是否使用第一帧GT位姿作为初始化
        gt_camera_intrinsics: GT位姿使用的相机内参，格式：{'fx': float, 'fy': float, 'cx': float, 'cy': float}
                            如果为None，则使用配置文件中的内参
    
    Returns:
        评估结果字典
    """
    # 加载配置
    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = MyLightningLogger("ir_tracking_eval", str(output_dir) if output_dir else ".")
    
    logger.info(f"Using device: {device}")
    logger.info(f"Image directory: {image_dir}")
    logger.info(f"GT pose file: {gt_pose_file}")
    
    # 加载GT位姿
    logger.info("Loading GT poses...")
    gt_poses = parse_gt_pose_file(gt_pose_file)
    logger.info(f"Loaded {len(gt_poses)} GT poses")
    
    # 加载模型
    logger.info("Loading model...")
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
    logger.info("Loading templates...")
    template_views, orientations, num_sample = load_template(cfg.object.pre_render_pkl, device)
    logger.info(f"Loaded {len(template_views)} template views")
    
    # 尝试从RealSense相机获取实际内参（与M3T一致）
    # 如果指定了GT相机内参，优先使用
    if gt_camera_intrinsics is not None:
        logger.info(f"Using provided GT camera intrinsics: fx={gt_camera_intrinsics['fx']:.2f}, "
                   f"fy={gt_camera_intrinsics['fy']:.2f}, "
                   f"cx={gt_camera_intrinsics['cx']:.2f}, cy={gt_camera_intrinsics['cy']:.2f}")
        cfg.camera.fx = gt_camera_intrinsics['fx']
        cfg.camera.fy = gt_camera_intrinsics['fy']
        cfg.camera.cx = gt_camera_intrinsics['cx']
        cfg.camera.cy = gt_camera_intrinsics['cy']
    else:
        # 默认使用M3T追踪时使用的IR相机内参（640x480）
        # 这些是M3T实际使用的内参，应该与GT位姿文件一致
        logger.info("Using default M3T IR camera intrinsics (640x480)")
        cfg.camera.fx = 390.777
        cfg.camera.fy = 390.777
        cfg.camera.cx = 326.226
        cfg.camera.cy = 242.749
        logger.info(f"Default M3T intrinsics: fx={cfg.camera.fx:.2f}, fy={cfg.camera.fy:.2f}, "
                   f"cx={cfg.camera.cx:.2f}, cy={cfg.camera.cy:.2f}")
        
        # 不再尝试从RealSense相机获取内参，因为可能不匹配
        # 如果需要使用RealSense内参，请通过命令行参数指定
        # use_realsense_intrinsics = False
        # if use_realsense_intrinsics:
        #     # 尝试从RealSense相机获取内参（如果相机可用）
        #     try:
        #         logger.info("Attempting to get camera intrinsics from RealSense camera...")
        #         realsense_camera = M3TRealSenseIRCamera(
        #             ir_index=cfg.camera.get("ir_index", 1),
        #             emitter_enabled=cfg.camera.get("emitter_enabled", False),
        #         )
        #         if realsense_camera.setup(
        #             width=cfg.camera.get("set_width", 1280),
        #             height=cfg.camera.get("set_height", 720),
        #             fps=cfg.camera.get("fps", 30),
        #         ):
        #             if realsense_camera.intrinsics is not None:
        #                 cfg.camera.fx = float(realsense_camera.intrinsics['fx'])
        #                 cfg.camera.fy = float(realsense_camera.intrinsics['fy'])
        #                 cfg.camera.cx = float(realsense_camera.intrinsics['cx'])
        #                 cfg.camera.cy = float(realsense_camera.intrinsics['cy'])
        #                 logger.info(f"Got RealSense camera intrinsics: fx={cfg.camera.fx:.2f}, "
        #                            f"fy={cfg.camera.fy:.2f}, "
        #                            f"cx={cfg.camera.cx:.2f}, cy={cfg.camera.cy:.2f}")
        #                 realsense_camera.stop()
        #             else:
        #                 logger.info("Failed to get RealSense intrinsics, using config values")
        #         else:
        #             logger.info("Failed to setup RealSense camera, using config values")
        #     except Exception as e:
        #         logger.info(f"Failed to get RealSense camera intrinsics: {e}, using config values")
        
        logger.info(f"Using camera intrinsics: fx={cfg.camera.get('fx', 0):.2f}, "
                   f"fy={cfg.camera.get('fy', 0):.2f}, "
                   f"cx={cfg.camera.get('cx', 0):.2f}, cy={cfg.camera.get('cy', 0):.2f}")
    
    # 准备初始位姿
    initial_view_idx, initial_pose, initial_template = prepare_initial_pose(
        cfg, template_views, orientations, device
    )
    
    # 如果使用第一帧GT作为初始化
    if use_first_gt_as_init and 0 in gt_poses:
        gt_pose_0 = gt_poses[0]
        R_init = torch.from_numpy(gt_pose_0['R']).float().to(device).unsqueeze(0)  # (1, 3, 3)
        t_init = torch.from_numpy(gt_pose_0['t']).float().to(device).unsqueeze(0)  # (1, 3)
        initial_pose = Pose.from_Rt(R_init, t_init)
        logger.info("Using first frame GT pose as initialization")
    
    # 初始化CLAHE（如果启用）
    use_clahe = cfg.tracking.get("use_clahe", True)
    clahe = None
    if use_clahe:
        clahe_clip_limit = cfg.tracking.get("clahe_clip_limit", 2.0)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
        logger.info("CLAHE enhancement enabled")
    
    # 获取几何单位（用于模型点云）
    geometry_unit = cfg.object.get("geometry_unit_in_meter", 0.001)
    
    # 从模板文件中提取模型点云
    # 模板文件中的template_view包含3D点坐标（前3列）
    logger.info("Extracting model points from template file...")
    try:
        import pickle
        with open(cfg.object.pre_render_pkl, "rb") as f:
            pre = pickle.load(f)
        
        template_view = pre.get("template_view", None)
        if template_view is None:
            template_view = pre.get("template_views", None)
        
        if template_view is not None:
            template_view = np.array(template_view)
            # template_view shape: (N_views * N_points, 8) 或 (N_views, N_points, 8)
            if template_view.ndim == 2:
                # 提取所有3D点（前3列），去重
                all_points = template_view[:, :3]  # (N, 3)
                # 使用numpy的unique去重（基于坐标）
                # 由于浮点数精度问题，先四舍五入到一定精度再去重
                points_rounded = np.round(all_points, decimals=6)
                unique_points = np.unique(points_rounded, axis=0)
                model_points = unique_points.astype(np.float32)
                logger.info(f"Extracted {len(model_points)} unique model points from template")
            elif template_view.ndim == 3:
                # (N_views, N_points, 8)
                all_points = template_view[:, :, :3].reshape(-1, 3)  # (N_views * N_points, 3)
                points_rounded = np.round(all_points, decimals=6)
                unique_points = np.unique(points_rounded, axis=0)
                model_points = unique_points.astype(np.float32)
                logger.info(f"Extracted {len(model_points)} unique model points from template")
            else:
                raise ValueError(f"Unexpected template_view shape: {template_view.shape}")
        else:
            raise ValueError("template_view not found in pkl file")
    except Exception as e:
        logger.info(f"Failed to extract model points from template: {e}")
        logger.info("Falling back to default cube model points")
        # 生成模型点云（使用立方体的8个顶点）
        # 假设立方体边长为0.056m（5.6cm），这是cube的默认大小
        cube_size = 0.056 / 2  # 半边长
        model_points = np.array([
            [-cube_size, -cube_size, -cube_size],
            [cube_size, -cube_size, -cube_size],
            [cube_size, cube_size, -cube_size],
            [-cube_size, cube_size, -cube_size],
            [-cube_size, -cube_size, cube_size],
            [cube_size, -cube_size, cube_size],
            [cube_size, cube_size, cube_size],
            [-cube_size, cube_size, cube_size],
        ], dtype=np.float32)
        logger.info(f"Using default cube model with {len(model_points)} points")
    
    # 初始化追踪状态
    current_pose = initial_pose
    fore_hist = None
    back_hist = None
    initialized = False
    
    # 评估结果
    results = {
        'total_frames': 0,
        'valid_frames': 0,
        'add_errors': [],
        'rotation_errors': [],
        'translation_errors': [],
        'add_accuracy': 0.0,
        'rotation_accuracy': 0.0,
        'combined_accuracy': 0.0,  # ADD(5cm/5°)
        'predicted_poses': {},  # 保存预测位姿
    }
    
    # 获取图像文件列表
    image_files = sorted(image_dir.glob("frame_*.png"))
    if end_frame is None:
        end_frame = len(image_files) - 1
    
    logger.info(f"Processing frames {start_frame} to {end_frame} ({end_frame - start_frame + 1} frames)")
    
    # 处理每一帧
    with torch.inference_mode():
        for frame_idx in tqdm(range(start_frame, min(end_frame + 1, len(image_files))), desc="Tracking"):
            results['total_frames'] += 1
            
            # 加载图像
            image_file = image_files[frame_idx]
            if not image_file.exists():
                logger.info(f"Image file not found: {image_file}")
                continue
            
            ir_image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
            if ir_image is None:
                logger.info(f"Failed to load image: {image_file}")
                continue
            
            # 获取实际图像尺寸
            h_img_actual, w_img_actual = ir_image.shape[:2]
            
            # 不再自动缩放相机内参，因为我们已经使用了正确的M3T内参（640x480）
            # 如果图像尺寸不匹配，应该检查图像文件是否正确
            cfg_cx = cfg.camera.get('cx', 0)
            cfg_cy = cfg.camera.get('cy', 0)
            
            # 检查图像尺寸是否与内参匹配
            if frame_idx == 0:
                logger.info(f"Frame {frame_idx} - Image and intrinsics check:")
                logger.info(f"  Image size: {w_img_actual}x{h_img_actual}")
                logger.info(f"  Camera intrinsics: fx={cfg.camera.get('fx', 0):.2f}, fy={cfg.camera.get('fy', 0):.2f}, "
                           f"cx={cfg_cx:.2f}, cy={cfg_cy:.2f}")
                if cfg_cx > w_img_actual or cfg_cy > h_img_actual:
                    logger.info(f"  WARNING: Camera cx/cy ({cfg_cx:.1f}, {cfg_cy:.1f}) exceeds image size ({w_img_actual}x{h_img_actual})")
                    logger.info(f"  This may indicate incorrect intrinsics or image resolution mismatch")
                else:
                    logger.info(f"  Camera intrinsics match image size")
            
            # 应用CLAHE增强（如果启用）
            if clahe is not None:
                ir_image = clahe.apply(ir_image)
            
            # 转换为BGR格式
            frame_bgr = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
            
            # 预处理 - 使用实际图像尺寸构建相机
            frame_rgb, ori_camera_cpu = preprocess_frame(frame_bgr, cfg.camera, device)
            ori_camera = ori_camera_cpu.to(device)
            
            # 调试：检查第一帧的图像尺寸和相机内参
            if frame_idx == 0:
                logger.info(f"Frame {frame_idx} - Image info:")
                logger.info(f"  Actual image size: {w_img_actual}x{h_img_actual}")
                logger.info(f"  Config camera: fx={cfg.camera.get('fx', 0):.2f}, fy={cfg.camera.get('fy', 0):.2f}, "
                           f"cx={cfg.camera.get('cx', 0):.2f}, cy={cfg.camera.get('cy', 0):.2f}")
                try:
                    cam_size = ori_camera.size
                    if cam_size.ndim == 2:
                        cam_w = cam_size[0, 0].item()
                        cam_h = cam_size[0, 1].item()
                    else:
                        cam_w = cam_size[0].item()
                        cam_h = cam_size[1].item()
                    logger.info(f"  Camera image size: {cam_w:.0f}x{cam_h:.0f}")
                except Exception as e:
                    logger.info(f"  Failed to get camera size: {e}")
            
            # 如果是第一帧，需要初始化直方图
            if not initialized:
                bbox_trim_ratio = cfg.tracking.get("bbox_trim_ratio", 0.0)
                try:
                    fore_hist, back_hist, last_bbox, last_centers, last_valid = initialize_from_pose_with_preprocess(
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
                    logger.info(f"Initialized tracking state for frame {frame_idx}")
                except Exception as e:
                    logger.info(f"Initialization failed for frame {frame_idx}: {e}")
                    continue
            
            # 计算边缘强度图（只在追踪阶段需要）
            edge_strength_map = compute_edge_strength(ir_image)
            
            # IR边缘可信度图
            use_fast_edge_confidence = cfg.tracking.get("use_fast_edge_confidence", True)
            edge_confidence_map = compute_ir_edge_confidence(
                ir_image=ir_image,
                edge_strength_map=edge_strength_map,
                ksize=1,
                tau=cfg.tracking.get("edge_confidence_tau", 5.0),
                use_fast_mode=use_fast_edge_confidence,
            )
            
            # 追踪步骤
            bbox_trim_ratio = cfg.tracking.get("bbox_trim_ratio", 0.0)
            
            tracking_success = False
            # 对于第一帧，如果使用GT作为初始化，可以选择跳过追踪直接使用GT位姿
            # 这样可以检查GT位姿是否正确
            skip_tracking_for_first_frame = cfg.tracking.get("skip_tracking_for_first_frame_gt", False)
            if frame_idx == 0 and skip_tracking_for_first_frame and use_first_gt_as_init:
                # 第一帧直接使用GT位姿，不进行追踪优化
                logger.info(f"Frame 0: Skipping tracking, using GT pose directly for visualization")
                tracking_success = True
            else:
                try:
                    (
                        current_pose,
                        fore_hist,
                        back_hist,
                        bbox,
                        centers,
                        valid,
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
                        depth_map=None,
                        edge_confidence_map=edge_confidence_map,
                        bbox_trim_ratio=bbox_trim_ratio,
                    )
                    tracking_success = True
                except Exception as e:
                    logger.info(f"Tracking failed for frame {frame_idx}: {e}")
                    # 即使追踪失败，也继续处理（使用当前位姿进行可视化）
                    if frame_idx == 0:
                        logger.info(f"Frame 0 tracking failed, using initial pose for visualization")
            
            # 检查是否有GT位姿
            if frame_idx not in gt_poses:
                logger.info(f"No GT pose for frame {frame_idx} - skipping GT visualization")
                # 即使没有GT，也进行可视化（仅显示预测位姿）
                if output_dir and tracking_success:
                    # 可视化：仅绘制预测位姿
                    if len(ir_image.shape) == 2:
                        vis_image_bgr = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
                    else:
                        vis_image_bgr = frame_bgr.copy()
                    vis_image_rgb = cv2.cvtColor(vis_image_bgr, cv2.COLOR_BGR2RGB)
                    
                    pred_template_idx = get_closest_template_view_index(current_pose, orientations)
                    pred_template = template_views[pred_template_idx:pred_template_idx+1]
                    pred_data_lines = project_correspondences_line(
                        pred_template, current_pose, ori_camera
                    )
                    pred_bbox = get_bbox_from_p2d(
                        pred_data_lines["centers_in_image"][0], trim_ratio=0.0
                    )
                    pred_centers = pred_data_lines["centers_in_image"][0]
                    pred_valid = pred_data_lines["centers_valid"][0]
                    
                    vis_image_rgb = draw_overlay(
                        vis_image_rgb,
                        pred_bbox,
                        pred_centers,
                        pred_valid,
                        color=(0, 255, 0),
                    )
                    vis_image = cv2.cvtColor(vis_image_rgb, cv2.COLOR_RGB2BGR)
                    
                    vis_dir = output_dir / "visualizations"
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    vis_path = vis_dir / f"frame_{frame_idx:03d}.png"
                    cv2.imwrite(str(vis_path), vis_image)
                continue
            
            # 获取预测位姿
            R_pred, t_pred = pose_to_numpy(current_pose)
            
            # 获取GT位姿
            R_gt = gt_poses[frame_idx]['R']
            t_gt = gt_poses[frame_idx]['t']
            
            # 计算误差
            add_error = compute_add_error(R_pred, t_pred, R_gt, t_gt, model_points)
            rotation_error = compute_rotation_error(R_pred, R_gt)
            translation_error = compute_translation_error(t_pred, t_gt)
            
            results['add_errors'].append(add_error)
            results['rotation_errors'].append(rotation_error)
            results['translation_errors'].append(translation_error)
            results['valid_frames'] += 1
            
            # 保存预测位姿
            results['predicted_poses'][frame_idx] = {
                'R': R_pred.tolist(),
                't': t_pred.tolist(),
            }
            
            # 检查是否满足ADD(5cm/5°)标准
            if add_error <= add_threshold and rotation_error <= rotation_threshold:
                results['combined_accuracy'] += 1
            
            # 可视化：绘制预测位姿和GT位姿
            if output_dir:
                # 使用原始BGR图像进行可视化（与demo.py一致）
                # 将灰度图转换为BGR用于可视化
                if len(ir_image.shape) == 2:
                    vis_image_bgr = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
                else:
                    vis_image_bgr = frame_bgr.copy()
                
                # 转换为RGB格式用于draw_overlay
                # draw_overlay期望RGB输入，但内部会转换为BGR进行绘制，然后返回BGR
                vis_image_rgb = cv2.cvtColor(vis_image_bgr, cv2.COLOR_BGR2RGB)
                
                # 获取图像尺寸
                h_img, w_img = vis_image_rgb.shape[:2]
                
                # 获取最接近的模板视图
                pred_template_idx = get_closest_template_view_index(current_pose, orientations)
                pred_template = template_views[pred_template_idx:pred_template_idx+1]
                
                # 创建GT位姿的Pose对象
                gt_pose_torch = Pose.from_Rt(
                    torch.from_numpy(R_gt).float().unsqueeze(0).to(device),
                    torch.from_numpy(t_gt).float().unsqueeze(0).to(device)
                )
                
                # 调试：检查GT位姿和预测位姿的差异
                if frame_idx == 0:
                    logger.info(f"Frame {frame_idx} - Pose comparison:")
                    logger.info(f"  Predicted: t=({t_pred[0]:.4f}, {t_pred[1]:.4f}, {t_pred[2]:.4f})")
                    logger.info(f"  GT:        t=({t_gt[0]:.4f}, {t_gt[1]:.4f}, {t_gt[2]:.4f})")
                    logger.info(f"  Translation diff: ({t_pred[0]-t_gt[0]:.4f}, {t_pred[1]-t_gt[1]:.4f}, {t_pred[2]-t_gt[2]:.4f})")
                    
                    # 检查旋转矩阵
                    R_pred_np = R_pred
                    R_gt_np = R_gt
                    logger.info(f"  Rotation diff (trace): {np.trace(R_pred_np @ R_gt_np.T):.4f}")
                    logger.info(f"  Predicted rotation matrix R:")
                    logger.info(f"    [{R_pred_np[0, 0]:.4f}, {R_pred_np[0, 1]:.4f}, {R_pred_np[0, 2]:.4f}]")
                    logger.info(f"    [{R_pred_np[1, 0]:.4f}, {R_pred_np[1, 1]:.4f}, {R_pred_np[1, 2]:.4f}]")
                    logger.info(f"    [{R_pred_np[2, 0]:.4f}, {R_pred_np[2, 1]:.4f}, {R_pred_np[2, 2]:.4f}]")
                    logger.info(f"  GT rotation matrix R:")
                    logger.info(f"    [{R_gt_np[0, 0]:.4f}, {R_gt_np[0, 1]:.4f}, {R_gt_np[0, 2]:.4f}]")
                    logger.info(f"    [{R_gt_np[1, 0]:.4f}, {R_gt_np[1, 1]:.4f}, {R_gt_np[1, 2]:.4f}]")
                    logger.info(f"    [{R_gt_np[2, 0]:.4f}, {R_gt_np[2, 1]:.4f}, {R_gt_np[2, 2]:.4f}]")
                    
                    # 检查旋转矩阵是否将x和y都对齐到光轴
                    # 如果旋转矩阵的第三列（z轴方向）是[0, 0, 1]或接近，说明物体正对相机
                    z_axis_pred = R_pred_np[:, 2]
                    z_axis_gt = R_gt_np[:, 2]
                    logger.info(f"  Predicted R z-axis (camera forward): [{z_axis_pred[0]:.4f}, {z_axis_pred[1]:.4f}, {z_axis_pred[2]:.4f}]")
                    logger.info(f"  GT R z-axis (camera forward): [{z_axis_gt[0]:.4f}, {z_axis_gt[1]:.4f}, {z_axis_gt[2]:.4f}]")
                    
                    # 检查投影点的差异
                    pred_template_idx = get_closest_template_view_index(current_pose, orientations)
                    pred_template = template_views[pred_template_idx:pred_template_idx+1]
                    pred_data_lines = project_correspondences_line(
                        pred_template, current_pose, ori_camera
                    )
                    pred_centers_np = pred_data_lines["centers_in_image"][0].detach().cpu().numpy()
                    pred_valid_np = pred_data_lines["centers_valid"][0].detach().cpu().numpy()
                    pred_centers_valid = pred_centers_np[pred_valid_np]
                    
                    gt_template_idx = get_closest_template_view_index(gt_pose_torch, orientations)
                    gt_template = template_views[gt_template_idx:gt_template_idx+1]
                    gt_data_lines = project_correspondences_line(
                        gt_template, gt_pose_torch, ori_camera
                    )
                    gt_centers_np = gt_data_lines["centers_in_image"][0].detach().cpu().numpy()
                    gt_valid_np = gt_data_lines["centers_valid"][0].detach().cpu().numpy()
                    gt_centers_valid = gt_centers_np[gt_valid_np]
                    
                    if len(pred_centers_valid) > 0 and len(gt_centers_valid) > 0:
                        pred_center_mean = pred_centers_valid.mean(axis=0)
                        gt_center_mean = gt_centers_valid.mean(axis=0)
                        logger.info(f"  Projected center diff: ({pred_center_mean[0]-gt_center_mean[0]:.1f}, {pred_center_mean[1]-gt_center_mean[1]:.1f})")
                
                gt_template_idx = get_closest_template_view_index(gt_pose_torch, orientations)
                gt_template = template_views[gt_template_idx:gt_template_idx+1]
                
                # 绘制预测位姿（绿色）- 使用原始相机进行投影
                pred_data_lines = project_correspondences_line(
                    pred_template, current_pose, ori_camera
                )
                pred_bbox = get_bbox_from_p2d(
                    pred_data_lines["centers_in_image"][0], trim_ratio=0.0
                )
                pred_centers = pred_data_lines["centers_in_image"][0]
                pred_valid = pred_data_lines["centers_valid"][0]
                
                # 调试信息：检查预测位姿投影
                if frame_idx == 0:
                    pred_valid_count = pred_valid.sum().item() if isinstance(pred_valid, torch.Tensor) else pred_valid.sum()
                    pred_centers_np = pred_centers.detach().cpu().numpy() if isinstance(pred_centers, torch.Tensor) else pred_centers
                    pred_valid_np = pred_valid.detach().cpu().numpy() if isinstance(pred_valid, torch.Tensor) else pred_valid
                    all_centers = pred_centers_np
                    valid_centers = pred_centers_np[pred_valid_np] if pred_valid_np.any() else None
                    
                    # 检查3D点的分布
                    pred_template_3d = pred_template[0, :, :3]  # 获取3D点（body坐标系）
                    pred_template_3d_np = pred_template_3d.detach().cpu().numpy()
                    pred_pose_3d = current_pose.transform(pred_template_3d)  # 转换到相机坐标系
                    if pred_pose_3d.ndim == 3:
                        pred_pose_3d = pred_pose_3d[0]
                    pred_pose_3d_np = pred_pose_3d.detach().cpu().numpy()
                    
                    logger.info(f"Frame {frame_idx} - Predicted pose:")
                    logger.info(f"  Valid points: {pred_valid_count}/{len(pred_valid)}")
                    logger.info(f"  Template 3D points (body frame) range:")
                    logger.info(f"    x=[{pred_template_3d_np[:, 0].min():.4f}, {pred_template_3d_np[:, 0].max():.4f}], "
                               f"y=[{pred_template_3d_np[:, 1].min():.4f}, {pred_template_3d_np[:, 1].max():.4f}], "
                               f"z=[{pred_template_3d_np[:, 2].min():.4f}, {pred_template_3d_np[:, 2].max():.4f}]")
                    logger.info(f"  3D points in camera frame range:")
                    logger.info(f"    x=[{pred_pose_3d_np[:, 0].min():.4f}, {pred_pose_3d_np[:, 0].max():.4f}], "
                               f"y=[{pred_pose_3d_np[:, 1].min():.4f}, {pred_pose_3d_np[:, 1].max():.4f}], "
                               f"z=[{pred_pose_3d_np[:, 2].min():.4f}, {pred_pose_3d_np[:, 2].max():.4f}]")
                    logger.info(f"  Points in front of camera (z>0): {(pred_pose_3d_np[:, 2] > 0).sum()}/{len(pred_pose_3d_np)}")
                    logger.info(f"  All centers range: x=[{all_centers[:, 0].min():.1f}, {all_centers[:, 0].max():.1f}], "
                               f"y=[{all_centers[:, 1].min():.1f}, {all_centers[:, 1].max():.1f}]")
                    if valid_centers is not None and len(valid_centers) > 0:
                        logger.info(f"  Valid centers range: x=[{valid_centers[:, 0].min():.1f}, {valid_centers[:, 0].max():.1f}], "
                                   f"y=[{valid_centers[:, 1].min():.1f}, {valid_centers[:, 1].max():.1f}]")
                        # 检查点的分布是否异常（所有点都在同一个位置）
                        x_std = valid_centers[:, 0].std()
                        y_std = valid_centers[:, 1].std()
                        logger.info(f"  Valid centers std: x={x_std:.3f}, y={y_std:.3f}")
                        if x_std < 1.0 or y_std < 1.0:
                            logger.info(f"  WARNING: All points are clustered in a very small area!")
                            logger.info(f"    This suggests the template view may have all points at the same location,")
                            logger.info(f"    or the pose rotation is causing all points to align along the camera axis.")
                    if pred_bbox is not None:
                        bbox_np = pred_bbox.detach().cpu().numpy() if isinstance(pred_bbox, torch.Tensor) else pred_bbox
                        logger.info(f"  Bbox: {bbox_np}")
                    
                    # 检查相机内参
                    try:
                        if ori_camera.f.ndim == 2:
                            fx_val = ori_camera.f[0, 0].item()
                            fy_val = ori_camera.f[0, 1].item()
                        else:
                            fx_val = ori_camera.f[0].item()
                            fy_val = ori_camera.f[1].item()
                        if ori_camera.c.ndim == 2:
                            cx_val = ori_camera.c[0, 0].item()
                            cy_val = ori_camera.c[0, 1].item()
                        else:
                            cx_val = ori_camera.c[0].item()
                            cy_val = ori_camera.c[1].item()
                        logger.info(f"  Camera intrinsics: fx={fx_val:.2f}, fy={fy_val:.2f}, "
                                   f"cx={cx_val:.2f}, cy={cy_val:.2f}")
                        logger.info(f"  Camera image size: {ori_camera.width.item():.0f}x{ori_camera.height.item():.0f}")
                    except Exception as e:
                        logger.info(f"  Failed to get camera intrinsics: {e}")
                
                # 直接绘制预测位姿（绿色）- 参考demo_cube.py的方式
                # 使用BGR格式直接绘制
                if pred_centers is not None and pred_valid is not None:
                    try:
                        # 处理维度
                        if pred_centers.ndim == 3:
                            centers_flat = pred_centers[0]
                        elif pred_centers.ndim == 2:
                            centers_flat = pred_centers
                        else:
                            centers_flat = None
                        
                        if pred_valid.ndim > 1:
                            valid_flat = pred_valid.squeeze()
                        else:
                            valid_flat = pred_valid
                        
                        if valid_flat.dtype != torch.bool:
                            valid_flat = valid_flat.bool()
                        
                        if centers_flat is not None and valid_flat is not None:
                            pts = centers_flat[valid_flat].detach().cpu().numpy()
                            if pts.shape[0] >= 3:
                                pts = np.clip(pts, [0, 0], [w_img - 1, h_img - 1])
                                # 参考demo_cube.py：直接使用原始点顺序绘制（模板视图中的轮廓点本身就是有序的）
                                # 不使用凸包，因为凸包会形成比实际轮廓更大的框，不适合非凸形状（如cat模型）
                                # 模板视图中的轮廓点是从mask中提取的，已经按轮廓顺序排列
                                pts_int = pts.reshape(-1, 1, 2).astype(np.int32)
                                # 绘制闭合多边形（BGR格式：绿色=(0, 255, 0)）
                                cv2.polylines(vis_image_bgr, [pts_int], isClosed=True, color=(0, 255, 0), thickness=3)
                            elif pts.shape[0] > 0:
                                # 如果点数量少于3个，至少绘制点
                                for pt in pts:
                                    cv2.circle(vis_image_bgr, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
                                if frame_idx == 0:
                                    logger.info(f"Predicted contour has only {pts.shape[0]} points, drawing as circles")
                    except Exception as e:
                        if frame_idx == 0:
                            logger.info(f"Failed to draw predicted contour: {e}")
                            import traceback
                            logger.info(traceback.format_exc())
                
                # 绘制预测位姿的bbox（绿色）
                if pred_bbox is not None:
                    try:
                        x1, y1, x2, y2 = tensor_bbox_to_xyxy(pred_bbox.cpu())
                        cv2.rectangle(vis_image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    except Exception as e:
                        if frame_idx == 0:
                            logger.info(f"Failed to draw predicted bbox: {e}")
                
                # 绘制GT位姿（红色）- 使用原始相机进行投影
                gt_data_lines = project_correspondences_line(
                    gt_template, gt_pose_torch, ori_camera
                )
                gt_bbox = get_bbox_from_p2d(
                    gt_data_lines["centers_in_image"][0], trim_ratio=0.0
                )
                gt_centers = gt_data_lines["centers_in_image"][0]
                gt_valid = gt_data_lines["centers_valid"][0]
                
                # 调试信息：检查GT位姿投影
                if frame_idx == 0:
                    gt_valid_count = gt_valid.sum().item() if isinstance(gt_valid, torch.Tensor) else gt_valid.sum()
                    gt_centers_np = gt_centers.detach().cpu().numpy() if isinstance(gt_centers, torch.Tensor) else gt_centers
                    gt_valid_np = gt_valid.detach().cpu().numpy() if isinstance(gt_valid, torch.Tensor) else gt_valid
                    valid_gt_centers = gt_centers_np[gt_valid_np]
                    
                    # 检查GT的3D点分布
                    gt_template_3d = gt_template[0, :, :3]  # 获取3D点（body坐标系）
                    gt_template_3d_np = gt_template_3d.detach().cpu().numpy()
                    gt_pose_3d = gt_pose_torch.transform(gt_template_3d)  # 转换到相机坐标系
                    if gt_pose_3d.ndim == 3:
                        gt_pose_3d = gt_pose_3d[0]
                    gt_pose_3d_np = gt_pose_3d.detach().cpu().numpy()
                    
                    logger.info(f"Frame {frame_idx} - GT pose:")
                    logger.info(f"  Valid points: {gt_valid_count}/{len(gt_valid)}")
                    logger.info(f"  Template 3D points (body frame) range:")
                    logger.info(f"    x=[{gt_template_3d_np[:, 0].min():.4f}, {gt_template_3d_np[:, 0].max():.4f}], "
                               f"y=[{gt_template_3d_np[:, 1].min():.4f}, {gt_template_3d_np[:, 1].max():.4f}], "
                               f"z=[{gt_template_3d_np[:, 2].min():.4f}, {gt_template_3d_np[:, 2].max():.4f}]")
                    logger.info(f"  3D points in camera frame range:")
                    logger.info(f"    x=[{gt_pose_3d_np[:, 0].min():.4f}, {gt_pose_3d_np[:, 0].max():.4f}], "
                               f"y=[{gt_pose_3d_np[:, 1].min():.4f}, {gt_pose_3d_np[:, 1].max():.4f}], "
                               f"z=[{gt_pose_3d_np[:, 2].min():.4f}, {gt_pose_3d_np[:, 2].max():.4f}]")
                    if len(valid_gt_centers) > 0:
                        logger.info(f"  Valid centers range: x=[{valid_gt_centers[:, 0].min():.1f}, {valid_gt_centers[:, 0].max():.1f}], "
                                   f"y=[{valid_gt_centers[:, 1].min():.1f}, {valid_gt_centers[:, 1].max():.1f}]")
                        # 检查点的分布是否异常
                        x_std = valid_gt_centers[:, 0].std()
                        y_std = valid_gt_centers[:, 1].std()
                        logger.info(f"  Valid centers std: x={x_std:.3f}, y={y_std:.3f}")
                        if x_std < 1.0 or y_std < 1.0:
                            logger.info(f"  WARNING: All GT points are clustered in a very small area!")
                    if gt_bbox is not None:
                        bbox_np = gt_bbox.detach().cpu().numpy() if isinstance(gt_bbox, torch.Tensor) else gt_bbox
                        logger.info(f"  Bbox: {bbox_np}")
                
                # 直接绘制GT位姿（红色）- 参考demo_cube.py的方式
                if gt_centers is not None and gt_valid is not None:
                    try:
                        # 处理维度
                        if gt_centers.ndim == 3:
                            centers_flat = gt_centers[0]
                        elif gt_centers.ndim == 2:
                            centers_flat = gt_centers
                        else:
                            centers_flat = None
                        
                        if gt_valid.ndim > 1:
                            valid_flat = gt_valid.squeeze()
                        else:
                            valid_flat = gt_valid
                        
                        if valid_flat.dtype != torch.bool:
                            valid_flat = valid_flat.bool()
                        
                        if centers_flat is not None and valid_flat is not None:
                            pts = centers_flat[valid_flat].detach().cpu().numpy()
                            if pts.shape[0] >= 3:
                                pts = np.clip(pts, [0, 0], [w_img - 1, h_img - 1])
                                # 参考demo_cube.py：直接使用原始点顺序绘制（模板视图中的轮廓点本身就是有序的）
                                # 不使用凸包，因为凸包会形成比实际轮廓更大的框，不适合非凸形状（如cat模型）
                                # 模板视图中的轮廓点是从mask中提取的，已经按轮廓顺序排列
                                pts_int = pts.reshape(-1, 1, 2).astype(np.int32)
                                # 绘制闭合多边形（BGR格式：红色=(0, 0, 255)）
                                cv2.polylines(vis_image_bgr, [pts_int], isClosed=True, color=(0, 0, 255), thickness=3)
                            elif pts.shape[0] > 0:
                                # 如果点数量少于3个，至少绘制点
                                for pt in pts:
                                    cv2.circle(vis_image_bgr, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
                                if frame_idx == 0:
                                    logger.info(f"GT contour has only {pts.shape[0]} points, drawing as circles")
                    except Exception as e:
                        if frame_idx == 0:
                            logger.info(f"Failed to draw GT contour: {e}")
                            import traceback
                            logger.info(traceback.format_exc())
                
                # 绘制GT位姿的bbox（红色）
                if gt_bbox is not None:
                    try:
                        x1, y1, x2, y2 = tensor_bbox_to_xyxy(gt_bbox.cpu())
                        cv2.rectangle(vis_image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    except Exception as e:
                        if frame_idx == 0:
                            logger.info(f"Failed to draw GT bbox: {e}")
                
                # 直接使用BGR图像进行文本绘制和保存
                vis_image = vis_image_bgr
                
                # 添加文本信息
                text_y = 30
                cv2.putText(
                    vis_image,
                    f"Frame: {frame_idx}",
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                text_y += 30
                cv2.putText(
                    vis_image,
                    f"ADD: {add_error*1000:.2f}mm",
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                text_y += 30
                cv2.putText(
                    vis_image,
                    f"Rot: {rotation_error:.2f}deg",
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                text_y += 30
                cv2.putText(
                    vis_image,
                    f"Trans: {translation_error*1000:.2f}mm",
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                
                # 状态指示（是否满足5cm/5°标准）
                if add_error <= add_threshold and rotation_error <= rotation_threshold:
                    status_color = (0, 255, 0)  # 绿色：通过
                    status_text = "PASS"
                else:
                    status_color = (0, 0, 255)  # 红色：失败
                    status_text = "FAIL"
                
                text_y += 30
                cv2.putText(
                    vis_image,
                    f"Status: {status_text}",
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    status_color,
                    2,
                )
                
                # 图例
                legend_y = vis_image.shape[0] - 80
                cv2.putText(
                    vis_image,
                    "Green: Predicted",
                    (10, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                legend_y += 25
                cv2.putText(
                    vis_image,
                    "Red: Ground Truth",
                    (10, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                
                # 保存可视化图像
                vis_dir = output_dir / "visualizations"
                vis_dir.mkdir(parents=True, exist_ok=True)
                vis_path = vis_dir / f"frame_{frame_idx:03d}.png"
                cv2.imwrite(str(vis_path), vis_image)
    
    # 计算准确率
    if results['valid_frames'] > 0:
        results['add_accuracy'] = sum(
            1 for e in results['add_errors'] if e <= add_threshold
        ) / results['valid_frames']
        results['rotation_accuracy'] = sum(
            1 for e in results['rotation_errors'] if e <= rotation_threshold
        ) / results['valid_frames']
        results['combined_accuracy'] /= results['valid_frames']
        
        # 计算平均误差
        results['mean_add_error'] = float(np.mean(results['add_errors']))
        results['mean_rotation_error'] = float(np.mean(results['rotation_errors']))
        results['mean_translation_error'] = float(np.mean(results['translation_errors']))
        
        # 计算中位数误差
        results['median_add_error'] = float(np.median(results['add_errors']))
        results['median_rotation_error'] = float(np.median(results['rotation_errors']))
        results['median_translation_error'] = float(np.median(results['translation_errors']))
        
        # 计算标准差
        results['std_add_error'] = float(np.std(results['add_errors']))
        results['std_rotation_error'] = float(np.std(results['rotation_errors']))
        results['std_translation_error'] = float(np.std(results['translation_errors']))
    else:
        results['mean_add_error'] = 0.0
        results['mean_rotation_error'] = 0.0
        results['mean_translation_error'] = 0.0
        results['median_add_error'] = 0.0
        results['median_rotation_error'] = 0.0
        results['median_translation_error'] = 0.0
        results['std_add_error'] = 0.0
        results['std_rotation_error'] = 0.0
        results['std_translation_error'] = 0.0
    
    # 保存结果
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存评估结果JSON
        output_json = output_dir / "evaluation_results.json"
        output_data = {
            'total_frames': results['total_frames'],
            'valid_frames': results['valid_frames'],
            'add_accuracy': results['add_accuracy'],
            'rotation_accuracy': results['rotation_accuracy'],
            'combined_accuracy': results['combined_accuracy'],
            'mean_add_error': results['mean_add_error'],
            'mean_rotation_error': results['mean_rotation_error'],
            'mean_translation_error': results['mean_translation_error'],
            'median_add_error': results['median_add_error'],
            'median_rotation_error': results['median_rotation_error'],
            'median_translation_error': results['median_translation_error'],
            'std_add_error': results['std_add_error'],
            'std_rotation_error': results['std_rotation_error'],
            'std_translation_error': results['std_translation_error'],
        }
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Evaluation results saved to {output_json}")
        
        # 保存预测位姿
        pred_poses_json = output_dir / "predicted_poses.json"
        with open(pred_poses_json, 'w') as f:
            json.dump(results['predicted_poses'], f, indent=2)
        logger.info(f"Predicted poses saved to {pred_poses_json}")
        
        # 可视化图像已保存在 visualizations/ 目录中
        vis_dir = output_dir / "visualizations"
        if vis_dir.exists():
            logger.info(f"Visualization images saved to {vis_dir}")
    
    return results


def print_results(results: Dict):
    """打印评估结果"""
    print("\n" + "="*60)
    print("IR Tracking Evaluation Results (ADD 5cm/5°)")
    print("="*60)
    print(f"Total frames: {results['total_frames']}")
    print(f"Valid frames: {results['valid_frames']}")
    print(f"\nAccuracy Metrics:")
    print(f"  ADD(5cm) accuracy:     {results['add_accuracy']*100:.2f}%")
    print(f"  Rotation(5°) accuracy: {results['rotation_accuracy']*100:.2f}%")
    print(f"  Combined ADD(5cm/5°):  {results['combined_accuracy']*100:.2f}%")
    print(f"\nAverage Errors:")
    print(f"  ADD error:            {results['mean_add_error']*1000:.2f} mm")
    print(f"  Rotation error:       {results['mean_rotation_error']:.2f} deg")
    print(f"  Translation error:    {results['mean_translation_error']*1000:.2f} mm")
    print(f"\nMedian Errors:")
    print(f"  ADD error:            {results['median_add_error']*1000:.2f} mm")
    print(f"  Rotation error:       {results['median_rotation_error']:.2f} deg")
    print(f"  Translation error:    {results['median_translation_error']*1000:.2f} mm")
    print(f"\nStandard Deviation:")
    print(f"  ADD error:            {results['std_add_error']*1000:.2f} mm")
    print(f"  Rotation error:       {results['std_rotation_error']:.2f} deg")
    print(f"  Translation error:    {results['std_translation_error']*1000:.2f} mm")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate IR tracking algorithm")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing IR images (frame_*.png)")
    parser.add_argument("--gt_poses", type=str, required=True,
                        help="Path to GT poses file (pose_ir.txt)")
    parser.add_argument("--cfg", type=str, required=True,
                        help="Path to config file (e.g., realsense_ir_tracking.yaml)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results (optional)")
    parser.add_argument("--add_threshold", type=float, default=0.05,
                        help="ADD threshold in meters (default: 0.05m = 5cm)")
    parser.add_argument("--rotation_threshold", type=float, default=5.0,
                        help="Rotation threshold in degrees (default: 5°)")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Start frame index (default: 0)")
    parser.add_argument("--end_frame", type=int, default=None,
                        help="End frame index (default: None = all frames)")
    parser.add_argument("--no_gt_init", action="store_true",
                        help="Don't use first frame GT pose as initialization")
    parser.add_argument("--gt_camera_fx", type=float, default=None,
                        help="GT camera intrinsics fx (if different from config)")
    parser.add_argument("--gt_camera_fy", type=float, default=None,
                        help="GT camera intrinsics fy (if different from config)")
    parser.add_argument("--gt_camera_cx", type=float, default=None,
                        help="GT camera intrinsics cx (if different from config)")
    parser.add_argument("--gt_camera_cy", type=float, default=None,
                        help="GT camera intrinsics cy (if different from config)")
    
    args = parser.parse_args()
    
    # 构建GT相机内参字典（如果提供了）
    gt_camera_intrinsics = None
    if args.gt_camera_fx is not None or args.gt_camera_fy is not None or \
       args.gt_camera_cx is not None or args.gt_camera_cy is not None:
        if not all([args.gt_camera_fx is not None, args.gt_camera_fy is not None,
                   args.gt_camera_cx is not None, args.gt_camera_cy is not None]):
            print("Error: All GT camera intrinsics (fx, fy, cx, cy) must be provided together")
            return
        gt_camera_intrinsics = {
            'fx': args.gt_camera_fx,
            'fy': args.gt_camera_fy,
            'cx': args.gt_camera_cx,
            'cy': args.gt_camera_cy,
        }
    
    # 评估
    results = evaluate_ir_tracking(
        Path(args.image_dir),
        Path(args.gt_poses),
        Path(args.cfg),
        Path(args.output_dir) if args.output_dir else None,
        args.add_threshold,
        args.rotation_threshold,
        args.start_frame,
        args.end_frame,
        use_first_gt_as_init=not args.no_gt_init,
        gt_camera_intrinsics=gt_camera_intrinsics,
    )
    
    # 打印结果
    print_results(results)


if __name__ == "__main__":
    main()

