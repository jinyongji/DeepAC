"""
估计红外图像第一帧的初始位姿并生成initial_projection_check.jpg
"""
import os
from pathlib import Path
import pickle

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from src_open.utils.geometry.wrappers import Pose, Camera
from src_open.models import get_model
from src_open.utils.lightening_utils import MyLightningLogger, convert_old_model, load_model_weight
from src_open.dataset.utils import read_image, resize, numpy_image_to_torch, crop, zero_pad
from src_open.utils.utils import (
    project_correspondences_line,
    get_closest_template_view_index,
    get_bbox_from_p2d,
)
from src_open.models.deep_ac import calculate_basic_line_data


def create_K_txt(K_matrix, output_path):
    """创建K.txt文件，格式为3x3矩阵"""
    np.savetxt(output_path, K_matrix, fmt='%.8e')
    print(f"已创建K.txt文件: {output_path}")
    print(f"K矩阵内容:\n{K_matrix}")


def preprocess_image(img, bbox2d, camera_cpu, data_conf, device):
    """与demo_cube.py完全相同的预处理流程"""
    # 确保bbox2d是numpy数组格式
    if isinstance(bbox2d, torch.Tensor):
        bbox2d = bbox2d.detach().cpu().numpy()
    elif not isinstance(bbox2d, np.ndarray):
        bbox2d = np.array(bbox2d)
    
    bbox2d = bbox2d.copy()  # 避免修改原始数组
    bbox2d[2:] += data_conf.crop_border * 2
    img_crop, camera_out, _ = crop(img, bbox2d, camera=camera_cpu, return_bbox=True)
    
    if img_crop.size == 0:  # Fallback if crop results in empty image
        img_crop = img
        camera_out = camera_cpu
    
    if isinstance(data_conf.resize, int) and data_conf.resize > 0:
        img_crop, scales = resize(img_crop, data_conf.resize, fn=max if data_conf.resize_by == "max" else min)
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


@torch.no_grad()
def detect_initial_pose(
    frame_rgb,
    ori_camera_cpu,
    template_views,
    orientations,
    model,
    device,
    data_conf,
    init_depth=0.45,
    max_detect_views=240,
    bbox_trim_ratio=0.0,
    depth_search_range=None,
    obj_diameter=None,
    force_bbox_optimization=False,
    bbox_center_x=None,
    bbox_center_y=None,
):
    """检测第一帧的初始位姿，支持多深度搜索和基于bbox的深度优化"""
    ori_camera = ori_camera_cpu.to(device)
    h_img, w_img = frame_rgb.shape[:2]
    
    # 使用图像中心作为初始bbox（可以手动调整中心位置）
    # 如果物体不在中心，可以通过参数调整cx, cy
    center_x = bbox_center_x if bbox_center_x is not None else w_img / 2.0
    center_y = bbox_center_y if bbox_center_y is not None else h_img / 2.0
    bbox_size_w = float(w_img) * 0.3
    bbox_size_h = float(h_img) * 0.3
    bbox2d = torch.tensor([center_x, center_y, bbox_size_w, bbox_size_h], 
                          dtype=torch.float32, device=device)
    if bbox_center_x is not None or bbox_center_y is not None:
        print(f"使用手动指定的bbox中心: ({center_x:.1f}, {center_y:.1f})")
    
    img_tensor, camera = preprocess_image(frame_rgb, bbox2d.cpu().numpy().copy(), ori_camera_cpu, data_conf, device)
    
    # 遍历模板视图寻找最佳匹配
    num_views = template_views.shape[0]
    step = max(1, num_views // max_detect_views)
    
    # 如果提供了深度搜索范围，使用多深度搜索
    if depth_search_range is not None:
        depth_min, depth_max, depth_steps = depth_search_range
        depth_candidates = torch.linspace(depth_min, depth_max, depth_steps, device=device)
        print(f"使用多深度搜索: {depth_min:.2f}m 到 {depth_max:.2f}m，共 {depth_steps} 个深度值")
    else:
        depth_candidates = torch.tensor([init_depth], device=device)
    
    best = {
        "score": -float("inf"),
        "idx": None,
        "pose": None,
        "bbox": None,
        "depth": None,
    }
    
    cam_data = camera._data
    print(f"正在检测初始位姿，遍历 {num_views // step} 个模板视图...")
    
    for idx in tqdm(range(0, num_views, step), desc="检测中"):
        template_view = template_views[idx : idx + 1]
        aa = orientations[idx : idx + 1]
        
        # 对每个模板视图，尝试多个深度值
        for depth in depth_candidates:
            depth_val = depth.item() if isinstance(depth, torch.Tensor) else depth
            pose = Pose.from_aa(aa, torch.tensor([[0.0, 0.0, depth_val]], device=device))
            
            (
                _,
                _,
                centers_in_image,
                centers_valid,
                normals_in_image,
                fg_dist,
                bg_dist,
                _,
            ) = calculate_basic_line_data(template_view, pose._data, cam_data, 1, 0)
            
            if not centers_valid.any():
                continue
            
            fore_hist, back_hist = model.histogram.calculate_histogram(
                img_tensor[None], centers_in_image, centers_valid, normals_in_image, fg_dist, bg_dist, True
            )
            score = torch.mean(fore_hist).item()
            if score > best["score"]:
                valid_mask = centers_valid[0]
                if valid_mask.any():
                    # 使用原始相机投影到原始图像坐标系
                    centers_ori, valid_ori = ori_camera.view2image(pose.transform(template_view[0, :, :3]))
                    bbox = get_bbox_from_p2d(centers_ori[valid_ori], trim_ratio=bbox_trim_ratio)
                else:
                    bbox = None
                best.update({
                    "score": score,
                    "idx": idx,
                    "pose": pose,
                    "bbox": bbox,
                    "depth": depth_val,
                })
    
    if best["idx"] is None:
        raise RuntimeError("未能检测到初始位姿，请检查图像和模型")
    
    # 如果检测到了bbox且提供了物体直径，使用bbox优化深度估计
    if best["bbox"] is not None and obj_diameter is not None:
        print(f"\n检测到bbox，使用bbox优化深度估计...")
        print(f"检测到的bbox: center=({best['bbox'][0]:.1f}, {best['bbox'][1]:.1f}), size=({best['bbox'][2]:.1f}, {best['bbox'][3]:.1f})")
        
        # 使用bbox估计深度
        cam_data_ori = ori_camera._data
        if cam_data_ori.ndim > 1:
            cam_data_ori = cam_data_ori[0]
        fx = float(cam_data_ori[2])
        fy = float(cam_data_ori[3])
        cx = float(cam_data_ori[4])
        cy = float(cam_data_ori[5])
        
        box_w = best["bbox"][2].item()
        box_h = best["bbox"][3].item()
        
        if box_w > 1e-6 and box_h > 1e-6:
            # 使用宽度和高度估计深度，取平均值
            # 注意：魔方是立方体，应该使用宽度或高度中较大的值来估计深度，避免低估
            z_w = fx * obj_diameter / box_w
            z_h = fy * obj_diameter / box_h
            # 使用较大的深度值，避免低估距离
            z_estimated = max(z_w, z_h)
            
            center_x = best["bbox"][0].item()
            center_y = best["bbox"][1].item()
            # 计算平移：注意坐标系（相机坐标系：x右，y下，z前）
            x_estimated = (center_x - cx) / fx * z_estimated
            y_estimated = (center_y - cy) / fy * z_estimated
            
            print(f"Bbox尺寸: w={box_w:.1f}px, h={box_h:.1f}px")
            print(f"深度估计: z_w={z_w:.3f}m, z_h={z_h:.3f}m, 使用z={z_estimated:.3f}m")
            
            print(f"基于bbox估计的位姿: t=({x_estimated:.3f}, {y_estimated:.3f}, {z_estimated:.3f})m")
            
            # 更新位姿的平移部分
            aa = orientations[best["idx"] : best["idx"] + 1]
            pose_optimized = Pose.from_aa(aa, torch.tensor([[x_estimated, y_estimated, z_estimated]], device=device))
            
            # 验证优化后的位姿是否更好
            (
                _,
                _,
                centers_in_image_opt,
                centers_valid_opt,
                normals_in_image_opt,
                fg_dist_opt,
                bg_dist_opt,
                _,
            ) = calculate_basic_line_data(template_views[best["idx"] : best["idx"] + 1], pose_optimized._data, cam_data, 1, 0)
            
            if centers_valid_opt.any():
                fore_hist_opt, back_hist_opt = model.histogram.calculate_histogram(
                    img_tensor[None], centers_in_image_opt, centers_valid_opt, normals_in_image_opt, 
                    fg_dist_opt, bg_dist_opt, True
                )
                score_opt = torch.mean(fore_hist_opt).item()
                
                # 如果强制优化或得分不低于原来的90%，使用优化后的位姿
                use_optimized = force_bbox_optimization or score_opt > best["score"] * 0.9
                if use_optimized:
                    print(f"使用bbox优化的位姿（得分: {score_opt:.6f} vs 原得分: {best['score']:.6f}）")
                    best["pose"] = pose_optimized
                    best["depth"] = z_estimated
                    # 重新计算bbox
                    centers_ori_opt, valid_ori_opt = ori_camera.view2image(
                        pose_optimized.transform(template_views[best["idx"], :, :3])
                    )
                    best["bbox"] = get_bbox_from_p2d(centers_ori_opt[valid_ori_opt], trim_ratio=bbox_trim_ratio)
                else:
                    print(f"保持原检测位姿（优化得分较低: {score_opt:.6f}）")
    
    print(f"\n检测完成！")
    print(f"  最佳匹配模板视图索引: {best['idx']}")
    print(f"  得分: {best['score']:.6f}")
    print(f"  深度: {best['depth']:.3f}m")
    if best["bbox"] is not None:
        print(f"  Bbox: center=({best['bbox'][0]:.1f}, {best['bbox'][1]:.1f}), size=({best['bbox'][2]:.1f}, {best['bbox'][3]:.1f})")
    
    return best


def save_pose_txt(pose, output_path, translation_scale=0.001):
    """保存位姿到pose.txt文件，格式为3x4矩阵（每行一个pose）"""
    R = pose.R.detach().cpu().numpy()
    t = pose.t.detach().cpu().numpy() / translation_scale  # 转换为mm
    
    # 处理batch维度：如果R是[1, 3, 3]，t是[1, 3]，需要去掉batch维度
    if R.ndim == 3:
        R = R[0]  # 取第一个batch，变成[3, 3]
    if t.ndim == 2:
        t = t[0]  # 取第一个batch，变成[3]
    
    # 确保R是[3, 3]，t是[3]
    assert R.shape == (3, 3), f"R shape should be (3, 3), got {R.shape}"
    assert t.shape == (3,), f"t shape should be (3,), got {t.shape}"
    
    # 格式：3x4矩阵 [R | t]
    pose_mat = np.hstack([R, t.reshape(3, 1)])
    
    # 保存为12个数字一行，或3x4矩阵
    # DeepAC通常使用3x4格式（12个数字一行）
    pose_flat = pose_mat.flatten()
    
    np.savetxt(output_path, pose_flat.reshape(1, -1), fmt='%.8e')
    print(f"已保存初始位姿到: {output_path}")
    print(f"位姿矩阵:\n{pose_mat}")
    print(f"平移 (mm): {t}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="配置文件路径")
    parser.add_argument("--first_frame", type=str, 
                       default="/media/jyj/JYJ/cube/frames/IR/frame_00000.jpg",
                       help="第一帧图像路径")
    parser.add_argument("--data_dir", type=str, 
                       default="/media/jyj/JYJ/cube",
                       help="数据目录")
    parser.add_argument("--obj_name", type=str, default="cube", help="物体名称")
    parser.add_argument("--init_depth", type=float, default=0.45, help="初始深度（米）")
    parser.add_argument("--max_detect_views", type=int, default=240, help="最大检测视图数")
    parser.add_argument("--depth_min", type=float, default=0.3, help="深度搜索最小值（米）")
    parser.add_argument("--depth_max", type=float, default=0.8, help="深度搜索最大值（米）")
    parser.add_argument("--depth_steps", type=int, default=10, help="深度搜索步数")
    parser.add_argument("--obj_diameter", type=float, default=0.056, help="物体直径（米），用于bbox优化深度")
    parser.add_argument("--bbox_center_x", type=float, default=None, help="手动指定初始bbox中心x坐标（像素），None表示使用图像中心")
    parser.add_argument("--bbox_center_y", type=float, default=None, help="手动指定初始bbox中心y坐标（像素），None表示使用图像中心")
    parser.add_argument("--force_bbox_optimization", action="store_true", help="强制使用bbox优化的位姿，即使得分略低")
    parser.add_argument("--manual_translation_x", type=float, default=None, help="手动指定平移x（米），用于微调位姿（绝对值）")
    parser.add_argument("--manual_translation_y", type=float, default=None, help="手动指定平移y（米），用于微调位姿（绝对值）")
    parser.add_argument("--manual_translation_z", type=float, default=None, help="手动指定平移z（米），用于微调位姿（绝对值）")
    parser.add_argument("--delta_translation_x", type=float, default=None, help="增量平移x（米），相对于当前位姿向右为正")
    parser.add_argument("--delta_translation_y", type=float, default=None, help="增量平移y（米），相对于当前位姿向下为正")
    parser.add_argument("--delta_translation_z", type=float, default=None, help="增量平移z（米），相对于当前位姿向前为正（向后为负）")
    parser.add_argument("--manual_rotation_x", type=float, default=None, help="手动指定绕x轴旋转角度（度），正值为逆时针，负值为顺时针")
    parser.add_argument("--manual_rotation_y", type=float, default=None, help="手动指定绕y轴旋转角度（度），正值为逆时针，负值为顺时针")
    parser.add_argument("--manual_rotation_z", type=float, default=None, help="手动指定绕z轴旋转角度（度），正值为逆时针，负值为顺时针")
    parser.add_argument("--visual_scale", type=float, default=1.0, help="可视化缩放因子，用于调整模型在图像中的显示大小（>1.0放大，<1.0缩小），不影响实际位姿")
    parser.add_argument("--draw_contour_points", action="store_true", help="绘制轮廓点（红色点），默认不绘制，只绘制立方体线框")
    parser.add_argument("--skip_manual_adjust", action="store_true", help="跳过手动调整，只使用检测结果（用于调试）")
    args = parser.parse_args()
    
    # 加载配置
    cfg = OmegaConf.load(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    if device.type == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    
    logger = MyLightningLogger("DeepAC_IR_Init", cfg.save_dir)
    logger.dump_cfg(cfg, "init_cfg.yml")
    
    train_cfg = OmegaConf.load(cfg.load_cfg)
    data_conf = train_cfg.data
    data_conf.crop_border = cfg.crop_border
    data_conf.resize = cfg.resize
    data_conf.pad = cfg.pad
    data_conf.resize_by = cfg.resize_by
    
    # 加载模型
    model = get_model(train_cfg.models.name)(train_cfg.models)
    ckpt = torch.load(cfg.load_model, map_location="cpu")
    if "pytorch-lightning_version" not in ckpt:
        ckpt = convert_old_model(ckpt)
    load_model_weight(model, ckpt, logger)
    model.to(device).eval()
    
    # 加载模板
    data_dir = args.data_dir
    obj_name = args.obj_name
    template_path = os.path.join(data_dir, obj_name, "pre_render", f"{obj_name}.pkl")
    
    with open(template_path, "rb") as f:
        pre_render_dict = pickle.load(f)
    head = pre_render_dict["head"]
    num_sample = head["num_sample_contour_point"]
    template_views_flat = torch.from_numpy(pre_render_dict["template_view"]).float()
    num_views = template_views_flat.shape[0] // num_sample
    template_views = template_views_flat.view(num_views, num_sample, -1).to(device)
    orientations = torch.from_numpy(pre_render_dict["orientation_in_body"]).float().to(device)
    
    # 创建K.txt文件
    # 红外相机内参矩阵
    K_matrix = np.array([
        [1.19769228e+03, 0.00000000e+00, 6.35855824e+02],
        [0.00000000e+00, 1.19628881e+03, 4.19254040e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])
    K_path = os.path.join(data_dir, "K.txt")
    create_K_txt(K_matrix, K_path)
    
    # 读取第一帧图像
    first_frame_path = args.first_frame
    if not os.path.exists(first_frame_path):
        raise FileNotFoundError(f"第一帧图像不存在: {first_frame_path}")
    
    # 读取红外图像（单通道灰度）
    first_img = read_image(first_frame_path, grayscale=True)  # 返回的是单通道灰度图
    # 转换为RGB格式（复制3次通道）以适配当前模型（需要3通道输入）
    if len(first_img.shape) == 2:
        first_img = cv2.cvtColor(first_img, cv2.COLOR_GRAY2RGB)
    elif first_img.shape[2] == 1:
        first_img = cv2.cvtColor(first_img, cv2.COLOR_GRAY2RGB)
    
    fh, fw = first_img.shape[:2]
    print(f"第一帧图像尺寸: {fw}x{fh}")
    
    # 构建相机
    K = torch.from_numpy(K_matrix).float()
    intrinsic_first = torch.tensor([fw, fh, K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=torch.float32)
    cam_first = Camera(intrinsic_first).to(device)
    cam_first_cpu = Camera(intrinsic_first)
    
    # 检测初始位姿
    print("\n开始检测初始位姿...")
    depth_search_range = (args.depth_min, args.depth_max, args.depth_steps)
    detect_result = detect_initial_pose(
        first_img,
        cam_first_cpu,
        template_views,
        orientations,
        model,
        device,
        data_conf,
        init_depth=args.init_depth,
        max_detect_views=args.max_detect_views,
        bbox_trim_ratio=float(getattr(cfg, "bbox_trim_ratio", 0.0)),
        depth_search_range=depth_search_range,
        obj_diameter=args.obj_diameter,
        force_bbox_optimization=args.force_bbox_optimization,
        bbox_center_x=args.bbox_center_x,
        bbox_center_y=args.bbox_center_y,
    )
    
    init_pose = detect_result["pose"]
    
    # 如果提供了手动调整且未跳过，应用调整
    has_translation = (args.manual_translation_x is not None or args.manual_translation_y is not None or 
                      args.manual_translation_z is not None or args.delta_translation_x is not None or
                      args.delta_translation_y is not None or args.delta_translation_z is not None)
    has_rotation = (args.manual_rotation_x is not None or args.manual_rotation_y is not None or 
                   args.manual_rotation_z is not None)
    
    if not args.skip_manual_adjust and (has_translation or has_rotation):
        print("\n应用手动位姿调整...")
        R = init_pose.R
        t = init_pose.t
        
        # 处理batch维度
        if R.ndim == 3:
            R = R[0]
        if t.ndim == 2:
            t = t[0]
        
        # 调整平移
        t_new = t.clone()
        # 先应用增量调整（相对于当前位姿）
        if args.delta_translation_x is not None:
            t_new[0] = t[0].item() + args.delta_translation_x
            print(f"  平移x（增量）: {t[0].item():.3f}m + {args.delta_translation_x:.3f}m = {t_new[0].item():.3f}m")
        if args.delta_translation_y is not None:
            t_new[1] = t[1].item() + args.delta_translation_y
            print(f"  平移y（增量）: {t[1].item():.3f}m + {args.delta_translation_y:.3f}m = {t_new[1].item():.3f}m")
        if args.delta_translation_z is not None:
            t_new[2] = t[2].item() + args.delta_translation_z
            print(f"  平移z（增量）: {t[2].item():.3f}m + {args.delta_translation_z:.3f}m = {t_new[2].item():.3f}m")
        
        # 再应用绝对值调整（覆盖增量调整的结果）
        if args.manual_translation_x is not None:
            t_new[0] = args.manual_translation_x
            print(f"  平移x（绝对值）: {t[0].item():.3f}m -> {args.manual_translation_x:.3f}m")
        if args.manual_translation_y is not None:
            t_new[1] = args.manual_translation_y
            print(f"  平移y（绝对值）: {t[1].item():.3f}m -> {args.manual_translation_y:.3f}m")
        if args.manual_translation_z is not None:
            t_new[2] = args.manual_translation_z
            print(f"  平移z（绝对值）: {t[2].item():.3f}m -> {args.manual_translation_z:.3f}m")
        
        # 调整旋转（按x, y, z顺序应用旋转）
        import math
        R_new = R.clone()
        
        if args.manual_rotation_x is not None:
            angle_rad = math.radians(args.manual_rotation_x)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            # 绕x轴旋转矩阵
            R_x = torch.tensor([
                [1,     0,      0],
                [0, cos_a, -sin_a],
                [0, sin_a,  cos_a]
            ], dtype=R.dtype, device=R.device)
            R_new = R_x @ R_new
            print(f"  绕x轴旋转: {args.manual_rotation_x:.1f}度")
        
        if args.manual_rotation_y is not None:
            angle_rad = math.radians(args.manual_rotation_y)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            # 绕y轴旋转矩阵
            R_y = torch.tensor([
                [ cos_a, 0, sin_a],
                [     0, 1,     0],
                [-sin_a, 0, cos_a]
            ], dtype=R.dtype, device=R.device)
            R_new = R_y @ R_new
            print(f"  绕y轴旋转: {args.manual_rotation_y:.1f}度")
        
        if args.manual_rotation_z is not None:
            angle_rad = math.radians(args.manual_rotation_z)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            # 绕z轴旋转矩阵
            R_z = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0],
                [0,      0,     1]
            ], dtype=R.dtype, device=R.device)
            R_new = R_z @ R_new
            print(f"  绕z轴旋转: {args.manual_rotation_z:.1f}度")
        
        # 重新构建位姿
        if R_new.ndim == 2:
            R_new = R_new.unsqueeze(0)
        if t_new.ndim == 1:
            t_new = t_new.unsqueeze(0)
        init_pose = Pose.from_Rt(R_new, t_new).to(device)
        
        # 检查位姿是否合理（z值应该为正，表示在相机前方）
        t_check = t_new[0] if t_new.ndim == 2 else t_new
        print(f"调整后的位姿: t=({t_check[0]:.3f}, {t_check[1]:.3f}, {t_check[2]:.3f})m")
        if t_check[2] < 0.1:  # z值太小，可能在相机后面
            print(f"警告: z值({t_check[2]:.3f}m)过小，模型可能在相机后方或太近！")
        
        # 检查旋转后的立方体是否在视野内
        # 注意：cam_first在后面定义，这里先跳过检查
    
    # 保存位姿到pose.txt
    pose_path = os.path.join(data_dir, "frames/IR/pose.txt")
    os.makedirs(os.path.dirname(pose_path), exist_ok=True)
    translation_scale = cfg.translation_unit_in_meter
    save_pose_txt(init_pose, pose_path, translation_scale=translation_scale)
    
    # 生成initial_projection_check.jpg
    img_vis = first_img.copy()
    
    # 方法1：绘制轮廓点（原始方法，可能不规则）- 默认不绘制，避免与立方体线框混淆
    if args.draw_contour_points:
        idx0 = detect_result["idx"]
        template_view_vis = template_views[idx0 : idx0 + 1]
        projected = project_correspondences_line(template_view_vis, init_pose, cam_first)
        centers = projected["centers_in_image"][0]
        valid_mask = projected["centers_valid"][0].bool()
        
        # 绘制红色轮廓点
        pts = centers[valid_mask].detach().cpu().numpy()
        for pt in pts:
            xi, yi = int(round(pt[0])), int(round(pt[1]))
            if 0 <= xi < fw and 0 <= yi < fh:
                cv2.circle(img_vis, (xi, yi), 2, (0, 0, 255), -1)  # 红色点
        print(f"\n已绘制 {len(pts)} 个轮廓点")
    
    # 先检查原始检测位姿（不应用手动调整时）
    print(f"\n原始检测位姿: t=({init_pose.t[0].item() if init_pose.t.ndim == 1 else init_pose.t[0, 0].item():.3f}, "
          f"{init_pose.t[1].item() if init_pose.t.ndim == 1 else init_pose.t[0, 1].item():.3f}, "
          f"{init_pose.t[2].item() if init_pose.t.ndim == 1 else init_pose.t[0, 2].item():.3f})m")
    
    # 方法2：绘制规则的立方体包围盒（如果提供了物体直径）
    if args.obj_diameter is not None:
        # 创建立方体的8个顶点（在物体坐标系中，以原点为中心）
        cube_size = args.obj_diameter * args.visual_scale  # 立方体边长等于直径乘以可视化缩放因子
        half_size = cube_size / 2.0
        if args.visual_scale != 1.0:
            print(f"\n应用可视化缩放: obj_diameter={args.obj_diameter:.3f}m × visual_scale={args.visual_scale:.2f} = {cube_size:.3f}m")
        
        # 立方体的8个顶点（物体坐标系）
        # 坐标系说明：
        # - x轴：向右（正方向）
        # - y轴：向下（正方向）
        # - z轴：向前/远离相机（正方向）
        # 立方体以原点为中心，边长为cube_size
        cube_vertices_body = torch.tensor([
            [-half_size, -half_size, -half_size],  # 0: 左下后（x-, y-, z-）
            [ half_size, -half_size, -half_size],  # 1: 右下后（x+, y-, z-）
            [ half_size,  half_size, -half_size],  # 2: 右上后（x+, y+, z-）
            [-half_size,  half_size, -half_size],  # 3: 左上后（x-, y+, z-）
            [-half_size, -half_size,  half_size],  # 4: 左下前（x-, y-, z+）
            [ half_size, -half_size,  half_size],  # 5: 右下前（x+, y-, z+）
            [ half_size,  half_size,  half_size],  # 6: 右上前（x+, y+, z+）
            [-half_size,  half_size,  half_size],  # 7: 左上前（x-, y+, z+）
        ], dtype=torch.float32, device=device)
        
        # 变换到相机坐标系
        cube_vertices_cam = init_pose.transform(cube_vertices_body)
        
        # 调试：打印变换后的形状
        print(f"\n调试: cube_vertices_cam变换后形状={cube_vertices_cam.shape}")
        
        # 处理batch维度（如果有）
        if cube_vertices_cam.ndim == 3:
            print(f"  检测到batch维度，去掉batch维度...")
            cube_vertices_cam = cube_vertices_cam[0]  # 去掉batch维度 [1, 8, 3] -> [8, 3]
            print(f"  去掉batch维度后形状={cube_vertices_cam.shape}")
        
        # 检查z值（深度），确保在相机前方
        z_values = cube_vertices_cam[:, 2].detach().cpu().numpy()
        # 确保z_values是一维数组
        if z_values.ndim > 1:
            z_values = z_values.flatten()
        # 调试信息
        print(f"  最终z_values形状={z_values.shape}, z_values={z_values}")
        min_z = z_values.min()
        max_z = z_values.max()
        print(f"\n立方体深度范围: z_min={min_z:.3f}m, z_max={max_z:.3f}m")
        if min_z < 0.1:
            print(f"警告: 立方体部分顶点在相机后方或太近（z < 0.1m）！")
        
        # 投影到2D
        cube_vertices_2d, cube_valid = cam_first.view2image(cube_vertices_cam)
        
        # 处理batch维度
        if cube_vertices_2d.ndim == 2 and cube_vertices_2d.shape[0] == 8:
            cube_vertices_2d_np = cube_vertices_2d.detach().cpu().numpy()
            cube_valid_np = cube_valid.detach().cpu().numpy()
        elif cube_vertices_2d.ndim == 3:
            cube_vertices_2d_np = cube_vertices_2d[0].detach().cpu().numpy()
            cube_valid_np = cube_valid[0].detach().cpu().numpy()
        else:
            cube_vertices_2d_np = cube_vertices_2d.detach().cpu().numpy()
            cube_valid_np = cube_valid.detach().cpu().numpy()
        
        # 检查有多少顶点在视野内
        valid_count = int(cube_valid_np.sum())
        print(f"立方体8个顶点中，{valid_count}/8 个在视野内")
        
        # 打印每个顶点的位置和有效性
        print("\n立方体顶点详情:")
        for i, (pt, valid) in enumerate(zip(cube_vertices_2d_np, cube_valid_np)):
            # 确保z_val是标量
            z_val_np = z_values[i]
            if isinstance(z_val_np, np.ndarray):
                z_val = float(z_val_np.item() if z_val_np.size == 1 else z_val_np[0])
            else:
                z_val = float(z_val_np)
            status = "有效" if valid else "无效"
            print(f"  顶点{i}: 2D=({pt[0]:.1f}, {pt[1]:.1f}), z={z_val:.3f}m, {status}")
        
        if valid_count == 0:
            print("\n错误: 立方体完全不在视野内！")
            print("可能的原因:")
            print("  1. 旋转角度过大，导致立方体旋转到视野外")
            print("  2. z值过小或为负，立方体在相机后方")
            print("  3. 平移调整导致立方体移出图像范围")
            print("\n建议:")
            print("  - 先不使用任何手动调整，检查原始检测结果")
            print("  - 如果原始结果正常，逐步添加调整参数")
            print("  - 减小旋转角度（建议每次调整5-10度）")
        elif valid_count < 4:
            print("警告: 立方体大部分不在视野内，可能旋转角度过大。")
        
        # 立方体的12条边（定义边的连接关系）
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 后表面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 前表面
            (0, 4), (1, 5), (2, 6), (3, 7),  # 连接前后表面的边
        ]
        
        # 绘制立方体的边（红色线条）
        drawn_edges = 0
        for edge in edges:
            v1_idx, v2_idx = edge
            if cube_valid_np[v1_idx] and cube_valid_np[v2_idx]:
                pt1 = cube_vertices_2d_np[v1_idx]
                pt2 = cube_vertices_2d_np[v2_idx]
                x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
                x2, y2 = int(round(pt2[0])), int(round(pt2[1]))
                # 确保在图像范围内
                if 0 <= x1 < fw and 0 <= y1 < fh and 0 <= x2 < fw and 0 <= y2 < fh:
                    cv2.line(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色线条
                    drawn_edges += 1
        
        print(f"绘制了 {drawn_edges}/12 条立方体边")
        
        # 绘制立方体的顶点（红色圆圈）
        drawn_vertices = 0
        for i, (pt, valid) in enumerate(zip(cube_vertices_2d_np, cube_valid_np)):
            if valid:
                xi, yi = int(round(pt[0])), int(round(pt[1]))
                if 0 <= xi < fw and 0 <= yi < fh:
                    cv2.circle(img_vis, (xi, yi), 4, (0, 0, 255), -1)  # 红色顶点
                    drawn_vertices += 1
        
        print(f"绘制了 {drawn_vertices}/{valid_count} 个立方体顶点")
        
        # 如果没有任何边或顶点被绘制，尝试绘制所有顶点（即使无效），用于调试
        if drawn_edges == 0 and drawn_vertices == 0:
            print("\n尝试绘制所有顶点（包括无效的）用于调试...")
            for i, (pt, valid) in enumerate(zip(cube_vertices_2d_np, cube_valid_np)):
                # 确保z_val是标量
                z_val_np = z_values[i]
                if isinstance(z_val_np, np.ndarray):
                    z_val = float(z_val_np.item() if z_val_np.size == 1 else z_val_np[0])
                else:
                    z_val = float(z_val_np)
                xi, yi = int(round(pt[0])), int(round(pt[1]))
                print(f"  调试顶点{i}: 2D坐标=({xi}, {yi}), 图像范围=[0-{fw}, 0-{fh}], z={z_val:.3f}m, valid={valid}")
                # 即使不在图像范围内，也尝试绘制（可能会被裁剪）
                if -1000 <= xi < fw + 1000 and -1000 <= yi < fh + 1000:
                    # 使用不同颜色标记有效/无效
                    color = (0, 255, 0) if valid else (0, 0, 255)  # 绿色=有效，红色=无效
                    # 限制在图像范围内
                    x_clamped = max(0, min(xi, fw-1))
                    y_clamped = max(0, min(yi, fh-1))
                    cv2.circle(img_vis, (x_clamped, y_clamped), 5, color, -1)
                    # 添加顶点编号
                    cv2.putText(img_vis, str(i), (max(0, min(xi+5, fw-20)), max(15, min(yi, fh-5))), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 额外调试：打印所有顶点的2D坐标范围
        if cube_vertices_2d_np.shape[0] > 0:
            x_coords = cube_vertices_2d_np[:, 0]
            y_coords = cube_vertices_2d_np[:, 1]
            print(f"\n立方体2D坐标范围: x=[{x_coords.min():.1f}, {x_coords.max():.1f}], y=[{y_coords.min():.1f}, {y_coords.max():.1f}]")
            print(f"图像范围: x=[0, {fw}], y=[0, {fh}]")
            if x_coords.min() < 0 or x_coords.max() >= fw or y_coords.min() < 0 or y_coords.max() >= fh:
                print("警告: 立方体顶点坐标超出图像范围！")
    
    save_dir = Path(logger.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    init_check_path = save_dir / "initial_projection_check.jpg"
    cv2.imwrite(str(init_check_path), img_vis)
    print(f"\n已保存初始投影检查图像到: {init_check_path}")
    
    # 也保存到数据目录
    ir_check_path = os.path.join(data_dir, "frames/IR/initial_projection_check.jpg")
    os.makedirs(os.path.dirname(ir_check_path), exist_ok=True)
    cv2.imwrite(ir_check_path, img_vis)
    print(f"已保存初始投影检查图像到: {ir_check_path}")


if __name__ == "__main__":
    main()

