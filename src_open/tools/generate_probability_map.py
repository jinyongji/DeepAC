"""
生成目标物体的概率图（Probability Map）
用于可视化和诊断追踪问题
"""
import torch
import numpy as np
import cv2
from src_open.utils.geometry.wrappers import Pose, Camera
from src_open.utils.geometry.render_geometry import calculate_basic_line_data


def calculate_pixel_probability_from_histogram(pixel_value, histogram, num_bins=32):
    """
    根据像素值和直方图计算概率
    
    Args:
        pixel_value: 像素值（归一化到[0, 1]）
        histogram: 直方图 [num_bins]
        num_bins: 直方图的bin数量
    
    Returns:
        prob: 概率值
    """
    # 将像素值映射到bin索引
    bin_idx = min(int(pixel_value * num_bins), num_bins - 1)
    return histogram[bin_idx].item()


@torch.no_grad()
def generate_object_probability_map(
    image_tensor,
    camera,
    current_pose,
    template_view,
    fore_hist,
    back_hist,
    model,
    device,
    sigma=5.0,
):
    """
    生成目标物体的概率图
    
    Args:
        image_tensor: 图像张量 [1, C, H, W]
        camera: 相机对象
        current_pose: 当前位姿
        template_view: 模板视图 [N, 8]
        fore_hist: 前景直方图 [num_bins]
        back_hist: 背景直方图 [num_bins]
        model: DeepAC模型
        device: 设备
        sigma: 高斯核的标准差（用于平滑概率图）
    
    Returns:
        prob_map: 概率图 [H, W]，值在[0, 1]之间，表示每个像素是目标物体的概率
        fore_prob_map: 前景概率图 [H, W]
        back_prob_map: 背景概率图 [H, W]
    """
    # 确保输入维度正确
    if isinstance(template_view, torch.Tensor):
        if template_view.ndim == 2:
            template_view = template_view.unsqueeze(0)  # [N, 8] -> [1, N, 8]
    
    # 计算轮廓线数据
    (
        centers_in_body,
        centers_in_view,
        centers_in_image,
        centers_valid,
        normals_in_image,
        fg_dist,
        bg_dist,
        valid_data_line,
    ) = calculate_basic_line_data(
        template_view, current_pose[None]._data, camera[None]._data, 1, 0
    )
    
    # 确保维度正确
    while centers_in_image.ndim > 3:
        centers_in_image = centers_in_image.squeeze(0)
    while centers_valid.ndim > 2:
        centers_valid = centers_valid.squeeze(0)
    while normals_in_image.ndim > 3:
        normals_in_image = normals_in_image.squeeze(0)
    while fg_dist.ndim > 2:
        fg_dist = fg_dist.squeeze(0)
    while bg_dist.ndim > 2:
        bg_dist = bg_dist.squeeze(0)
    
    # 获取图像尺寸
    _, H, W = image_tensor.shape
    
    # 初始化概率图
    fore_prob_map = torch.zeros(H, W, device=device, dtype=torch.float32)
    back_prob_map = torch.zeros(H, W, device=device, dtype=torch.float32)
    weight_map = torch.zeros(H, W, device=device, dtype=torch.float32)
    
    # 获取有效的轮廓点
    valid_mask = centers_valid[0]
    if not valid_mask.any():
        return torch.zeros(H, W, device=device), fore_prob_map, back_prob_map
    
    centers = centers_in_image[0][valid_mask]  # [N_valid, 2]
    normals = normals_in_image[0][valid_mask]  # [N_valid, 2]
    fg_dists = fg_dist[0][valid_mask]  # [N_valid]
    bg_dists = bg_dist[0][valid_mask]  # [N_valid]
    
    # 计算每个轮廓点的前景和背景概率
    # 使用直方图来计算概率
    num_bins = fore_hist.shape[-1]
    
    # 沿着轮廓线采样像素值
    for i in range(len(centers)):
        cx, cy = centers[i].cpu().numpy()
        nx, ny = normals[i].cpu().numpy()
        fg_d = fg_dists[i].item()
        bg_d = bg_dists[i].item()
        
        # 采样前景区域（沿着法线方向向内）
        if fg_d > 0:
            # 采样多个点
            num_samples = max(1, int(fg_d / 2))
            for s in range(num_samples):
                dist = fg_d * (s + 1) / num_samples
                px = int(round(cx - nx * dist))
                py = int(round(cy - ny * dist))
                
                if 0 <= px < W and 0 <= py < H:
                    # 获取像素值（归一化到[0, 1]）
                    pixel_val = image_tensor[0, :, py, px].mean().item()  # 灰度图或RGB的均值
                    pixel_val = max(0.0, min(1.0, pixel_val))  # 确保在[0, 1]范围内
                    
                    # 计算该像素值在前景直方图中的概率
                    bin_idx = min(int(pixel_val * num_bins), num_bins - 1)
                    fore_prob = fore_hist[bin_idx].item() if bin_idx < len(fore_hist) else 0.0
                    
                    # 使用高斯权重（距离轮廓线越近，权重越大）
                    weight = np.exp(-(dist / max(fg_d, 1.0)) ** 2 / (2 * (sigma / max(fg_d, 1.0)) ** 2))
                    fore_prob_map[py, px] += fore_prob * weight
                    weight_map[py, px] += weight
        
        # 采样背景区域（沿着法线方向向外）
        if bg_d > 0:
            num_samples = max(1, int(bg_d / 2))
            for s in range(num_samples):
                dist = bg_d * (s + 1) / num_samples
                px = int(round(cx + nx * dist))
                py = int(round(cy + ny * dist))
                
                if 0 <= px < W and 0 <= py < H:
                    # 获取像素值（归一化到[0, 1]）
                    pixel_val = image_tensor[0, :, py, px].mean().item()
                    pixel_val = max(0.0, min(1.0, pixel_val))  # 确保在[0, 1]范围内
                    
                    # 计算该像素值在背景直方图中的概率
                    bin_idx = min(int(pixel_val * num_bins), num_bins - 1)
                    back_prob = back_hist[bin_idx].item() if bin_idx < len(back_hist) else 0.0
                    
                    # 使用高斯权重
                    weight = np.exp(-(dist / max(bg_d, 1.0)) ** 2 / (2 * (sigma / max(bg_d, 1.0)) ** 2))
                    back_prob_map[py, px] += back_prob * weight
    
    # 归一化概率图
    eps = 1e-6
    fore_prob_map = fore_prob_map / (weight_map + eps)
    back_prob_map = back_prob_map / (weight_map + eps)
    
    # 计算目标物体的概率（前景概率 / (前景概率 + 背景概率)）
    prob_map = fore_prob_map / (fore_prob_map + back_prob_map + eps)
    
    # 应用高斯平滑
    if sigma > 0:
        prob_map_np = prob_map.cpu().numpy()
        prob_map_np = cv2.GaussianBlur(prob_map_np, (0, 0), sigma)
        prob_map = torch.from_numpy(prob_map_np).to(device)
    
    return prob_map, fore_prob_map, back_prob_map


def visualize_probability_map(prob_map, image=None, alpha=0.5):
    """
    可视化概率图
    
    Args:
        prob_map: 概率图 [H, W]，值在[0, 1]之间
        image: 原始图像（可选），用于叠加显示
        alpha: 叠加透明度
    
    Returns:
        vis_image: 可视化图像 [H, W, 3]，BGR格式
    """
    if isinstance(prob_map, torch.Tensor):
        prob_map = prob_map.cpu().numpy()
    
    # 归一化到[0, 255]
    prob_map_uint8 = (prob_map * 255).astype(np.uint8)
    
    # 应用颜色映射（热力图）
    prob_map_colored = cv2.applyColorMap(prob_map_uint8, cv2.COLORMAP_JET)
    
    if image is not None:
        # 如果提供了原始图像，叠加显示
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            if image.ndim == 3:
                if image.shape[0] == 1 or image.shape[0] == 3:
                    image = image.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
                if image.shape[2] == 1:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 确保图像是BGR格式
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 叠加显示
        vis_image = cv2.addWeighted(image, 1 - alpha, prob_map_colored, alpha, 0)
    else:
        vis_image = prob_map_colored
    
    return vis_image

