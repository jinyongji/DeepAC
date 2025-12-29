"""
边缘提取模块
实现梯度幅值计算和边缘强度评估
"""
import numpy as np
import cv2
import torch


def compute_gradient_magnitude(image):
    """
    计算图像的梯度幅值
    
    Args:
        image: 输入图像，可以是numpy array (H, W) 或 torch.Tensor (1, 1, H, W)
    
    Returns:
        gradient_magnitude: 梯度幅值图，与输入格式相同
    """
    if isinstance(image, torch.Tensor):
        # 转换为numpy进行处理
        img_np = image.squeeze().cpu().numpy() if image.is_cuda else image.squeeze().numpy()
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
    else:
        img_np = image.copy()
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
    
    # 使用Sobel算子计算梯度
    grad_x = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度幅值
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 归一化到[0, 1]
    if gradient_magnitude.max() > 0:
        gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
    
    # 转换回原始格式
    if isinstance(image, torch.Tensor):
        gradient_magnitude = torch.from_numpy(gradient_magnitude).float()
        if image.is_cuda:
            gradient_magnitude = gradient_magnitude.to(image.device)
        # 恢复原始维度
        while gradient_magnitude.ndim < image.ndim:
            gradient_magnitude = gradient_magnitude.unsqueeze(0)
    
    return gradient_magnitude


def compute_edge_strength(image, gradient_magnitude=None):
    """
    计算边缘强度评估
    
    Args:
        image: 输入图像，numpy array (H, W) 或 torch.Tensor
        gradient_magnitude: 梯度幅值图（可选，如果不提供则计算）
    
    Returns:
        edge_strength: 边缘强度图，值范围[0, 1]，值越大表示边缘越强
    """
    if gradient_magnitude is None:
        gradient_magnitude = compute_gradient_magnitude(image)
    
    # 转换为numpy进行处理
    if isinstance(gradient_magnitude, torch.Tensor):
        grad_np = gradient_magnitude.squeeze().cpu().numpy() if gradient_magnitude.is_cuda else gradient_magnitude.squeeze().numpy()
    else:
        grad_np = gradient_magnitude.copy()
    
    if isinstance(image, torch.Tensor):
        img_np = image.squeeze().cpu().numpy() if image.is_cuda else image.squeeze().numpy()
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
    else:
        img_np = image.copy()
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
    
    # 使用Canny边缘检测作为辅助
    canny_edges = cv2.Canny(img_np, 50, 150)
    canny_binary = (canny_edges > 0).astype(np.float32)
    
    # 结合梯度幅值和Canny边缘
    # 边缘强度 = 梯度幅值 * Canny二值化 * 局部对比度增强
    edge_strength = grad_np * canny_binary
    
    # 应用非极大值抑制（NMS）增强边缘
    # 使用形态学操作增强边缘连续性
    kernel = np.ones((3, 3), np.uint8)
    edge_strength = cv2.dilate(edge_strength, kernel, iterations=1)
    edge_strength = cv2.erode(edge_strength, kernel, iterations=1)
    
    # 归一化到[0, 1]
    if edge_strength.max() > 0:
        edge_strength = edge_strength / edge_strength.max()
    
    # 转换回原始格式
    if isinstance(image, torch.Tensor):
        edge_strength = torch.from_numpy(edge_strength).float()
        if image.is_cuda:
            edge_strength = edge_strength.to(image.device)
        # 恢复原始维度
        while edge_strength.ndim < image.ndim:
            edge_strength = edge_strength.unsqueeze(0)
    
    return edge_strength


def get_edge_strength_at_points(edge_strength, points_2d, valid_mask=None):
    """
    获取指定2D点的边缘强度值
    
    Args:
        edge_strength: 边缘强度图，numpy array (H, W) 或 torch.Tensor
        points_2d: 2D点坐标，shape (N, 2) 或 (1, N, 2)
        valid_mask: 有效性掩码，shape (N,) 或 (1, N)
    
    Returns:
        edge_values: 边缘强度值，shape (N,)
    """
    # 转换为numpy
    if isinstance(edge_strength, torch.Tensor):
        edge_np = edge_strength.squeeze().cpu().numpy() if edge_strength.is_cuda else edge_strength.squeeze().numpy()
    else:
        edge_np = edge_strength.copy()
    
    if isinstance(points_2d, torch.Tensor):
        points_np = points_2d.squeeze().cpu().numpy() if points_2d.is_cuda else points_2d.squeeze().numpy()
    else:
        points_np = points_2d.copy()
    
    # 处理维度
    if points_np.ndim == 3:
        points_np = points_np[0]  # (1, N, 2) -> (N, 2)
    
    # 处理valid_mask
    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            valid_np = valid_mask.squeeze().cpu().numpy() if valid_mask.is_cuda else valid_mask.squeeze().numpy()
        else:
            valid_np = valid_mask.copy()
        if valid_np.ndim > 1:
            valid_np = valid_np[0] if valid_np.ndim == 2 else valid_np.flatten()
        points_np = points_np[valid_np]
    
    # 提取边缘强度值（使用双线性插值）
    h, w = edge_np.shape
    edge_values = np.zeros(len(points_np))
    
    for i, (x, y) in enumerate(points_np):
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        
        # 双线性插值
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        
        fx = x - x0
        fy = y - y0
        
        edge_values[i] = (
            edge_np[y0, x0] * (1 - fx) * (1 - fy) +
            edge_np[y0, x1] * fx * (1 - fy) +
            edge_np[y1, x0] * (1 - fx) * fy +
            edge_np[y1, x1] * fx * fy
        )
    
    return edge_values


