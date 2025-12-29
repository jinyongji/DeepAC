"""
轮廓点置信度估计模块
结合IR深度信息和边缘强度计算轮廓点置信度
"""
import numpy as np
import torch
from typing import Optional


def estimate_contour_confidence(
    edge_strength_map,
    points_2d,
    depth_map=None,
    depth_confidence_map=None,
    valid_mask=None,
    edge_weight=0.6,
    depth_weight=0.4,
):
    """
    估计轮廓点的置信度
    
    Args:
        edge_strength_map: 边缘强度图，numpy array (H, W) 或 torch.Tensor
        points_2d: 轮廓点2D坐标，shape (N, 2) 或 (1, N, 2)
        depth_map: 深度图（可选），numpy array (H, W) 或 torch.Tensor
        depth_confidence_map: 深度置信度图（可选），numpy array (H, W) 或 torch.Tensor
        valid_mask: 有效性掩码，shape (N,) 或 (1, N)
        edge_weight: 边缘强度权重（默认0.6）
        depth_weight: 深度信息权重（默认0.4）
    
    Returns:
        confidence: 轮廓点置信度，shape (N,)，值范围[0, 1]
    """
    # 转换为numpy
    if isinstance(edge_strength_map, torch.Tensor):
        edge_np = edge_strength_map.detach().squeeze().cpu().numpy() if edge_strength_map.is_cuda else edge_strength_map.detach().squeeze().numpy()
    else:
        edge_np = edge_strength_map.copy()
    
    if isinstance(points_2d, torch.Tensor):
        points_np = points_2d.detach().squeeze().cpu().numpy() if points_2d.is_cuda else points_2d.detach().squeeze().numpy()
    else:
        points_np = points_2d.copy()
    
    # 处理维度
    if points_np.ndim == 3:
        points_np = points_np[0]  # (1, N, 2) -> (N, 2)
    
    # 处理valid_mask
    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            valid_np = valid_mask.detach().squeeze().cpu().numpy() if valid_mask.is_cuda else valid_mask.detach().squeeze().numpy()
        else:
            valid_np = valid_mask.copy()
        if valid_np.ndim > 1:
            valid_np = valid_np[0] if valid_np.ndim == 2 else valid_np.flatten()
        points_np = points_np[valid_np]
    
    N = len(points_np)
    confidence = np.zeros(N)
    
    # 提取边缘强度值
    h, w = edge_np.shape
    edge_values = np.zeros(N)
    
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
    
    # 如果有深度信息，结合深度置信度
    if depth_map is not None:
        if isinstance(depth_map, torch.Tensor):
            depth_np = depth_map.detach().squeeze().cpu().numpy() if depth_map.is_cuda else depth_map.detach().squeeze().numpy()
        else:
            depth_np = depth_map.copy()
        
        depth_values = np.zeros(N)
        depth_conf_values = np.ones(N)  # 默认深度置信度为1
        
        for i, (x, y) in enumerate(points_np):
            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)
            
            # 双线性插值
            x0, y0 = int(x), int(y)
            x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
            
            fx = x - x0
            fy = y - y0
            
            depth_values[i] = (
                depth_np[y0, x0] * (1 - fx) * (1 - fy) +
                depth_np[y0, x1] * fx * (1 - fy) +
                depth_np[y1, x0] * (1 - fx) * fy +
                depth_np[y1, x1] * fx * fy
            )
            
            if depth_confidence_map is not None:
                if isinstance(depth_confidence_map, torch.Tensor):
                    depth_conf_np = depth_confidence_map.detach().squeeze().cpu().numpy() if depth_confidence_map.is_cuda else depth_confidence_map.detach().squeeze().numpy()
                else:
                    depth_conf_np = depth_confidence_map.copy()
                
                depth_conf_values[i] = (
                    depth_conf_np[y0, x0] * (1 - fx) * (1 - fy) +
                    depth_conf_np[y0, x1] * fx * (1 - fy) +
                    depth_conf_np[y1, x0] * (1 - fx) * fy +
                    depth_conf_np[y1, x1] * fx * fy
                )
        
        # 深度一致性：如果深度值合理（非零且在合理范围内），增加置信度
        depth_valid = (depth_values > 0) & (depth_values < 10.0)  # 假设深度单位是米
        depth_consistency = depth_valid.astype(np.float32) * depth_conf_values
        
        # 综合置信度 = 边缘强度 * edge_weight + 深度一致性 * depth_weight
        confidence = edge_values * edge_weight + depth_consistency * depth_weight
    else:
        # 只有边缘强度
        confidence = edge_values
    
    # 归一化到[0, 1]
    if confidence.max() > 0:
        confidence = confidence / confidence.max()
    
    return confidence

