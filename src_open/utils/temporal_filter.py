"""
时间一致性滤波模块
使用指数移动平均（EMA）或Kalman滤波平滑位姿更新
"""
import torch
import numpy as np
from src_open.utils.geometry.wrappers import Pose


class ExponentialMovingAverageFilter:
    """
    指数移动平均滤波器
    用于平滑位姿更新
    """
    def __init__(self, alpha=0.7):
        """
        Args:
            alpha: 平滑系数，范围[0, 1]，越大越不平滑（更响应新值）
        """
        self.alpha = alpha
        self.initialized = False
        self.smoothed_pose = None

    def update(self, pose: Pose):
        """
        更新平滑位姿

        Args:
            pose: 当前位姿

        Returns:
            smoothed_pose: 平滑后的位姿
        """
        if not self.initialized:
            self.smoothed_pose = Pose(pose._data.clone())
            self.initialized = True
            return self.smoothed_pose

        # 指数移动平均
        # smoothed = alpha * current + (1 - alpha) * previous
        smoothed_data = self.alpha * pose._data + (1 - self.alpha) * self.smoothed_pose._data
        self.smoothed_pose = Pose(smoothed_data)

        return self.smoothed_pose

    def reset(self):
        """重置滤波器"""
        self.initialized = False
        self.smoothed_pose = None


class AdaptiveTemporalFilter:
    """
    自适应时间一致性滤波器
    根据位姿变化幅度和轮廓置信度调整平滑强度
    """
    def __init__(self, base_alpha=0.3, min_alpha=0.1, max_alpha=0.6):
        """
        Args:
            base_alpha: 基础平滑系数
            min_alpha: 最小平滑系数（更平滑）
            max_alpha: 最大平滑系数（更响应）
        """
        self.base_alpha = base_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.initialized = False
        self.smoothed_pose = None
        self.prev_pose = None

    def update(self, pose: Pose, pose_change_t=None, pose_change_R=None, contour_confidence=None):
        """
        自适应更新平滑位姿

        Args:
            pose: 当前位姿
            pose_change_t: 平移变化（米）
            pose_change_R: 旋转变化（度）
            contour_confidence: 平均轮廓置信度

        Returns:
            smoothed_pose: 平滑后的位姿
        """
        if not self.initialized:
            self.smoothed_pose = Pose(pose._data.clone())
            self.prev_pose = Pose(pose._data.clone())
            self.initialized = True
            return self.smoothed_pose

        # 根据位姿变化和置信度调整alpha
        alpha = self.base_alpha

        # 如果位姿变化很大，减少平滑（更响应）
        if pose_change_t is not None and pose_change_t > 0.01:  # 1cm
            alpha = min(self.max_alpha, alpha + 0.1)
        elif pose_change_t is not None and pose_change_t < 0.001:  # 1mm
            alpha = max(self.min_alpha, alpha - 0.1)

        # 如果轮廓置信度低，增加平滑（减少噪声影响）
        if contour_confidence is not None:
            if contour_confidence < 0.1:  # 低置信度（暗处边缘）
                alpha = self.min_alpha  # 直接使用最小alpha，最平滑
            elif contour_confidence < 0.15:  # 中等低置信度
                alpha = max(self.min_alpha, alpha - 0.15)  # 更激进的平滑
            elif contour_confidence < 0.2:  # 中等置信度
                alpha = max(self.min_alpha, alpha - 0.05)
            elif contour_confidence > 0.25:  # 高置信度
                alpha = min(self.max_alpha, alpha + 0.05)

        # 指数移动平均（使用detach避免梯度累积）
        smoothed_data = alpha * pose._data.detach() + (1 - alpha) * self.smoothed_pose._data.detach()
        self.smoothed_pose = Pose(smoothed_data)
        self.prev_pose = Pose(pose._data.detach().clone())

        return self.smoothed_pose

    def reset(self):
        """重置滤波器"""
        self.initialized = False
        self.smoothed_pose = None
        self.prev_pose = None
