"""
追踪质量评估模块
评估追踪质量，失败时触发自恢复
"""
import torch
import numpy as np


def evaluate_tracking_quality(
    contour_confidence=None,
    pose_change_t=None,
    pose_change_R=None,
    template_idx=None,
    fore_hist_mean=None,
    back_hist_mean=None,
    min_confidence=0.05,
    max_pose_change_t=0.02,
    max_pose_change_R=15.0,
    min_hist_distinction=0.0001,
):
    """
    评估追踪质量

    Args:
        contour_confidence: 平均轮廓置信度
        pose_change_t: 平移变化（米）
        pose_change_R: 旋转变化（度）
        template_idx: 当前模板索引
        fore_hist_mean: 前景直方图均值
        back_hist_mean: 背景直方图均值
        min_confidence: 最小置信度阈值
        max_pose_change_t: 最大平移变化阈值（超过认为不稳定）
        max_pose_change_R: 最大旋转变化阈值（超过认为不稳定）
        min_hist_distinction: 最小直方图区分度阈值

    Returns:
        quality_score: 质量分数，范围[0, 1]，越高越好
        is_good: 是否认为追踪质量良好
        issues: 问题列表
    """
    quality_score = 1.0
    issues = []

    # 检查轮廓置信度
    if contour_confidence is not None:
        if contour_confidence < min_confidence:
            quality_score *= 0.5
            issues.append(f"Low contour confidence: {contour_confidence:.3f}")

    # 检查位姿变化幅度（过大说明不稳定）
    if pose_change_t is not None and pose_change_t > max_pose_change_t:
        quality_score *= 0.7
        issues.append(f"Large translation change: {pose_change_t:.6f}m")

    if pose_change_R is not None and pose_change_R > max_pose_change_R:
        quality_score *= 0.7
        issues.append(f"Large rotation change: {pose_change_R:.3f}deg")

    # 检查直方图区分度
    if fore_hist_mean is not None and back_hist_mean is not None:
        hist_distinction = abs(fore_hist_mean - back_hist_mean) / (fore_hist_mean + back_hist_mean + 1e-7)
        if hist_distinction < min_hist_distinction:
            quality_score *= 0.6
            issues.append(f"Low histogram distinction: {hist_distinction:.6f}")

    # 检查是否在正面视角附近徘徊（可能卡住）
    if template_idx is not None:
        if 1200 <= template_idx <= 1400:
            quality_score *= 0.8
            issues.append(f"Near front view (template_idx={template_idx})")

    is_good = quality_score > 0.3 or len(issues) < 3

    return quality_score, is_good, issues


def should_trigger_recovery(
    quality_score,
    consecutive_bad_frames=0,
    min_quality_threshold=0.15,  # 降低阈值，避免过于频繁触发恢复
    max_consecutive_bad=100,  # 增加连续低质量帧数阈值
):
    """
    判断是否应该触发自恢复

    Args:
        quality_score: 当前质量分数
        consecutive_bad_frames: 连续低质量帧数
        min_quality_threshold: 最低质量阈值（默认0.15，更宽松）
        max_consecutive_bad: 最大连续低质量帧数（默认100，更宽松）

    Returns:
        should_recover: 是否应该触发恢复
        # 注意：阈值设置得比较宽松，避免正常追踪时频繁触发恢复
    """
    # 如果质量分数很低，或者连续多帧质量差，触发恢复
    if quality_score < min_quality_threshold:
        return True

    if consecutive_bad_frames >= max_consecutive_bad:
        return True

    return False

