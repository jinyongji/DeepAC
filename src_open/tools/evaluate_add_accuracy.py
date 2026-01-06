"""
评估ADD(5cm/5°)准确率
ADD (Average Distance) 是6DoF位姿估计的常用评估指标
"""
import sys
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import argparse
from omegaconf import OmegaConf
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src_open.utils.gt_pose_aruco import load_gt_poses
from src_open.utils.geometry.wrappers import Pose


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
        t_pred: 预测的平移向量 (3,)
        R_gt: GT旋转矩阵 (3x3)
        t_gt: GT平移向量 (3,)
        model_points: 模型点云 (N, 3)，在物体坐标系中
    
    Returns:
        ADD误差（米）
    """
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


def evaluate_add_accuracy(
    pred_poses_file: Path,
    gt_poses_file: Path,
    model_points_file: Optional[Path] = None,
    add_threshold: float = 0.05,  # 5cm
    rotation_threshold: float = 5.0,  # 5度
    geometry_unit: float = 0.001,  # 几何单位（米）
) -> Dict:
    """
    评估ADD(5cm/5°)准确率
    
    Args:
        pred_poses_file: 预测位姿文件路径（每行12个数字：R的9个元素+t的3个元素）
        gt_poses_file: GT位姿文件路径（JSON格式）
        model_points_file: 模型点云文件路径（可选，如果为None则使用简单的8个顶点）
        add_threshold: ADD阈值（米），默认5cm
        rotation_threshold: 旋转阈值（度），默认5度
        geometry_unit: 几何单位（米），用于转换预测位姿的单位
    
    Returns:
        评估结果字典
    """
    # 加载GT位姿
    gt_poses = load_gt_poses(gt_poses_file)
    
    # 加载预测位姿
    pred_poses = {}
    with open(pred_poses_file, 'r') as f:
        for frame_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            values = list(map(float, line.split()))
            if len(values) != 12:
                continue
            
            # 解析旋转矩阵和平移向量
            R_flat = np.array(values[:9])
            t_flat = np.array(values[9:12])
            
            R = R_flat.reshape(3, 3)
            t = t_flat * geometry_unit  # 转换为米
            
            pred_poses[frame_idx] = {'R': R, 't': t}
    
    # 加载或生成模型点云
    if model_points_file is not None and model_points_file.exists():
        # 从文件加载模型点云
        model_points = np.loadtxt(model_points_file)
    else:
        # 使用简单的立方体顶点（假设边长为0.056m，即5.6cm）
        # 这是cube的默认大小
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
    
    # 评估每一帧
    results = {
        'total_frames': 0,
        'valid_frames': 0,
        'add_errors': [],
        'rotation_errors': [],
        'translation_errors': [],
        'add_accuracy': 0.0,
        'rotation_accuracy': 0.0,
        'combined_accuracy': 0.0,  # ADD(5cm/5°)
    }
    
    for frame_idx in sorted(set(pred_poses.keys()) & set(gt_poses.keys())):
        results['total_frames'] += 1
        
        R_pred = pred_poses[frame_idx]['R']
        t_pred = pred_poses[frame_idx]['t']
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
        
        # 检查是否满足ADD(5cm/5°)标准
        if add_error <= add_threshold and rotation_error <= rotation_threshold:
            results['combined_accuracy'] += 1
    
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
        results['mean_add_error'] = np.mean(results['add_errors'])
        results['mean_rotation_error'] = np.mean(results['rotation_errors'])
        results['mean_translation_error'] = np.mean(results['translation_errors'])
        
        # 计算中位数误差
        results['median_add_error'] = np.median(results['add_errors'])
        results['median_rotation_error'] = np.median(results['rotation_errors'])
        results['median_translation_error'] = np.median(results['translation_errors'])
    else:
        results['mean_add_error'] = 0.0
        results['mean_rotation_error'] = 0.0
        results['mean_translation_error'] = 0.0
        results['median_add_error'] = 0.0
        results['median_rotation_error'] = 0.0
        results['median_translation_error'] = 0.0
    
    return results


def print_results(results: Dict):
    """打印评估结果"""
    print("\n" + "="*60)
    print("ADD(5cm/5°) Accuracy Evaluation Results")
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
    print(f"  Translation error:   {results['mean_translation_error']*1000:.2f} mm")
    print(f"\nMedian Errors:")
    print(f"  ADD error:            {results['median_add_error']*1000:.2f} mm")
    print(f"  Rotation error:       {results['median_rotation_error']:.2f} deg")
    print(f"  Translation error:   {results['median_translation_error']*1000:.2f} mm")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ADD(5cm/5°) accuracy")
    parser.add_argument("--pred_poses", type=str, required=True,
                        help="Path to predicted poses file (pose.txt)")
    parser.add_argument("--gt_poses", type=str, required=True,
                        help="Path to GT poses file (JSON format)")
    parser.add_argument("--model_points", type=str, default=None,
                        help="Path to model points file (optional)")
    parser.add_argument("--add_threshold", type=float, default=0.05,
                        help="ADD threshold in meters (default: 0.05m = 5cm)")
    parser.add_argument("--rotation_threshold", type=float, default=5.0,
                        help="Rotation threshold in degrees (default: 5°)")
    parser.add_argument("--geometry_unit", type=float, default=0.001,
                        help="Geometry unit in meters (default: 0.001m = 1mm)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (optional)")
    
    args = parser.parse_args()
    
    # 评估
    results = evaluate_add_accuracy(
        Path(args.pred_poses),
        Path(args.gt_poses),
        Path(args.model_points) if args.model_points else None,
        args.add_threshold,
        args.rotation_threshold,
        args.geometry_unit,
    )
    
    # 打印结果
    print_results(results)
    
    # 保存结果
    if args.output:
        # 转换numpy数组为列表以便JSON序列化
        output_data = {
            'total_frames': results['total_frames'],
            'valid_frames': results['valid_frames'],
            'add_accuracy': float(results['add_accuracy']),
            'rotation_accuracy': float(results['rotation_accuracy']),
            'combined_accuracy': float(results['combined_accuracy']),
            'mean_add_error': float(results['mean_add_error']),
            'mean_rotation_error': float(results['mean_rotation_error']),
            'mean_translation_error': float(results['mean_translation_error']),
            'median_add_error': float(results['median_add_error']),
            'median_rotation_error': float(results['median_rotation_error']),
            'median_translation_error': float(results['median_translation_error']),
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

