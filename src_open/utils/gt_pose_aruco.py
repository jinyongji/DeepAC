"""
使用ArUco标记获取GT位姿
ArUco标记是一个已知大小的方形标记，可以用于精确的位姿估计
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Dict
from pathlib import Path
import json


class ArUcoGTGenerator:
    """使用ArUco标记生成GT位姿"""
    
    def __init__(
        self,
        marker_size: float = 0.05,  # 标记大小（米），默认5cm
        dictionary_id: int = cv2.aruco.DICT_6X6_250,  # ArUco字典ID
        marker_id: int = 0,  # 使用的标记ID
    ):
        """
        初始化ArUco GT生成器
        
        Args:
            marker_size: ArUco标记的物理大小（米）
            dictionary_id: ArUco字典ID
            marker_id: 要检测的标记ID
        """
        self.marker_size = marker_size
        self.marker_id = marker_id
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.parameters = cv2.aruco.DetectorParameters()
        
        # 定义标记的3D坐标（在标记坐标系中）
        # 标记中心为原点，Z轴垂直于标记平面
        self.marker_corners_3d = np.array([
            [-marker_size/2, marker_size/2, 0],   # 左上
            [marker_size/2, marker_size/2, 0],     # 右上
            [marker_size/2, -marker_size/2, 0],    # 右下
            [-marker_size/2, -marker_size/2, 0],  # 左下
        ], dtype=np.float32)
    
    def detect_marker_pose(
        self,
        image: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        检测ArUco标记并估计位姿
        
        Args:
            image: 输入图像（灰度或BGR）
            camera_matrix: 相机内参矩阵 (3x3)
            dist_coeffs: 畸变系数，如果为None则假设无畸变
        
        Returns:
            (R, t) 如果检测到标记，否则None
            R: 旋转矩阵 (3x3)，从标记坐标系到相机坐标系
            t: 平移向量 (3x1)，标记中心在相机坐标系中的位置
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 检测标记
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.parameters
        )
        
        if ids is None or len(ids) == 0:
            return None
        
        # 查找指定的标记ID
        marker_idx = None
        for i, detected_id in enumerate(ids.flatten()):
            if detected_id == self.marker_id:
                marker_idx = i
                break
        
        if marker_idx is None:
            return None
        
        # 使用solvePnP估计位姿
        if dist_coeffs is None:
            dist_coeffs = np.zeros((4, 1))
        
        # corners是list，每个元素是(1, 4, 2)的数组
        marker_corners_2d = corners[marker_idx].reshape(4, 2)
        
        success, rvec, tvec = cv2.solvePnP(
            self.marker_corners_3d,
            marker_corners_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
        
        # 将旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)
        
        return R, tvec
    
    def get_object_pose_from_marker(
        self,
        marker_R: np.ndarray,
        marker_t: np.ndarray,
        object_to_marker_transform: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从标记位姿计算物体位姿
        
        Args:
            marker_R: 标记的旋转矩阵（标记坐标系到相机坐标系）
            marker_t: 标记的平移向量（标记中心在相机坐标系中的位置）
            object_to_marker_transform: 物体到标记的变换矩阵 (4x4)，
                                       如果为None，假设物体中心与标记中心重合
        
        Returns:
            (R_object, t_object): 物体的旋转矩阵和平移向量（物体坐标系到相机坐标系）
        """
        if object_to_marker_transform is None:
            # 假设物体中心与标记中心重合
            return marker_R.copy(), marker_t.copy()
        
        # 提取旋转和平移
        R_obj_to_marker = object_to_marker_transform[:3, :3]
        t_obj_to_marker = object_to_marker_transform[:3, 3]
        
        # 计算物体在相机坐标系中的位姿
        # R_object = R_marker * R_obj_to_marker^T
        # t_object = R_marker * t_obj_to_marker + t_marker
        R_object = marker_R @ R_obj_to_marker.T
        t_object = marker_R @ t_obj_to_marker.reshape(3, 1) + marker_t.reshape(3, 1)
        
        return R_object, t_object.reshape(3)
    
    def visualize_detection(
        self,
        image: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 输入图像
            R: 旋转矩阵
            t: 平移向量
            camera_matrix: 相机内参
        
        Returns:
            绘制了坐标轴的图像
        """
        vis_image = image.copy()
        
        # 绘制坐标轴（长度=marker_size）
        axis_length = self.marker_size * 0.8
        axis_points = np.array([
            [0, 0, 0],           # 原点
            [axis_length, 0, 0],  # X轴
            [0, axis_length, 0],  # Y轴
            [0, 0, -axis_length], # Z轴（负Z指向相机）
        ], dtype=np.float32)
        
        # 投影到图像平面
        rvec, _ = cv2.Rodrigues(R)
        image_points, _ = cv2.projectPoints(
            axis_points, rvec, t, camera_matrix, None
        )
        image_points = image_points.reshape(-1, 2).astype(np.int32)
        
        # 绘制坐标轴
        origin = tuple(image_points[0])
        cv2.line(vis_image, origin, tuple(image_points[1]), (0, 0, 255), 3)  # X轴：红色
        cv2.line(vis_image, origin, tuple(image_points[2]), (0, 255, 0), 3)  # Y轴：绿色
        cv2.line(vis_image, origin, tuple(image_points[3]), (255, 0, 0), 3)  # Z轴：蓝色
        
        return vis_image


def save_gt_poses(
    poses: Dict[int, Dict[str, np.ndarray]],
    output_path: Path,
):
    """
    保存GT位姿到文件
    
    Args:
        poses: 字典，key为帧索引，value为包含'R'和't'的字典
        output_path: 输出文件路径
    """
    output_data = {}
    for frame_idx, pose_data in poses.items():
        R = pose_data['R']
        t = pose_data['t']
        # 转换为列表以便JSON序列化
        output_data[str(frame_idx)] = {
            'R': R.tolist(),
            't': t.tolist(),
        }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def load_gt_poses(input_path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    """
    从文件加载GT位姿
    
    Args:
        input_path: 输入文件路径
    
    Returns:
        字典，key为帧索引，value为包含'R'和't'的字典
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    poses = {}
    for frame_idx_str, pose_data in data.items():
        frame_idx = int(frame_idx_str)
        poses[frame_idx] = {
            'R': np.array(pose_data['R']),
            't': np.array(pose_data['t']),
        }
    
    return poses








