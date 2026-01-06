"""
使用ArUco标记捕获GT位姿
实时从RealSense IR相机捕获图像并检测ArUco标记，保存GT位姿
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import argparse
from omegaconf import OmegaConf
import json
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src_open.utils.m3t_realsense_camera import M3TRealSenseIRCamera
from src_open.utils.gt_pose_aruco import ArUcoGTGenerator, save_gt_poses


def main():
    parser = argparse.ArgumentParser(description="Capture GT poses using ArUco markers")
    parser.add_argument("--cfg", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--marker_size", type=float, default=0.05,
                        help="ArUco marker size in meters (default: 0.05m = 5cm)")
    parser.add_argument("--marker_id", type=int, default=0,
                        help="ArUco marker ID to detect (default: 0)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: use save_dir from config)")
    parser.add_argument("--save_images", action="store_true",
                        help="Save captured images")
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg)
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(cfg.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # 获取相机内参
    intrinsics = realsense_camera.get_intrinsics()
    camera_matrix = np.array([
        [intrinsics['fx'], 0, intrinsics['cx']],
        [0, intrinsics['fy'], intrinsics['cy']],
        [0, 0, 1],
    ], dtype=np.float32)
    
    print(f"Camera intrinsics: fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}, "
          f"cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}")
    
    # 初始化ArUco GT生成器
    aruco_gt = ArUcoGTGenerator(
        marker_size=args.marker_size,
        marker_id=args.marker_id,
    )
    
    # 存储GT位姿
    gt_poses = {}
    frame_idx = 0
    
    print("\n" + "="*60)
    print("GT Pose Capture Tool")
    print("="*60)
    print("Instructions:")
    print("  - Place ArUco marker on the object")
    print("  - Press SPACE to capture a frame")
    print("  - Press 's' to save and exit")
    print("  - Press ESC or 'q' to exit without saving")
    print("="*60 + "\n")
    
    try:
        while True:
            # 获取IR图像
            ir_image = realsense_camera.get_image()
            if ir_image is None:
                print("Failed to grab frame")
                continue
            
            # 检测标记
            result = aruco_gt.detect_marker_pose(
                ir_image,
                camera_matrix,
            )
            
            # 显示图像
            display_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
            
            if result is not None:
                R, t = result
                # 可视化检测结果
                display_image = aruco_gt.visualize_detection(
                    display_image, R, t, camera_matrix
                )
                
                # 显示状态
                cv2.putText(
                    display_image,
                    f"Marker detected! Frame: {frame_idx}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    display_image,
                    "Press SPACE to capture, 's' to save, ESC to exit",
                    (10, display_image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    display_image,
                    "No marker detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    display_image,
                    "Press SPACE to capture, 's' to save, ESC to exit",
                    (10, display_image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
            
            cv2.imshow("GT Pose Capture", display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # SPACE键：捕获当前帧
                if result is not None:
                    R, t = result
                    gt_poses[frame_idx] = {
                        'R': R.copy(),
                        't': t.copy().reshape(3),
                    }
                    
                    if args.save_images:
                        image_path = output_dir / f"frame_{frame_idx:06d}.png"
                        cv2.imwrite(str(image_path), ir_image)
                    
                    print(f"Captured frame {frame_idx}: t=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})")
                    frame_idx += 1
                else:
                    print("Warning: No marker detected, cannot capture")
            
            elif key == ord('s'):  # 保存并退出
                break
            
            elif key == 27 or key == ord('q'):  # ESC或q：退出不保存
                print("Exiting without saving")
                gt_poses = {}
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        realsense_camera.stop()
        cv2.destroyAllWindows()
        
        # 保存GT位姿
        if len(gt_poses) > 0:
            gt_poses_file = output_dir / "gt_poses.json"
            save_gt_poses(gt_poses, gt_poses_file)
            print(f"\nSaved {len(gt_poses)} GT poses to {gt_poses_file}")
        else:
            print("\nNo poses captured")


if __name__ == "__main__":
    main()








