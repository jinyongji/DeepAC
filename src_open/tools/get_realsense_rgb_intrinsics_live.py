"""
从运行中的live_tracking获取RealSense RGB相机的内参
这个脚本会启动相机并输出实际内参
"""
import cv2
import sys

def get_realsense_rgb_intrinsics_from_camera(camera_id=6):
    """通过OpenCV获取RealSense RGB相机的内参（如果支持）"""
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"无法打开相机 ID {camera_id}")
        return None
    
    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"相机 ID {camera_id} 信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    # 尝试读取一帧
    ret, frame = cap.read()
    if ret:
        print(f"  成功读取帧: {frame.shape}")
    else:
        print("  无法读取帧")
        cap.release()
        return None
    
    cap.release()
    
    print("\n注意：OpenCV无法直接获取相机内参。")
    print("请使用以下方法之一获取RealSense RGB内参：")
    print("1. 运行: python src_open/tools/get_realsense_rgb_intrinsics.py")
    print("2. 使用RealSense SDK的rs-enumerate-devices工具")
    print("3. 使用相机标定工具（如OpenCV的calibrateCamera）")
    
    return None

if __name__ == "__main__":
    camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    get_realsense_rgb_intrinsics_from_camera(camera_id)












