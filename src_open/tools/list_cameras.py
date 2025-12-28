"""
列出所有可用的摄像头，并尝试识别RealSense相机
"""
import cv2
import sys

def list_cameras():
    """列出所有可用的摄像头"""
    print("Scanning for available cameras...")
    print("=" * 60)
    
    available_cameras = []
    
    # 尝试打开前20个摄像头ID
    for camera_id in range(20):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            # 获取摄像头信息
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 尝试读取一帧来确认摄像头可用
            ret, frame = cap.read()
            if ret:
                # 尝试获取后端名称（可能包含设备信息）
                backend = cap.getBackendName()
                
                camera_info = {
                    'id': camera_id,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'backend': backend,
                    'frame_shape': frame.shape if ret else None,
                }
                available_cameras.append(camera_info)
                
                print(f"Camera ID {camera_id}:")
                print(f"  Resolution: {width}x{height}")
                print(f"  FPS: {fps}")
                print(f"  Backend: {backend}")
                print(f"  Frame shape: {frame.shape}")
                
                # 尝试识别RealSense相机
                # RealSense相机通常通过V4L2后端访问，且分辨率可能是640x480或1280x720
                is_realsense = False
                if 'v4l2' in backend.lower() or 'v4l' in backend.lower():
                    # 检查是否是常见的RealSense分辨率
                    if (width, height) in [(640, 480), (1280, 720), (848, 480), (1920, 1080)]:
                        is_realsense = True
                        print(f"  ⚠️  Possibly RealSense camera (V4L2 backend)")
                
                print()
            cap.release()
    
    print("=" * 60)
    print(f"Found {len(available_cameras)} available camera(s)")
    
    if len(available_cameras) == 0:
        print("No cameras found!")
        return None
    
    # 尝试使用pyrealsense2来确认RealSense相机
    print("\nChecking for RealSense cameras using pyrealsense2...")
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) > 0:
            print(f"Found {len(devices)} RealSense device(s):")
            for i, device in enumerate(devices):
                device_name = device.get_info(rs.camera_info.name)
                serial_number = device.get_info(rs.camera_info.serial_number)
                print(f"  Device {i}: {device_name} (Serial: {serial_number})")
                
                # 检查RGB流配置
                sensors = device.query_sensors()
                for sensor in sensors:
                    if sensor.get_info(rs.camera_info.name) == "RGB Camera":
                        print(f"    RGB Camera found")
                        # 列出可用的RGB流配置
                        profiles = sensor.get_stream_profiles()
                        rgb_profiles = [p for p in profiles if p.stream_type() == rs.stream.color]
                        if rgb_profiles:
                            print(f"    Available RGB resolutions:")
                            for p in rgb_profiles[:5]:  # 只显示前5个
                                vp = p.as_video_stream_profile()
                                print(f"      {vp.width()}x{vp.height()}@{vp.fps()}fps")
        else:
            print("  No RealSense devices found via pyrealsense2")
    except ImportError:
        print("  pyrealsense2 not installed, skipping RealSense detection")
    except Exception as e:
        print(f"  Error checking RealSense: {e}")
    
    print("\n" + "=" * 60)
    print("Recommendations:")
    print("1. If you see a RealSense device above, note its camera ID")
    print("2. Update your config file with: camera_id: <correct_id>")
    print("3. Or use command line: --camera_id <correct_id>")
    
    return available_cameras

if __name__ == "__main__":
    cameras = list_cameras()
    
    if cameras:
        print("\nAvailable camera IDs:", [c['id'] for c in cameras])
        
        # 提示用户如何选择
        print("\nTo use a specific camera, update your config file:")
        print("  camera:")
        print("    camera_id: <id>")
        print("\nOr use command line argument:")
        print("  python -m src_open.tools.live_tracking --cfg <config> --camera_id <id>")
