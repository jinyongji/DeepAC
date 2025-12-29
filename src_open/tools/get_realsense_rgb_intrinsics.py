"""
获取RealSense RGB相机的内参
用于重新生成cube.pkl文件
"""
import pyrealsense2 as rs
import sys

def get_realsense_rgb_intrinsics(width=1280, height=720, fps=30):
    """获取RealSense RGB相机的内参"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 配置RGB流
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    
    try:
        profile = pipeline.start(config)
        
        # 获取RGB流的内参
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        print(f"RealSense RGB Camera Intrinsics ({width}x{height}@{fps}fps):")
        print(f"  fx = {intrinsics.fx:.6f}")
        print(f"  fy = {intrinsics.fy:.6f}")
        print(f"  cx = {intrinsics.ppx:.6f}")
        print(f"  cy = {intrinsics.ppy:.6f}")
        print(f"  width = {intrinsics.width}")
        print(f"  height = {intrinsics.height}")
        print(f"\nFor config file:")
        print(f"  fx: {intrinsics.fx:.6f}")
        print(f"  fy: {intrinsics.fy:.6f}")
        print(f"  cx: {intrinsics.ppx:.6f}")
        print(f"  cy: {intrinsics.ppy:.6f}")
        
        pipeline.stop()
        
        return {
            'fx': intrinsics.fx,
            'fy': intrinsics.fy,
            'cx': intrinsics.ppx,
            'cy': intrinsics.ppy,
            'width': intrinsics.width,
            'height': intrinsics.height,
        }
    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying alternative resolutions...")
        
        # 尝试其他分辨率
        resolutions = [
            (1280, 720),
            (640, 480),
            (848, 480),
            (1920, 1080),
        ]
        
        for w, h in resolutions:
            try:
                config = rs.config()
                config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
                profile = pipeline.start(config)
                color_stream = profile.get_stream(rs.stream.color)
                intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
                
                print(f"\nFound working resolution: {w}x{h}")
                print(f"  fx = {intrinsics.fx:.6f}")
                print(f"  fy = {intrinsics.fy:.6f}")
                print(f"  cx = {intrinsics.ppx:.6f}")
                print(f"  cy = {intrinsics.ppy:.6f}")
                
                pipeline.stop()
                return {
                    'fx': intrinsics.fx,
                    'fy': intrinsics.fy,
                    'cx': intrinsics.ppx,
                    'cy': intrinsics.ppy,
                    'width': intrinsics.width,
                    'height': intrinsics.height,
                }
            except:
                continue
        
        print("Failed to get intrinsics from any resolution")
        return None

if __name__ == "__main__":
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 1280
    height = int(sys.argv[2]) if len(sys.argv) > 2 else 720
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    intrinsics = get_realsense_rgb_intrinsics(width, height, fps)
    
    if intrinsics:
        print("\n✓ Successfully retrieved RealSense RGB intrinsics")
    else:
        print("\n✗ Failed to retrieve RealSense RGB intrinsics")
        sys.exit(1)













