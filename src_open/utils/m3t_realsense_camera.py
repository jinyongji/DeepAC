"""
M3T风格的RealSense IR相机接口
提供与M3T C++接口类似的Python接口，用于DeepAC实时追踪
"""
import numpy as np
import cv2
try:
    import pyrealsense2 as rs
except ImportError:
    raise ImportError("pyrealsense2 not installed. Install with: pip install pyrealsense2")


class M3TRealSenseIRCamera:
    """M3T风格的RealSense IR相机接口"""
    
    def __init__(self, ir_index=1, emitter_enabled=False):
        """
        初始化RealSense IR相机
        
        Args:
            ir_index: IR相机索引，1=左IR, 2=右IR
            emitter_enabled: 是否启用结构光发射器（默认False，避免IR图像中的光点）
        """
        self.ir_index = ir_index
        self.emitter_enabled = emitter_enabled
        self.pipeline = None
        self.config = None
        self.profile = None
        self.intrinsics = None
        self.width = None
        self.height = None
        
    def setup(self, width=640, height=480, fps=30):
        """
        设置并启动IR流
        
        Args:
            width: 图像宽度
            height: 图像高度
            fps: 帧率
            
        Returns:
            bool: 是否成功设置
        """
        try:
            # 检查RealSense设备是否可用
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                raise RuntimeError("No RealSense devices found. Check USB connection.")
            
            device = devices[0]
            device_name = device.get_info(rs.camera_info.name)
            print(f"Found RealSense device: {device_name}")
            
            # 列出可用的IR流配置
            sensor = device.query_sensors()[0]  # 通常第一个sensor是stereo module
            stream_profiles = sensor.get_stream_profiles()
            ir_profiles = [p for p in stream_profiles if p.stream_type() == rs.stream.infrared and p.format() == rs.format.y8]
            
            if len(ir_profiles) == 0:
                raise RuntimeError("No IR stream profiles found. Check if IR camera is available.")
            
            # 查找匹配的配置
            matching_profile = None
            for profile in ir_profiles:
                if profile.stream_index() == self.ir_index:
                    vp = profile.as_video_stream_profile()
                    if vp.width() == width and vp.height() == height and vp.fps() == fps:
                        matching_profile = profile
                        break
            
            # 如果没有精确匹配，尝试找最接近的
            if matching_profile is None:
                print(f"Warning: Exact resolution {width}x{height}@{fps}fps not found for IR index {self.ir_index}")
                print("Available IR stream profiles:")
                for profile in ir_profiles:
                    if profile.stream_index() == self.ir_index:
                        vp = profile.as_video_stream_profile()
                        print(f"  IR{profile.stream_index()}: {vp.width()}x{vp.height()}@{vp.fps()}fps")
                
                # 尝试使用第一个匹配索引的配置
                for profile in ir_profiles:
                    if profile.stream_index() == self.ir_index:
                        matching_profile = profile
                        vp = profile.as_video_stream_profile()
                        width = vp.width()
                        height = vp.height()
                        fps = vp.fps()
                        print(f"Using available resolution: {width}x{height}@{fps}fps")
                        break
            
            if matching_profile is None:
                available_indices = set(p.stream_index() for p in ir_profiles)
                if self.ir_index not in available_indices:
                    # 如果请求的索引不可用，使用第一个可用的索引
                    if len(available_indices) > 0:
                        self.ir_index = min(available_indices)
                        print(f"Warning: Requested IR index not available. Using available index: {self.ir_index}")
                        # 重新查找匹配的配置
                        for profile in ir_profiles:
                            if profile.stream_index() == self.ir_index:
                                matching_profile = profile
                                vp = profile.as_video_stream_profile()
                                width = vp.width()
                                height = vp.height()
                                fps = vp.fps()
                                print(f"Using available resolution: {width}x{height}@{fps}fps")
                                break
                    else:
                        raise RuntimeError(f"No IR stream profiles available")
            
            # 创建pipeline和config
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # 启用IR流（使用实际可用的分辨率）
            self.config.enable_stream(
                rs.stream.infrared, 
                self.ir_index,
                width, 
                height, 
                rs.format.y8, 
                fps
            )
            
            # 启动pipeline
            self.profile = self.pipeline.start(self.config)
            
            # 获取内参
            ir_stream = self.profile.get_stream(rs.stream.infrared, self.ir_index)
            ir_intrinsics = ir_stream.as_video_stream_profile().get_intrinsics()
            
            self.intrinsics = {
                "fx": float(ir_intrinsics.fx),
                "fy": float(ir_intrinsics.fy),
                "cx": float(ir_intrinsics.ppx),
                "cy": float(ir_intrinsics.ppy),
                "width": ir_intrinsics.width,
                "height": ir_intrinsics.height,
            }
            self.width = ir_intrinsics.width
            self.height = ir_intrinsics.height
            
            # 配置发射器（结构光）
            device = self.profile.get_device()
            depth_sensor = device.first_depth_sensor()
            if depth_sensor:
                depth_sensor.set_option(rs.option.emitter_enabled, 1 if self.emitter_enabled else 0)
            
            return True
        except RuntimeError as e:
            # RuntimeError（包括RealSense错误）
            print(f"Failed to setup RealSense IR camera: {e}")
            return False
        except Exception as e:
            print(f"Failed to setup RealSense IR camera: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update_image(self, synchronized=True):
        """
        更新图像
        
        Args:
            synchronized: 是否同步等待帧
            
        Returns:
            bool: 是否成功更新
        """
        if self.pipeline is None:
            return False
        
        try:
            if synchronized:
                # 等待帧（超时5秒）
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            else:
                # 非阻塞获取帧
                frames = self.pipeline.poll_for_frames()
                if not frames:
                    return False
            
            ir_frame = frames.get_infrared_frame(self.ir_index)
            if ir_frame:
                return True
            return False
        except RuntimeError as e:
            if "Frame didn't arrive" in str(e):
                return False
            raise
        except Exception as e:
            print(f"Failed to update IR image: {e}")
            return False
    
    def get_image(self):
        """
        获取当前IR图像（单通道灰度图）
        
        Returns:
            numpy.ndarray: IR图像 (H, W)，uint8格式，如果失败返回None
        """
        if self.pipeline is None:
            return None
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            ir_frame = frames.get_infrared_frame(self.ir_index)
            if ir_frame:
                # 转换为numpy数组
                ir_image = np.asanyarray(ir_frame.get_data())
                return ir_image
            return None
        except RuntimeError as e:
            if "Frame didn't arrive" in str(e):
                return None
            raise
        except Exception as e:
            print(f"Failed to get IR image: {e}")
            return None
    
    def get_intrinsics(self):
        """
        获取相机内参
        
        Returns:
            dict: 包含fx, fy, cx, cy, width, height的字典，如果失败返回None
        """
        return self.intrinsics
    
    def stop(self):
        """停止相机"""
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception as e:
                print(f"Error stopping RealSense pipeline: {e}")
            finally:
                self.pipeline = None
                self.profile = None
                self.config = None

