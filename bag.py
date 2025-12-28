import pyrealsense2 as rs
import numpy as np
import cv2
import os



# ========= 路径配置 =========
bag_path = "/home/jyj/Documents/4.bag"
save_dir = "/home/jyj/Documents/ir_frames"
os.makedirs(save_dir, exist_ok=True)

# ========= RealSense 管线 =========
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_path, repeat_playback=False)
config.enable_stream(rs.stream.infrared, 1)

profile = pipeline.start(config)

# 关闭实时模式（防止丢帧）
playback = profile.get_device().as_playback()
playback.set_real_time(False)

frame_id = 0

try:
    while True:
        frames = pipeline.wait_for_frames(timeout_ms=10000)
        ir_frame = frames.get_infrared_frame(1)

        if not ir_frame:
            continue

        # IR 是 16bit
        ir_image = np.asanyarray(ir_frame.get_data())

        # 为了显示，归一化到 8bit
        ir_8u = cv2.normalize(
            ir_image, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        # 显示窗口
        cv2.imshow("IR Stream", ir_8u)

        # 保存当前帧
        save_path = os.path.join(save_dir, f"ir_{frame_id:06d}.png")
        cv2.imwrite(save_path, ir_8u)

        frame_id += 1

        # 控制播放速度（30ms ≈ 33FPS）
        key = cv2.waitKey(30)
        if key == 27:  # ESC
            break

except RuntimeError:
    print("Playback finished")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"已保存 {frame_id} 帧 IR 图像到 {save_dir}")
