import cv2
import glob
import os

# 你的图像目录
img_dir = "/tmp/m3t_tracking_output"
out_video = "/tmp/m3t_tracking_output/output_video.mp4"

# 找到所有符合名称格式的图片
images = sorted(glob.glob(os.path.join(img_dir, "combined_image_*.png")))

if len(images) == 0:
    print("未找到任何 combined_image_*.png 文件！")
    exit()

# 读第一张确定宽高
frame = cv2.imread(images[0])
h, w, _ = frame.shape

# 创建视频写入器（fps 可以改）
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 15
video = cv2.VideoWriter(out_video, fourcc, fps, (w, h))

# 逐帧写入
for img_path in images:
    img = cv2.imread(img_path)
    video.write(img)

video.release()
print("视频已生成：", out_video)
