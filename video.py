import cv2
import os


def images_to_video(image_folder, video_name, fps=24.0):
    """
    将指定文件夹中的图像按顺序合成为一个视频文件。

    参数:
    image_folder (str): 包含图像文件的文件夹路径。
    video_name (str): 输出的视频文件名（例如 'output.mp4'）。
    fps (float): 视频的帧率（每秒显示的帧数）。
    """
    # 检查图像文件夹是否存在
    if not os.path.exists(image_folder):
        print(f"错误：图像文件夹 '{image_folder}' 不存在。")
        return

    # 获取文件夹中的所有文件，并按名称排序
    # sorted() 确保了文件是按顺序处理的（例如 frame_001.png, frame_002.png...）
    images = sorted(
        [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))])

    # 如果没有找到任何图像文件
    if not images:
        print(f"在文件夹 '{image_folder}' 中没有找到任何支持的图像文件。")
        return

    # 获取第一张图像的尺寸，以确定视频的分辨率
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"错误：无法读取第一张图像 '{first_image_path}'。")
        return

    height, width, layers = frame.shape

    # 定义视频编码器和创建VideoWriter对象
    # 'mp4v' 是一个常用的MP4编码器，具有很好的兼容性
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    print(f"开始转换，共 {len(images)} 张图片...")

    # 遍历所有图像文件
    for i, image in enumerate(images):
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)

        # 检查图像是否成功读取
        if frame is None:
            print(f"警告：跳过无法读取的文件 '{image_path}'")
            continue

        # 将当前帧写入视频
        video.write(frame)

        # 打印进度
        if (i + 1) % 10 == 0 or (i + 1) == len(images):
            print(f"已处理 {i + 1}/{len(images)} 张图片...")

    # 释放VideoWriter对象
    video.release()
    cv2.destroyAllWindows()  # 关闭所有可能由OpenCV创建的窗口

    print(f"\n视频已成功创建！")
    print(f"输出文件: {os.path.abspath(video_name)}")


if __name__ == '__main__':
    # --- 用户配置区域 ---

    # 1. 包含图像的目录
    # 请确保路径正确，并且该目录下只有你想要转换的图像文件
    input_image_folder = '/home/jyj/Projects/DeepAC/evaluation_results_cat/visualizations'

    # 2. 输出的视频文件名
    # 你可以自定义文件名和格式（例如 'output.avi', 'my_video.mp4'）
    output_video_name = 'output_video_cat.mp4'

    # 3. 视频帧率 (FPS)
    # 例如，24或30是比较常见的帧率
    video_fps = 24.0

    # --- 执行转换 ---
    images_to_video(input_image_folder, output_video_name, fps=video_fps)
