import cv2
import os


def slow_down_video(input_path, output_path, slow_factor=2):
    """
    将视频放慢指定的倍数，并保存为MP4格式。

    :param input_path: 输入视频文件的路径。
    :param output_path: 输出视频文件的路径 (应为 .mp4)。
    :param slow_factor: 放慢的倍数，默认为2（即速度减半）。
    """
    # 1. 打开输入视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 '{input_path}'")
        return

    # 2. 获取视频属性
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"原始视频信息:")
    print(f"  - 路径: {input_path}")
    print(f"  - 帧率 (FPS): {original_fps}")
    print(f"  - 分辨率: {width}x{height}")
    print(f"  - 总帧数: {total_frames}")

    # 3. 定义输出视频的编码器和属性
    # --- 修改点 1: 更改编码器 ---
    # 对于 .mp4 文件，'mp4v' 是一个常用且兼容性好的编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    new_fps = original_fps / slow_factor

    print(f"\n处理中...")
    print(f"  - 放慢倍数: {slow_factor}")
    print(f"  - 新帧率 (FPS): {new_fps}")
    print(f"  - 输出路径: {output_path}")

    # 创建 VideoWriter 对象
    out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))

    if not out.isOpened():
        print(f"错误：无法创建输出视频文件 '{output_path}'。请检查编码器和文件扩展名是否匹配。")
        cap.release()
        return

    # 4. 逐帧读取并写入
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)

        frame_count += 1
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  进度: {frame_count}/{total_frames} ({progress:.1f}%)")

    # 5. 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n处理完成！视频已成功放慢并保存为MP4格式。")


# --- 主程序入口 ---
if __name__ == '__main__':
    input_video_filename = '/home/jyj/Projects/DeepAC/workspace/live_cube_ir_innovative/live_ir_innovative_20251231_180647.avi'
    # --- 修改点 2: 更改输出文件名后缀为 .mp4 ---
    output_video_filename = '/home/jyj/Projects/DeepAC/workspace/live_cube_ir_innovative/live_ir_innovative_slowed_down.mp4'

    if not os.path.exists(input_video_filename):
        print(f"错误：在当前目录下找不到文件 '{input_video_filename}'。")
        print("请确保视频文件与此Python脚本在同一个文件夹中，或者提供完整的文件路径。")
    else:
        slow_down_video(input_video_filename, output_video_filename, slow_factor=2)
