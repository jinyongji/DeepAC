#!/bin/bash
# 将追踪程序保存的图像转换为视频

OUTPUT_DIR="/tmp/m3t_tracking_output"
VIDEO_OUTPUT="${OUTPUT_DIR}/tracking_video.mp4"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory not found: $OUTPUT_DIR"
    echo "Please run the tracking program first."
    exit 1
fi

# 检查是否有图像文件
IMAGE_COUNT=$(ls -1 ${OUTPUT_DIR}/color_viewer_image_*.png 2>/dev/null | wc -l)
if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "Error: No images found in $OUTPUT_DIR"
    exit 1
fi

echo "Found $IMAGE_COUNT images. Creating video..."

# 使用 ffmpeg 创建视频
# 帧率设置为30fps，可以根据需要调整
ffmpeg -y -framerate 30 -pattern_type glob -i "${OUTPUT_DIR}/color_viewer_image_*.png \
    -c:v libx264 -pix_fmt yuv420p -crf 23 \
    "$VIDEO_OUTPUT" 2>&1 | tail -10

if [ $? -eq 0 ]; then
    echo "Video created successfully: $VIDEO_OUTPUT"
else
    echo "Error creating video. Make sure ffmpeg is installed."
    echo "Install with: sudo apt-get install ffmpeg"
fi






