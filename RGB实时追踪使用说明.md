# RGB实时追踪使用说明

## 概述

本配置用于在光照充足环境下使用RGB相机进行实时魔方追踪。相机在RGB模式下拍摄的是3通道RGB图像，与IR模式下的单通道灰度图像不同。

## 相机参数

- **分辨率**: 1280x720
- **内参**:
  - fx: 1197.69228
  - fy: 1196.28881
  - cx: 635.855824
  - cy: 419.25404

## 使用方法

### 1. 查找可用的摄像头ID

如果不知道USB摄像头的ID，先运行以下命令列出所有可用摄像头：

```bash
conda run -n deepac python -m src_open.tools.list_cameras
```

这会显示所有可用的摄像头及其ID。通常：
- **ID 0**: 笔记本内置摄像头
- **ID 1**: USB摄像头（第一个USB摄像头）
- **ID 2+**: 其他USB摄像头

### 2. 启动实时追踪

**方法1: 使用配置文件中的摄像头ID（默认是1）**

```bash
conda run -n deepac python -m src_open.tools.live_tracking \
    --cfg /home/jyj/Projects/DeepAC/src_open/configs/live/cube_rgb.yaml
```

**方法2: 使用命令行参数覆盖摄像头ID**

如果USB摄像头不是ID 1，可以使用 `--camera_id` 参数指定：

```bash
conda run -n deepac python -m src_open.tools.live_tracking \
    --cfg /home/jyj/Projects/DeepAC/src_open/configs/live/cube_rgb.yaml \
    --camera_id 1
```

例如，如果USB摄像头是ID 2：
```bash
conda run -n deepac python -m src_open.tools.live_tracking \
    --cfg /home/jyj/Projects/DeepAC/src_open/configs/live/cube_rgb.yaml \
    --camera_id 2
```

### 3. 操作说明

- **自动开始追踪**: 程序会自动检测魔方并开始追踪（如果`auto_start_enabled: true`）
- **手动初始化**: 按 `s` 键使用当前引导位姿手动初始化追踪
- **退出程序**: 按 `q` 键或 `ESC` 键退出

### 4. 输出文件

- **视频文件**: 保存在 `workspace/live_cube_rgb/` 目录下
- **日志文件**: 保存在同一目录下

## 配置说明

### 主要参数（已优化以提高稳定性）

- **`fore_learn_rate: 0.02`**: 前景直方图学习率（已降低以提高稳定性）
- **`back_learn_rate: 0.02`**: 背景直方图学习率（已降低以提高稳定性）
- **`template_top_k: 30`**: 使用的模板视图数量（已增加以提高稳定性）
- **`smooth_alpha: 0.3`**: Pose平滑系数（已启用以减少抖动）
- **`resize: 384`**: 预处理图像尺寸
- **`crop_border: 5`**: 裁剪边界大小

### 与IR模式的差异

1. **图像类型**: RGB模式使用3通道彩色图像，IR模式使用单通道灰度图像
2. **学习率**: RGB模式下可以使用稍高的学习率（0.03 vs 0.02），因为图像质量更好
3. **相机内参**: 相同（同一相机）

## 故障排除

### 问题1: 追踪不稳定或漂移（已优化）

**当前配置已优化**:
- ✅ 学习率已降低：`fore_learn_rate: 0.02`, `back_learn_rate: 0.02`
- ✅ 模板数量已增加：`template_top_k: 30`
- ✅ Pose平滑已启用：`smooth_alpha: 0.3`

**如果仍有问题，可以进一步调整**:
- 进一步降低学习率：改为`0.015`或`0.01`
- 增加平滑系数：将`smooth_alpha`改为`0.5`或`0.7`（值越大越平滑，但可能增加延迟）
- 增加模板数量：将`template_top_k`改为`40`或`50`

### 问题2: 初始化失败

**解决方案**:
- 确保光照充足，魔方清晰可见
- 调整`init_depth`参数（默认0.45米）
- 手动按`s`键初始化

### 问题3: 追踪速度慢

**解决方案**:
- 降低`resize`参数（如改为`320`）
- 减少`template_top_k`（如改为`10`）
- 增加`detect_interval`（如改为`10`）

## 注意事项

1. **光照条件**: 确保光照充足，避免过暗或过亮
2. **背景**: 尽量使用对比度高的背景，便于追踪
3. **移动速度**: 避免过快移动魔方，可能导致追踪失败
4. **相机位置**: 保持相机稳定，避免剧烈晃动

