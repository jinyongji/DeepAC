# 增强版IR追踪系统使用说明

## 一、系统概述

本系统是基于DeepAC的增强版IR图像物体追踪系统，专门针对RealSense IR相机和魔方追踪进行了优化。系统整合了三个核心模块：

1. **梯度直方图模块**：计算IR图像的梯度直方图，增强前景/背景区分度
2. **轻量级特征检测网络**：生成特征点概率图，提供额外的特征约束
3. **轮廓线分布提取器**：基于Line Distribution Extractor获取轮廓线概率分布

三个模块的结果共同输入优化引擎，实现鲁棒的6D位姿估计。

## 二、系统架构

### 2.1 核心模块

#### （1）梯度直方图模块 (`GradientHistogram`)
- **位置**：`src_open/models/gradient_histogram.py`
- **功能**：
  - 计算IR图像的Sobel梯度
  - 融合原始直方图和梯度直方图
  - 提高前景/背景区分度
- **关键参数**：
  - `use_gradient`: 是否使用梯度直方图
  - `gradient_weight`: 梯度直方图权重（默认0.5）

#### （2）轻量级特征检测网络 (`LightweightKeypointDetector`)
- **位置**：`src_open/models/keypoint_detector.py`
- **功能**：
  - 使用轻量级backbone（MobileNetV2/EfficientNet-Lite/Simple）提取特征
  - 生成特征点概率图
  - 提供额外的特征约束
- **关键参数**：
  - `backbone`: 可选 'mobilenet_v2', 'efficientnet_lite0', 'simple'
  - `num_keypoints`: 特征点数量（默认64）
  - `in_channels`: 输入通道数（IR图像为1）

#### （3）增强版DeepAC模型 (`DeepACIREnhanced`)
- **位置**：`src_open/models/deep_ac_ir_enhanced.py`
- **功能**：
  - 整合三个模块的结果
  - 增强的轮廓特征提取器
  - 优化的位姿估计
- **关键参数**：
  - `use_gradient_histogram`: 是否使用梯度直方图
  - `use_keypoint_detector`: 是否使用特征点检测
  - `keypoint_weight`: 特征点权重（默认0.3）

### 2.2 追踪程序

#### 增强版IR追踪程序 (`live_tracking_ir_enhanced.py`)
- **位置**：`src_open/tools/live_tracking_ir_enhanced.py`
- **功能**：
  - RealSense IR相机初始化和管理
  - 预渲染模型对齐引导
  - 实时追踪和可视化
  - 位姿估计和更新

## 三、安装和配置

### 3.1 依赖安装

```bash
# 激活conda环境
conda activate deepac

# 安装pyrealsense2（如果未安装）
pip install pyrealsense2

# 确保已安装其他依赖
pip install -r requirements.txt
```

### 3.2 配置文件

配置文件位于：`src_open/configs/live/cube_ir_enhanced.yaml`

主要配置项：
```yaml
model:
  model_name: DeepACIREnhanced  # 使用增强模型
  load_cfg: <训练配置路径>
  load_model: <模型权重路径>

camera:
  use_realsense_ir: true
  ir_index: 2  # 1=左IR相机, 2=右IR相机

tracking:
  use_gradient_histogram: true  # 启用梯度直方图
  use_keypoint_detector: true    # 启用特征点检测
  gradient_weight: 0.5           # 梯度权重
  keypoint_weight: 0.3           # 特征点权重
  grayscale: true                # IR图像是灰度图
```

## 四、使用方法

### 4.1 启动追踪程序

```bash
conda activate deepac
cd /home/jyj/Projects/DeepAC
python -m src_open.tools.live_tracking_ir_enhanced \
    --cfg src_open/configs/live/cube_ir_enhanced.yaml
```

### 4.2 操作流程

1. **程序启动**
   - 自动初始化RealSense IR相机
   - 显示引导框（黄色轮廓）
   - 等待用户对齐魔方

2. **对齐阶段**
   - 将魔方与引导框对齐
   - 观察IoU值（如果启用自动检测）
   - 按**SPACE键**开始追踪

3. **追踪阶段**
   - 自动开始6D位姿估计
   - 实时显示追踪结果（绿色轮廓）
   - 按**Q键**退出

### 4.3 键盘控制

- **SPACE**: 开始追踪（在对齐阶段）
- **Q**: 退出程序

## 五、技术特点

### 5.1 梯度直方图增强

- **优势**：
  - IR图像中梯度信息对边缘检测很重要
  - 提高前景/背景区分度
  - 增强轮廓边界预测准确性

- **实现**：
  - 使用Sobel算子计算梯度
  - 融合原始直方图和梯度直方图
  - 可配置融合权重

### 5.2 特征点检测增强

- **优势**：
  - 提供额外的特征约束
  - 增强追踪鲁棒性
  - 轻量级设计，不影响实时性

- **实现**：
  - 使用轻量级backbone提取特征
  - 生成特征点概率图
  - 融合到轮廓特征中

### 5.3 多模块融合优化

- **优势**：
  - 三个模块互补，提高追踪精度
  - 梯度直方图提供边缘信息
  - 特征点检测提供局部特征
  - 轮廓线分布提供全局约束

## 六、性能优化建议

### 6.1 参数调优

1. **梯度权重** (`gradient_weight`)
   - 默认值：0.5
   - 如果IR图像边缘清晰，可以增大（0.6-0.7）
   - 如果图像噪声较大，可以减小（0.3-0.4）

2. **特征点权重** (`keypoint_weight`)
   - 默认值：0.3
   - 如果特征点检测效果好，可以增大（0.4-0.5）
   - 如果特征点检测不稳定，可以减小（0.1-0.2）

3. **特征点数量** (`num_keypoints`)
   - 默认值：64
   - 如果计算资源充足，可以增加（128-256）
   - 如果追求实时性，可以减少（32-48）

### 6.2 模型选择

1. **特征检测backbone**
   - `simple`: 最轻量，适合实时追踪
   - `mobilenet_v2`: 平衡性能和精度
   - `efficientnet_lite0`: 精度最高，但计算量较大

2. **是否启用特征点检测**
   - 如果追踪效果已经很好，可以禁用以提升速度
   - 如果追踪不稳定，建议启用

## 七、故障排除

### 7.1 RealSense相机连接问题

**问题**：无法初始化RealSense IR相机

**解决方案**：
1. 检查相机连接：`lsusb | grep Intel`
2. 检查权限：确保用户有访问USB设备的权限
3. 重启RealSense服务：`sudo systemctl restart realsense-udev-rules`

### 7.2 追踪不稳定

**问题**：追踪过程中位姿抖动或丢失

**解决方案**：
1. 调整梯度权重：减小`gradient_weight`（如果噪声大）
2. 调整特征点权重：增大`keypoint_weight`（如果特征点检测效果好）
3. 检查光照条件：确保IR图像对比度足够
4. 检查魔方位置：确保魔方在相机视野内

### 7.3 内存不足

**问题**：程序运行时内存占用过高

**解决方案**：
1. 减少特征点数量：`num_keypoints: 32`
2. 使用简单backbone：`backbone: simple`
3. 减小图像分辨率：`resize: 256`

## 八、创新点总结

1. **梯度直方图融合**：针对IR图像特点，融合梯度信息提高区分度
2. **多模块协同**：三个模块互补，提高追踪精度和鲁棒性
3. **轻量级设计**：特征检测网络采用轻量级backbone，保证实时性
4. **灵活配置**：所有模块都可以独立启用/禁用，便于调优

## 九、未来改进方向

1. **自适应权重调整**：根据追踪质量动态调整各模块权重
2. **多尺度特征融合**：在不同尺度上融合特征点信息
3. **时序信息利用**：利用历史帧信息提高追踪稳定性
4. **深度信息融合**：如果可用，融合深度信息提高精度

## 十、参考文献

- DeepAC: Deep Active Contour for Object Tracking
- RealSense SDK: https://github.com/IntelRealSense/librealsense
- MobileNetV2: MobileNetV2: Inverted Residuals and Linear Bottlenecks

