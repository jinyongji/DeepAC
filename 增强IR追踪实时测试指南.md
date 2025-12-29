# 增强IR追踪系统实时测试指南

## 📋 测试前准备

### 1. 检查环境

```bash
# 1. 激活conda环境
conda activate deepac

# 2. 检查Python版本（建议3.8+）
python --version

# 3. 检查必要的依赖
python -c "import torch; import cv2; import pyrealsense2; print('All dependencies OK')"
```

如果缺少`pyrealsense2`，安装：
```bash
pip install pyrealsense2
```

### 2. 检查RealSense相机连接

```bash
# 检查USB设备
lsusb | grep Intel

# 检查RealSense设备（如果已安装驱动）
v4l2-ctl --list-devices 2>/dev/null || echo "v4l2-utils not installed"
```

**预期输出**：应该能看到Intel RealSense设备

### 3. 验证模型文件存在

```bash
# 检查模型配置文件
ls -lh /home/jyj/Projects/DeepAC/workspace/train_rbot_cube_gray/logs-*/train_cfg.yml

# 检查模型权重文件
ls -lh /home/jyj/Projects/DeepAC/workspace/train_rbot_cube_gray/last.ckpt

# 检查预渲染文件
ls -lh /media/jyj/JYJ/cube/cube/pre_render/cube.pkl
```

**注意**：如果路径不同，需要修改配置文件中的路径

## 🚀 快速开始测试

### 方法1：使用增强版IR追踪程序（推荐）

```bash
# 1. 进入项目目录
cd /home/jyj/Projects/DeepAC

# 2. 激活环境
conda activate deepac

# 3. 运行增强版IR追踪
python -m src_open.tools.live_tracking_ir_enhanced \
    --cfg src_open/configs/live/cube_ir_enhanced.yaml
```

### 方法2：使用原有程序（如果增强版有问题）

```bash
python -m src_open.tools.live_tracking \
    --cfg src_open/configs/live/cube_realsense_ir_simple.yaml
```

## 📝 测试步骤详解

### 步骤1：程序启动检查

程序启动后，你应该看到：

1. **终端输出**：
   ```
   [INFO] RealSense IR camera (index 2) initialized: 1280x720, fx=651.30, fy=651.30
   [INFO] Enhanced IR tracking started. Press SPACE to start tracking after alignment.
   ```

2. **窗口显示**：
   - 显示IR图像（灰度图）
   - 显示黄色引导框（轮廓线）
   - 显示提示文字："Align cube with guide, then press SPACE"

### 步骤2：对齐魔方

1. **准备魔方**：
   - 将魔方放在相机视野内
   - 确保魔方清晰可见
   - 建议距离相机约40-50cm

2. **对齐操作**：
   - 观察黄色引导框
   - 移动/旋转魔方使其与引导框对齐
   - 尽量让魔方的轮廓与引导框重合

3. **检查对齐质量**：
   - 如果配置了自动检测，会显示IoU值
   - IoU值越高，对齐越好（建议>0.5）

### 步骤3：开始追踪

1. **按SPACE键**开始追踪
2. **观察变化**：
   - 引导框从黄色变为绿色
   - 提示文字变为"Tracking..."
   - 轮廓线应该跟随魔方移动

### 步骤4：测试追踪性能

1. **缓慢移动**：
   - 左右移动魔方
   - 前后移动魔方
   - 旋转魔方

2. **观察追踪效果**：
   - ✅ **成功**：绿色轮廓线始终跟随魔方
   - ⚠️ **部分成功**：轮廓线偶尔偏离，但能恢复
   - ❌ **失败**：轮廓线丢失或大幅偏离

3. **测试不同场景**：
   - 不同光照条件
   - 不同背景
   - 不同角度

### 步骤5：退出程序

按 **Q键** 退出程序

## 🔧 常见问题排查

### 问题1：RealSense相机初始化失败

**错误信息**：
```
Failed to initialize RealSense IR camera: ...
```

**解决方法**：

1. **检查相机连接**：
   ```bash
   lsusb | grep Intel
   ```

2. **检查权限**：
   ```bash
   # 检查USB设备权限
   ls -l /dev/video*
   # 如果权限不足，添加用户到video组
   sudo usermod -a -G video $USER
   # 重新登录或执行
   newgrp video
   ```

3. **检查pyrealsense2安装**：
   ```bash
   python -c "import pyrealsense2 as rs; print(rs.__version__)"
   ```

4. **尝试不同的IR索引**：
   修改配置文件中的`ir_index`：
   ```yaml
   camera:
     ir_index: 1  # 尝试1或2
   ```

### 问题2：模型文件找不到

**错误信息**：
```
FileNotFoundError: ...
```

**解决方法**：

1. **检查配置文件路径**：
   编辑 `src_open/configs/live/cube_ir_enhanced.yaml`，确认路径正确：
   ```yaml
   model:
     load_cfg: <你的实际路径>/train_cfg.yml
     load_model: <你的实际路径>/last.ckpt
   ```

2. **使用绝对路径**：
   确保使用完整的绝对路径

### 问题3：追踪不稳定或丢失

**现象**：
- 轮廓线抖动
- 追踪突然丢失
- 位姿估计错误

**解决方法**：

1. **调整参数**（编辑配置文件）：
   ```yaml
   tracking:
     gradient_weight: 0.3  # 减小梯度权重（如果噪声大）
     keypoint_weight: 0.5  # 增大特征点权重（如果特征点检测好）
   ```

2. **检查光照条件**：
   - 确保IR图像对比度足够
   - 避免强光直射
   - 确保背景与魔方有足够对比度

3. **调整相机位置**：
   - 保持合适的距离（40-50cm）
   - 避免极端角度
   - 确保魔方始终在视野内

4. **禁用某些模块**（如果问题持续）：
   ```yaml
   tracking:
     use_keypoint_detector: false  # 禁用特征点检测
     use_gradient_histogram: false  # 禁用梯度直方图
   ```

### 问题4：程序运行缓慢

**现象**：
- 帧率低（<10 FPS）
- 延迟明显

**解决方法**：

1. **减少特征点数量**：
   ```yaml
   # 需要修改模型配置或代码
   keypoint_detector:
     num_keypoints: 32  # 从64减少到32
   ```

2. **使用简单backbone**：
   ```yaml
   keypoint_detector:
     backbone: simple  # 使用最简单的backbone
   ```

3. **降低图像分辨率**：
   ```yaml
   tracking:
     resize: 256  # 从384降低到256
   ```

4. **禁用某些模块**：
   ```yaml
   tracking:
     use_keypoint_detector: false  # 禁用特征点检测以提升速度
   ```

### 问题5：窗口无法显示

**现象**：
- 程序运行但没有窗口
- 显示相关错误

**解决方法**：

1. **检查显示环境**：
   ```bash
   echo $DISPLAY
   # 如果为空，可能需要设置
   export DISPLAY=:0
   ```

2. **使用SSH连接时**：
   ```bash
   # 启用X11转发
   ssh -X user@host
   ```

3. **检查OpenCV显示支持**：
   ```bash
   python -c "import cv2; print(cv2.getBuildInformation())"
   ```

## 📊 性能评估指标

### 追踪质量评估

1. **轮廓匹配度**：
   - 观察绿色轮廓线与魔方边缘的重合度
   - 理想情况：轮廓线紧贴魔方边缘

2. **位姿稳定性**：
   - 魔方静止时，轮廓线应该稳定
   - 不应该有明显抖动

3. **追踪鲁棒性**：
   - 快速移动时是否能跟上
   - 遮挡后是否能恢复
   - 不同角度下是否稳定

### 性能指标

1. **帧率（FPS）**：
   - 理想：>20 FPS
   - 可接受：10-20 FPS
   - 需要优化：<10 FPS

2. **延迟**：
   - 理想：<100ms
   - 可接受：100-200ms
   - 需要优化：>200ms

## 🎯 测试检查清单

### 启动前检查
- [ ] conda环境已激活
- [ ] RealSense相机已连接
- [ ] 模型文件路径正确
- [ ] 预渲染文件存在
- [ ] pyrealsense2已安装

### 功能测试
- [ ] 程序能正常启动
- [ ] 能显示IR图像
- [ ] 能显示引导框
- [ ] SPACE键能开始追踪
- [ ] 追踪时轮廓线跟随魔方
- [ ] Q键能正常退出

### 性能测试
- [ ] 帧率满足要求（>10 FPS）
- [ ] 延迟可接受（<200ms）
- [ ] 追踪稳定（无频繁丢失）
- [ ] 不同角度下都能追踪

### 鲁棒性测试
- [ ] 缓慢移动时追踪稳定
- [ ] 快速移动时能跟上
- [ ] 部分遮挡后能恢复
- [ ] 不同光照条件下稳定

## 📞 获取帮助

如果遇到问题：

1. **查看日志**：
   程序运行时的终端输出包含重要信息

2. **检查配置文件**：
   确认所有路径和参数正确

3. **简化测试**：
   先禁用某些模块，逐步启用：
   ```yaml
   tracking:
     use_keypoint_detector: false
     use_gradient_histogram: false
   ```

4. **参考文档**：
   - `增强IR追踪系统使用说明.md`
   - `RealSense_IR集成完成说明.md`

## 🎉 成功标志

如果以下条件都满足，说明测试成功：

1. ✅ 程序能正常启动并显示IR图像
2. ✅ 引导框显示正常
3. ✅ 按SPACE后能开始追踪
4. ✅ 绿色轮廓线能跟随魔方移动
5. ✅ 追踪稳定，无明显抖动或丢失
6. ✅ 帧率满足实时要求（>10 FPS）

祝测试顺利！🎊



















