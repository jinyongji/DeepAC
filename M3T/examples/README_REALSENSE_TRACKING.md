# RealSense 实时追踪程序使用说明

## 概述

`run_realsense_tracking.cpp` 是一个使用 Intel RealSense 相机进行实时 3D 物体追踪的程序。该程序专门设计用于追踪魔方（或其他立方体物体）。

## 功能特点

1. **初始参考模型渲染**：程序启动后会在相机前 20cm 处渲染一个参考模型
2. **手动对齐**：用户可以移动魔方或相机，将魔方与渲染的参考模型对齐
3. **自动检测**：对齐后按空格键开始检测和追踪
4. **实时追踪**：检测成功后自动开始实时追踪

## 编译

程序会在启用 `USE_REALSENSE` 选项时自动编译。确保 CMake 配置中已启用该选项。

```bash
cd /home/jyj/Projects/M3T
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_REALSENSE=ON -S . -B cmake-build-release
cmake --build cmake-build-release --target run_realsense_tracking
```

## 使用方法

### 1. 准备 OBJ 文件

确保您的 OBJ 文件位于指定路径：
```
/media/jyj/JYJ/RBOT-dataset/cube/cube_scaled.obj
```

**重要**：如果您的 OBJ 文件单位不是米，需要修改代码中的 `geometry_unit_in_meter` 参数：
- 如果 OBJ 文件单位是**毫米**（mm），设置为 `0.001f`
- 如果 OBJ 文件单位是**厘米**（cm），设置为 `0.01f`
- 如果 OBJ 文件单位是**米**（m），设置为 `1.0f`

### 2. 运行程序

```bash
./cmake-build-release/examples/run_realsense_tracking
```

### 3. 操作步骤

1. **等待初始化**：程序启动后会初始化相机和模型
2. **查看参考模型**：窗口中会显示一个渲染的参考模型（在相机前 20cm 处）
3. **对齐魔方**：移动魔方或相机，使魔方与参考模型大致对齐
4. **开始追踪**：按 **SPACE** 键开始检测和追踪
5. **退出程序**：按 **ESC** 键退出

## 参数配置

可以在代码中修改以下参数：

```cpp
const std::string body_name = "cube";  // 物体名称
const std::filesystem::path obj_file_path = "/media/jyj/JYJ/RBOT-dataset/cube/cube_scaled.obj";  // OBJ 文件路径
const float cube_size_m = 0.056f;  // 魔方尺寸（米），56mm = 0.056m
const float initial_distance = 0.2f;  // 初始距离（米），20cm = 0.2m
```

## 模型文件生成

程序首次运行时会自动生成以下模型文件（存储在 `/tmp/m3t_cube_models/`）：
- `cube_region_model.bin` - 区域模型
- `cube_depth_model.bin` - 深度模型

这些文件会在首次运行时自动生成，后续运行会直接加载，加快启动速度。

## 故障排除

### 1. OBJ 文件未找到
- 检查文件路径是否正确
- 确保文件存在且有读取权限

### 2. 相机初始化失败
- 确保 RealSense 相机已连接
- 检查相机驱动是否正确安装
- 确保没有其他程序占用相机

### 3. 模型生成失败
- 检查 `/tmp/m3t_cube_models/` 目录是否有写入权限
- 确保 OBJ 文件格式正确

### 4. 追踪效果不佳
- 确保光照充足
- 魔方表面应该有足够的纹理特征
- 尝试调整初始对齐位置

## 技术细节

- **相机**：使用 RealSense D435 颜色和深度相机
- **追踪模态**：区域模态（Region Modality）和深度模态（Depth Modality）
- **检测器**：静态检测器（Static Detector），使用预设的初始姿态
- **优化器**：使用迭代优化算法进行姿态优化

## 注意事项

1. 首次运行需要生成模型文件，可能需要一些时间
2. 确保相机前有足够的空间（至少 20cm）
3. 追踪效果取决于光照条件和物体表面特征
4. 如果追踪丢失，可以重新对齐并按空格键重新开始






