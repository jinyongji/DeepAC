# 使用RealSense RGB内参重新生成cube.pkl文件

## 问题分析

从您的描述看，之前用其他相机测试时，手动对齐后按s键，绿色框线能自动"吸附"到魔方边缘并稳定追踪。这说明追踪逻辑本身是工作的。

现在的问题是：
1. **cube.pkl文件是用不同的相机内参生成的**：预渲染模板视图时使用的虚拟相机内参可能与RealSense RGB相机的实际内参不匹配
2. **配置文件中的内参可能不准确**：当前配置使用的是`fx=1197.69228, fy=1196.28881`，这可能不是RealSense RGB的实际内参

## 解决方案

### 步骤1：获取RealSense RGB相机的实际内参

运行以下命令获取RealSense RGB相机的实际内参：

```bash
python src_open/tools/get_realsense_rgb_intrinsics.py 1280 720 30
```

这会输出RealSense RGB相机在1280x720@30fps下的实际内参。

### 步骤2：检查当前cube.pkl的内参

检查当前cube.pkl文件使用的内参：

```python
import pickle
with open('/media/jyj/JYJ/cube/cube/pre_render/cube.pkl', 'rb') as f:
    data = pickle.load(f)
    print("cube.pkl内参:")
    print(f"  fx={data['head']['fx']}")
    print(f"  fy={data['head']['fy']}")
    print(f"  cx={data['head']['cx']}")
    print(f"  cy={data['head']['cy']}")
    print(f"  image_size={data['head']['image_size']}")
```

### 步骤3：重新生成cube.pkl（如果需要）

**重要**：从`prerender.py`的代码看，预渲染时使用的虚拟相机内参是根据`sphere_radius`和`maximum_body_diameter`计算的，**不是直接使用实际相机内参**。

虚拟相机内参的计算公式：
```python
focal_length = (image_size - image_border_size) * sphere_radius / maximum_body_diameter
principal_point = image_size / 2
```

这意味着**cube.pkl的内参是虚拟的，不需要与实际相机内参完全匹配**。DeepAC会在运行时使用配置文件中的实际相机内参进行投影。

## 更可能的问题

根据代码分析，问题可能不是cube.pkl的内参，而是：

1. **配置文件中的内参不准确**：当前使用的`fx=1197.69228`等可能不是RealSense RGB的实际内参
2. **初始化位姿不正确**：即使蓝色框对齐了，初始化位姿的计算可能有问题
3. **追踪逻辑差异**：之前工作的版本和当前版本可能有逻辑差异

## 建议的调试步骤

### 1. 获取RealSense RGB的实际内参

```bash
python src_open/tools/get_realsense_rgb_intrinsics.py 1280 720 30
```

### 2. 更新配置文件中的内参

将获取到的实际内参更新到`cube_rgb_demo_style.yaml`：

```yaml
camera:
  fx: <实际fx值>
  fy: <实际fy值>
  cx: <实际cx值>
  cy: <实际cy值>
```

### 3. 使用自动检测功能

按`d`键进行自动检测，而不是手动对齐。自动检测会遍历多个模板视图，找到最佳匹配。

### 4. 检查初始化后的直方图

如果初始化后直方图区分度仍然很低，说明：
- 初始化位姿不正确
- 或者目标物体在图像中的对比度太低

## 总结

**不需要重新生成cube.pkl**，因为：
1. cube.pkl使用的是虚拟相机内参，DeepAC会在运行时使用实际相机内参
2. 问题更可能是配置文件中的内参不准确

**建议**：
1. 先获取RealSense RGB的实际内参
2. 更新配置文件中的内参
3. 使用自动检测功能（按`d`键）进行初始化
4. 如果仍然不工作，检查初始化后的直方图区分度





















