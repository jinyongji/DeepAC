# Cat模型参数确定指南

## 参数说明

### 1. `geometry_unit_in_meter`
- **含义**: OBJ文件中坐标的单位转换为米的比例因子
- **常见值**:
  - `0.001`: 如果OBJ文件中的单位是毫米（mm）
  - `0.01`: 如果OBJ文件中的单位是厘米（cm）
  - `1.0`: 如果OBJ文件中的单位是米（m）

### 2. `diameter_in_meter`
- **含义**: 模型的直径（米），即从模型中心到最远顶点的距离 × 2
- **用途**: 用于某些计算和初始化

## 方法1：使用Python脚本自动分析（推荐）

### 步骤1：运行分析脚本

```bash
cd /home/jyj/Projects/DeepAC
python scripts/analyze_obj_model.py /path/to/cat.obj
```

例如，如果cat.obj在`/media/jyj/JYJ/cat/`目录下：

```bash
python scripts/analyze_obj_model.py /media/jyj/JYJ/cat/cat.obj
```

### 步骤2：查看输出结果

脚本会输出：
- 模型的原始坐标范围
- 推断的几何单位
- 计算出的直径（米）
- 配置文件建议值

### 步骤3：如果自动推断的单位不正确

如果你知道模型的实际单位，可以手动指定：

```bash
# 如果模型单位是毫米
python scripts/analyze_obj_model.py /path/to/cat.obj --manual-unit 0.001

# 如果模型单位是厘米
python scripts/analyze_obj_model.py /path/to/cat.obj --manual-unit 0.01

# 如果模型单位是米
python scripts/analyze_obj_model.py /path/to/cat.obj --manual-unit 1.0
```

## 方法2：使用Meshlab手动测量

### 步骤1：打开模型

1. 打开Meshlab
2. File → Import Mesh → 选择`cat.obj`

### 步骤2：查看模型信息

1. 点击菜单栏的 **Filters** → **Quality Measures and Computations** → **Compute Geometric Measures**
2. 查看输出窗口中的信息：
   - **Bounding Box Size**: 边界框尺寸
   - **Bounding Box Diagonal**: 边界框对角线长度（这可以作为直径的参考）

### 步骤3：确定几何单位

查看模型的尺寸：
- 如果边界框尺寸大约是 **几十到几百**（例如50-200），很可能是**毫米**单位 → `geometry_unit_in_meter: 0.001`
- 如果边界框尺寸大约是 **几到几十**（例如5-20），很可能是**厘米**单位 → `geometry_unit_in_meter: 0.01`
- 如果边界框尺寸大约是 **零点几到几**（例如0.05-2），很可能是**米**单位 → `geometry_unit_in_meter: 1.0`

### 步骤4：计算直径

1. 在Meshlab中，选择 **Filters** → **Selection** → **Select All**
2. 查看 **Bounding Box Diagonal** 的值
3. 将这个值乘以`geometry_unit_in_meter`得到直径（米）

例如：
- 如果Bounding Box Diagonal = 150，且单位是毫米
- 则 `diameter_in_meter = 150 × 0.001 = 0.15` 米

## 方法3：使用Meshlab测量工具

### 步骤1：测量模型尺寸

1. 打开模型后，点击工具栏的 **Measure Tool**（尺子图标）
2. 在模型上点击两个最远的点
3. 查看测量结果

### 步骤2：确定单位

根据测量结果判断单位：
- 如果测量值大约是 **几十到几百** → 毫米单位
- 如果测量值大约是 **几到几十** → 厘米单位
- 如果测量值大约是 **零点几到几** → 米单位

## 示例

假设分析脚本输出：

```
模型尺寸（模型单位）:
  宽度 (X): 120.5
  高度 (Y): 85.3
  深度 (Z): 95.8
  最大尺寸: 120.5

推断的几何单位:
  geometry_unit_in_meter: 0.001
  单位类型: 毫米 (mm)

模型直径（米）:
  0.156 m (156 mm)
```

则配置文件应设置为：

```yaml
object:
  geometry_unit_in_meter: 0.001
  diameter_in_meter: 0.156
```

## 验证

运行分析脚本后，检查：
1. **模型尺寸是否合理**: 
   - 如果`diameter_in_meter`是0.15米（15厘米），这对于一个小型cat模型是合理的
   - 如果`diameter_in_meter`是1.5米，这可能太大了，需要检查单位是否正确

2. **与cube模型对比**:
   - cube模型的`diameter_in_meter: 0.056`（5.6厘米）
   - cat模型应该与cube模型尺寸相近（如果都是小型物体）

## 更新配置文件

确定参数后，更新`src_open/configs/live/cat_ir_tracking.yaml`:

```yaml
object:
  name: cat
  pre_render_pkl: /media/jyj/JYJ/cat/pre_render/cat.pkl
  geometry_unit_in_meter: 0.001  # 根据分析结果修改
  diameter_in_meter: 0.156        # 根据分析结果修改
```

## 常见问题

### Q: 如何知道模型单位是毫米还是厘米？
A: 查看模型的尺寸：
- 如果模型宽度是100单位，实际物体宽度是10厘米，则单位是毫米（100 × 0.001 = 0.1米 = 10厘米）
- 如果模型宽度是10单位，实际物体宽度是10厘米，则单位是厘米（10 × 0.01 = 0.1米 = 10厘米）

### Q: diameter_in_meter必须精确吗？
A: 不需要非常精确，但应该接近实际值。这个参数主要用于某些初始化计算，误差在10-20%以内通常可以接受。

### Q: 如果模型尺寸很大（>1米）或很小（<1厘米）怎么办？
A: 检查单位是否正确。如果确认单位正确，使用计算出的值即可。

