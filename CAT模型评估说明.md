# Cat模型IR追踪评估说明

## 文件结构

```
/media/jyj/JYJ/cat/
├── frame/
│   └── IR/
│       ├── frame_000.png
│       ├── frame_001.png
│       ├── ...
│       └── pose_ir.txt          # GT位姿文件
└── pre_render/
    └── cat.pkl                  # 预渲染模板文件
```

## 相机内参

Cat数据集使用的IR相机内参（640x480分辨率）：
- `fu = 390.029`
- `fv = 390.029`
- `ppu = 320.854`
- `ppv = 241.846`

## 使用方法

### 方法1：使用评估脚本（推荐）

```bash
cd /home/jyj/Projects/DeepAC
./evaluate_cat_ir.sh
```

### 方法2：直接运行Python命令

```bash
cd /home/jyj/Projects/DeepAC
conda run -n deepac python src_open/tools/evaluate_ir_tracking.py \
    --image_dir /media/jyj/JYJ/cat/frame/IR \
    --gt_poses /media/jyj/JYJ/cat/frame/IR/pose_ir.txt \
    --cfg src_open/configs/live/cat_ir_tracking.yaml \
    --output_dir ./evaluation_results_cat \
    --add_threshold 0.05 \
    --rotation_threshold 5.0 \
    --gt_camera_fx 390.029 \
    --gt_camera_fy 390.029 \
    --gt_camera_cx 320.854 \
    --gt_camera_cy 241.846
```

## 参数说明

- `--image_dir`: IR图像目录（包含frame_*.png文件）
- `--gt_poses`: GT位姿文件路径（pose_ir.txt）
- `--cfg`: 配置文件路径（cat_ir_tracking.yaml）
- `--output_dir`: 输出目录（评估结果和可视化图像）
- `--add_threshold`: ADD阈值（米），默认0.05m（5cm）
- `--rotation_threshold`: 旋转阈值（度），默认5.0度
- `--gt_camera_fx/fy/cx/cy`: GT位姿使用的相机内参

## 输出文件

评估完成后，会在 `output_dir` 目录下生成：

1. **evaluation_results.json**: 评估结果摘要
   - 准确率指标（ADD、旋转、综合）
   - 平均误差、中位数误差、标准差

2. **predicted_poses.json**: 每帧的预测位姿
   - 格式：`{frame_idx: {'R': [...], 't': [...]}}`

3. **visualizations/**: 可视化图像目录
   - `frame_000.png`, `frame_001.png`, ...
   - 绿色框：预测位姿
   - 红色框：GT位姿
   - 显示误差信息和状态（PASS/FAIL）

## 评估指标

- **ADD(5cm)准确率**: ADD误差 ≤ 5cm 的帧占比
- **旋转(5°)准确率**: 旋转误差 ≤ 5° 的帧占比
- **综合准确率(ADD 5cm/5°)**: 同时满足ADD ≤ 5cm 且旋转 ≤ 5° 的帧占比

## 注意事项

1. **模型点云提取**: 程序会自动从 `cat.pkl` 文件中提取模型点云用于ADD误差计算
2. **相机内参**: 确保使用正确的相机内参（通过 `--gt_camera_*` 参数指定）
3. **图像格式**: 图像文件命名格式必须为 `frame_000.png`, `frame_001.png`, ...
4. **GT位姿格式**: GT位姿文件格式为：
   ```
   frame_index timestamp tx ty tz r00 r01 r02 r10 r11 r12 r20 r21 r22
   ```

## 配置文件

配置文件位于：`src_open/configs/live/cat_ir_tracking.yaml`

主要配置项：
- `object.pre_render_pkl`: 预渲染模板文件路径
- `camera.fx/fy/cx/cy`: 相机内参（会被GT内参覆盖）
- `tracking.*`: 追踪参数（学习率、模板数量等）

## 故障排除

1. **模型点云提取失败**: 如果无法从模板文件提取模型点云，程序会回退到默认的立方体模型点云
2. **图像加载失败**: 检查图像文件是否存在且格式正确
3. **GT位姿解析失败**: 检查pose_ir.txt文件格式是否正确
4. **相机内参不匹配**: 确保使用正确的相机内参，检查图像分辨率是否为640x480

