#!/bin/bash
# Cat模型IR追踪评估脚本

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

