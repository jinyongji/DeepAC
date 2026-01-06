#!/usr/bin/env python3
"""
分析OBJ模型文件，计算模型尺寸和几何单位
用于确定geometry_unit_in_meter和diameter_in_meter参数
"""
import sys
from pathlib import Path
import numpy as np
import argparse

def parse_obj_file(obj_path):
    """
    解析OBJ文件，提取顶点坐标
    
    Args:
        obj_path: OBJ文件路径
    
    Returns:
        vertices: 顶点坐标数组 (N, 3)
    """
    vertices = []
    
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) == 0:
                continue
            
            if parts[0] == 'v':  # 顶点
                if len(parts) >= 4:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        vertices.append([x, y, z])
                    except ValueError:
                        continue
    
    return np.array(vertices, dtype=np.float32)


def calculate_model_dimensions(vertices):
    """
    计算模型的尺寸信息
    
    Args:
        vertices: 顶点坐标数组 (N, 3)
    
    Returns:
        dict: 包含尺寸信息的字典
    """
    if len(vertices) == 0:
        return None
    
    # 计算边界框
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    
    # 尺寸（各轴的长度）
    size = max_coords - min_coords
    
    # 中心点
    center = (min_coords + max_coords) / 2.0
    
    # 从中心到最远点的距离（用于估计直径）
    distances_from_center = np.linalg.norm(vertices - center, axis=1)
    max_distance = distances_from_center.max()
    diameter = max_distance * 2.0  # 直径 = 2 * 最大距离
    
    # 计算几何单位（假设模型单位）
    # 如果模型尺寸很大（>100），可能是毫米单位
    # 如果模型尺寸很小（<1），可能是米单位
    # 如果模型尺寸中等（1-100），可能是厘米单位
    
    max_size = size.max()
    if max_size > 100:
        geometry_unit = 0.001  # 毫米单位（1单位 = 0.001米）
        unit_name = "毫米 (mm)"
    elif max_size < 1:
        geometry_unit = 1.0  # 米单位（1单位 = 1米）
        unit_name = "米 (m)"
    else:
        geometry_unit = 0.01  # 厘米单位（1单位 = 0.01米）
        unit_name = "厘米 (cm)"
    
    # 转换为米
    size_m = size * geometry_unit
    diameter_m = diameter * geometry_unit
    center_m = center * geometry_unit
    
    return {
        'num_vertices': len(vertices),
        'min_coords': min_coords,
        'max_coords': max_coords,
        'size': size,
        'size_m': size_m,
        'center': center,
        'center_m': center_m,
        'diameter': diameter,
        'diameter_m': diameter_m,
        'geometry_unit': geometry_unit,
        'unit_name': unit_name,
        'max_size': max_size,
    }


def print_results(info):
    """打印分析结果"""
    print("\n" + "="*70)
    print("OBJ模型尺寸分析结果")
    print("="*70)
    print(f"\n顶点数量: {info['num_vertices']}")
    
    print(f"\n原始坐标范围（模型单位）:")
    print(f"  X: [{info['min_coords'][0]:.6f}, {info['max_coords'][0]:.6f}]")
    print(f"  Y: [{info['min_coords'][1]:.6f}, {info['max_coords'][1]:.6f}]")
    print(f"  Z: [{info['min_coords'][2]:.6f}, {info['max_coords'][2]:.6f}]")
    
    print(f"\n模型尺寸（模型单位）:")
    print(f"  宽度 (X): {info['size'][0]:.6f}")
    print(f"  高度 (Y): {info['size'][1]:.6f}")
    print(f"  深度 (Z): {info['size'][2]:.6f}")
    print(f"  最大尺寸: {info['max_size']:.6f}")
    
    print(f"\n推断的几何单位:")
    print(f"  geometry_unit_in_meter: {info['geometry_unit']:.6f}")
    print(f"  单位类型: {info['unit_name']}")
    
    print(f"\n模型尺寸（米）:")
    print(f"  宽度 (X): {info['size_m'][0]:.6f} m ({info['size_m'][0]*1000:.2f} mm)")
    print(f"  高度 (Y): {info['size_m'][1]:.6f} m ({info['size_m'][1]*1000:.2f} mm)")
    print(f"  深度 (Z): {info['size_m'][2]:.6f} m ({info['size_m'][2]*1000:.2f} mm)")
    
    print(f"\n模型直径（从中心到最远点的距离 × 2）:")
    print(f"  模型单位: {info['diameter']:.6f}")
    print(f"  米: {info['diameter_m']:.6f} m ({info['diameter_m']*1000:.2f} mm)")
    
    print(f"\n模型中心（米）:")
    print(f"  X: {info['center_m'][0]:.6f} m")
    print(f"  Y: {info['center_m'][1]:.6f} m")
    print(f"  Z: {info['center_m'][2]:.6f} m")
    
    print("\n" + "="*70)
    print("配置文件建议值:")
    print("="*70)
    print(f"geometry_unit_in_meter: {info['geometry_unit']:.6f}")
    print(f"diameter_in_meter: {info['diameter_m']:.6f}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="分析OBJ模型文件，计算尺寸和几何单位")
    parser.add_argument("obj_file", type=str, help="OBJ文件路径")
    parser.add_argument("--manual-unit", type=float, default=None,
                        help="手动指定几何单位（米），例如：0.001表示毫米，0.01表示厘米，1.0表示米")
    
    args = parser.parse_args()
    
    obj_path = Path(args.obj_file)
    if not obj_path.exists():
        print(f"错误: 文件不存在: {obj_path}")
        return
    
    print(f"正在读取OBJ文件: {obj_path}")
    vertices = parse_obj_file(obj_path)
    
    if len(vertices) == 0:
        print("错误: 未找到任何顶点")
        return
    
    print(f"成功读取 {len(vertices)} 个顶点")
    
    info = calculate_model_dimensions(vertices)
    
    # 如果手动指定了单位，使用手动指定的单位
    if args.manual_unit is not None:
        print(f"\n使用手动指定的几何单位: {args.manual_unit} 米")
        info['geometry_unit'] = args.manual_unit
        info['size_m'] = info['size'] * info['geometry_unit']
        info['diameter_m'] = info['diameter'] * info['geometry_unit']
        info['center_m'] = info['center'] * info['geometry_unit']
        
        if args.manual_unit == 0.001:
            info['unit_name'] = "毫米 (mm)"
        elif args.manual_unit == 0.01:
            info['unit_name'] = "厘米 (cm)"
        elif args.manual_unit == 1.0:
            info['unit_name'] = "米 (m)"
        else:
            info['unit_name'] = f"自定义 ({args.manual_unit} 米/单位)"
    
    print_results(info)
    
    # 保存结果到文件
    output_file = obj_path.parent / f"{obj_path.stem}_analysis.txt"
    with open(output_file, 'w') as f:
        f.write("OBJ模型尺寸分析结果\n")
        f.write("="*70 + "\n")
        f.write(f"\n顶点数量: {info['num_vertices']}\n")
        f.write(f"\n原始坐标范围（模型单位）:\n")
        f.write(f"  X: [{info['min_coords'][0]:.6f}, {info['max_coords'][0]:.6f}]\n")
        f.write(f"  Y: [{info['min_coords'][1]:.6f}, {info['max_coords'][1]:.6f}]\n")
        f.write(f"  Z: [{info['min_coords'][2]:.6f}, {info['max_coords'][2]:.6f}]\n")
        f.write(f"\n模型尺寸（模型单位）:\n")
        f.write(f"  宽度 (X): {info['size'][0]:.6f}\n")
        f.write(f"  高度 (Y): {info['size'][1]:.6f}\n")
        f.write(f"  深度 (Z): {info['size'][2]:.6f}\n")
        f.write(f"\n配置文件建议值:\n")
        f.write(f"geometry_unit_in_meter: {info['geometry_unit']:.6f}\n")
        f.write(f"diameter_in_meter: {info['diameter_m']:.6f}\n")
    
    print(f"\n分析结果已保存到: {output_file}")


if __name__ == "__main__":
    main()

