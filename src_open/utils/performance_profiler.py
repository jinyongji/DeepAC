"""
性能分析工具
用于统计各模块的耗时，帮助识别性能瓶颈
"""
import time
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional
import contextlib


class PerformanceProfiler:
    """性能分析器，用于统计各模块耗时"""
    
    def __init__(self, enabled: bool = True):
        """
        初始化性能分析器
        
        Args:
            enabled: 是否启用性能分析
        """
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.current_timings: Dict[str, float] = {}
        self.frame_count = 0
        
    def reset(self):
        """重置所有统计数据"""
        self.timings.clear()
        self.current_timings.clear()
        self.frame_count = 0
    
    @contextlib.contextmanager
    def profile(self, name: str):
        """
        上下文管理器，用于统计代码块耗时
        
        Usage:
            with profiler.profile("edge_extraction"):
                # 边缘提取代码
                pass
        """
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = (time.time() - start_time) * 1000  # 转换为毫秒
            self.timings[name].append(elapsed)
            self.current_timings[name] = elapsed
    
    def start_frame(self):
        """开始新的一帧"""
        if self.enabled:
            self.frame_count += 1
            self.current_timings.clear()
    
    def get_stats(self, module_name: str) -> Dict[str, float]:
        """
        获取指定模块的统计信息
        
        Args:
            module_name: 模块名称
            
        Returns:
            包含avg, min, max, std的字典（单位：毫秒）
        """
        if not self.enabled or module_name not in self.timings:
            return {"avg": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "count": 0}
        
        timings = self.timings[module_name]
        if len(timings) == 0:
            return {"avg": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "count": 0}
        
        timings_array = np.array(timings)
        return {
            "avg": float(np.mean(timings_array)),
            "min": float(np.min(timings_array)),
            "max": float(np.max(timings_array)),
            "std": float(np.std(timings_array)),
            "count": len(timings),
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        获取所有模块的统计信息
        
        Returns:
            字典，key为模块名，value为统计信息
        """
        if not self.enabled:
            return {}
        
        stats = {}
        for module_name in self.timings.keys():
            stats[module_name] = self.get_stats(module_name)
        return stats
    
    def get_current_timings(self) -> Dict[str, float]:
        """获取当前帧的耗时（最近一次记录的）"""
        return self.current_timings.copy()
    
    def print_summary(self, top_k: int = 10):
        """
        打印性能摘要
        
        Args:
            top_k: 显示耗时最长的前k个模块
        """
        if not self.enabled:
            print("Performance profiling is disabled")
            return
        
        print("\n" + "="*60)
        print("Performance Profiling Summary")
        print("="*60)
        print(f"Total frames: {self.frame_count}")
        print(f"\nTop {top_k} modules by average time:")
        print("-"*60)
        print(f"{'Module':<30} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'Std(ms)':>10} {'Count':>8}")
        print("-"*60)
        
        all_stats = self.get_all_stats()
        # 按平均耗时排序
        sorted_modules = sorted(
            all_stats.items(),
            key=lambda x: x[1]["avg"],
            reverse=True
        )[:top_k]
        
        for module_name, stats in sorted_modules:
            print(
                f"{module_name:<30} "
                f"{stats['avg']:>10.2f} "
                f"{stats['min']:>10.2f} "
                f"{stats['max']:>10.2f} "
                f"{stats['std']:>10.2f} "
                f"{stats['count']:>8}"
            )
        print("="*60 + "\n")
    
    def save_to_file(self, filepath: str):
        """
        保存统计信息到文件
        
        Args:
            filepath: 输出文件路径
        """
        if not self.enabled:
            return
        
        with open(filepath, "w") as f:
            f.write("Performance Profiling Results\n")
            f.write("="*60 + "\n")
            f.write(f"Total frames: {self.frame_count}\n\n")
            
            all_stats = self.get_all_stats()
            sorted_modules = sorted(
                all_stats.items(),
                key=lambda x: x[1]["avg"],
                reverse=True
            )
            
            for module_name, stats in sorted_modules:
                f.write(f"\n{module_name}:\n")
                f.write(f"  Average: {stats['avg']:.2f} ms\n")
                f.write(f"  Min:     {stats['min']:.2f} ms\n")
                f.write(f"  Max:     {stats['max']:.2f} ms\n")
                f.write(f"  Std:     {stats['std']:.2f} ms\n")
                f.write(f"  Count:   {stats['count']}\n")








