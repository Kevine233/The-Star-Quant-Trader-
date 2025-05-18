"""
回测报告生成器模块

提供回测结果的可视化和报告生成功能。

日期：2025-05-17
"""

from .report_generator import ReportGenerator, 报告生成器
from .visualization import create_performance_chart, create_parameter_distribution_chart

__all__ = [
    'ReportGenerator',
    '报告生成器',
    'create_performance_chart',
    'create_parameter_distribution_chart'
] 