"""
回测模块，用于评估交易策略的历史表现。
本模块支持一键回测和自动参数优化功能。

日期：2025-05-17
"""

# 导入回测引擎
from .engines import 回测引擎, BacktestEngine, calculate_metrics

# 导入参数优化器
from .optimizers import (
    参数优化器, ParameterOptimizer,
    GridSearchOptimizer, 网格搜索优化器,
    RandomSearchOptimizer, 随机搜索优化器,
    BayesianOptimizer, 贝叶斯优化器,
    GeneticOptimizer, 遗传算法优化器,
    ParticleSwarmOptimizer, 粒子群优化器,
    SimulatedAnnealingOptimizer, 模拟退火优化器,
    MultiObjectiveOptimizer, 多目标优化器
)

# 导入报告生成器
from .reporters import (
    ReportGenerator, 报告生成器,
    create_performance_chart,
    create_parameter_distribution_chart
)

# 导入一键回测功能
from .one_click import OneClickBacktest, 一键回测器

__all__ = [
    # 回测引擎
    '回测引擎',
    'BacktestEngine',
    'calculate_metrics',
    
    # 参数优化器
    '参数优化器', 
    'ParameterOptimizer',
    'GridSearchOptimizer',
    '网格搜索优化器',
    'RandomSearchOptimizer',
    '随机搜索优化器',
    'BayesianOptimizer',
    '贝叶斯优化器',
    'GeneticOptimizer',
    '遗传算法优化器',
    'ParticleSwarmOptimizer',
    '粒子群优化器',
    'SimulatedAnnealingOptimizer',
    '模拟退火优化器',
    'MultiObjectiveOptimizer',
    '多目标优化器',
    
    # 报告生成器
    'ReportGenerator',
    '报告生成器',
    'create_performance_chart',
    'create_parameter_distribution_chart',
    
    # 一键回测功能
    'OneClickBacktest',
    '一键回测器'
]
