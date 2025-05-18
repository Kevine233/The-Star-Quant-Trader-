"""
回测引擎模块

提供各种回测引擎实现，用于评估交易策略的历史表现。

日期：2025-05-17
"""

from .backtest_engine import 回测引擎, BacktestEngine  # 支持新旧命名
from .performance_metrics import calculate_metrics

__all__ = [
    '回测引擎',
    'BacktestEngine',
    'calculate_metrics'
] 