"""
参数优化器基类模块

定义参数优化器的基本接口，为各种优化算法提供统一的标准。

日期：2025-05-17
"""

import logging
import multiprocessing
from typing import Dict, List, Tuple, Any, Callable, Optional, Union
import pandas as pd

# 配置日志
logger = logging.getLogger(__name__)

class ParameterOptimizer:
    """参数优化器基类（英文命名版本）"""
    
    def __init__(self, 
                engine,
                parallel_processes: int = None,
                logger_instance = None):
        """
        初始化参数优化器。
        
        参数:
            engine: 回测引擎实例，用于执行回测
            parallel_processes: 并行优化使用的进程数，默认为CPU核心数的一半
            logger_instance: 日志实例，默认使用模块级日志器
        """
        self.engine = engine
        self.parallel_processes = parallel_processes if parallel_processes is not None else max(1, multiprocessing.cpu_count() // 2)
        self.logger = logger_instance or logger
        
        self.logger.info(f"参数优化器初始化，并行进程数: {self.parallel_processes}")
        
    def optimize(self, 
                strategy,
                market_data: pd.DataFrame,
                smart_money_data: pd.DataFrame,
                symbol: str,
                parameter_space: Dict,
                objective: str = 'calmar_ratio',
                **kwargs) -> List[Dict[str, Any]]:
        """
        优化策略参数（基类方法）。
        
        参数:
            strategy: 要优化的策略实例
            market_data: 市场数据DataFrame
            smart_money_data: 庄家指标数据DataFrame
            symbol: 交易标的代码
            parameter_space: 参数空间定义
            objective: 优化目标指标，默认为'calmar_ratio'
            **kwargs: 特定优化算法的额外参数
            
        返回:
            按目标指标排序的参数组合和结果列表
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def _execute_single_backtest(self, task_data: Tuple) -> Dict[str, Any]:
        """
        执行单次回测任务。
        
        参数:
            task_data: (策略实例, 市场数据, 庄家数据, 股票代码, 参数字典, 目标指标)的元组
            
        返回:
            包含参数和回测结果的字典
        """
        strategy_instance, market_data, smart_money_data, symbol, params, _ = task_data
        
        # 设置策略参数
        strategy_instance.set_parameters(params)
        
        # 生成信号
        signal_data = strategy_instance.generate_signals(market_data, smart_money_data)
        
        # 执行回测
        backtest_results = self.engine.run_backtest(market_data, signal_data, symbol)
        
        return {
            'parameters': params,
            'backtest_results': backtest_results
        }
    
    def _calculate_parameter_space_size(self, parameter_grid: Dict[str, List[Any]]) -> int:
        """计算参数空间的大小。"""
        size = 1
        for value_list in parameter_grid.values():
            size *= len(value_list)
        return size


# 中文命名版本，功能完全相同
class 参数优化器:
    """参数优化器基类（中文命名版本）"""
    
    def __init__(self, 
                回测引擎,
                并行进程数: int = None,
                日志器 = None):
        """
        初始化参数优化器。
        
        参数:
            回测引擎: 回测引擎实例，用于执行回测
            并行进程数: 并行优化使用的进程数，默认为CPU核心数的一半
            日志器: 日志实例，默认使用模块级日志器
        """
        self.回测引擎 = 回测引擎
        self.并行进程数 = 并行进程数 if 并行进程数 is not None else max(1, multiprocessing.cpu_count() // 2)
        self.日志器 = 日志器 or logger
        
        self.日志器.info(f"参数优化器初始化，并行进程数: {self.并行进程数}")
        
    def 优化(self, 
           策略,
           市场数据: pd.DataFrame,
           庄家数据: pd.DataFrame,
           股票代码: str,
           参数空间: Dict,
           目标指标: str = '收益回撤比',
           **kwargs) -> List[Dict[str, Any]]:
        """
        优化策略参数（基类方法）。
        
        参数:
            策略: 要优化的策略实例
            市场数据: 市场数据DataFrame
            庄家数据: 庄家指标数据DataFrame
            股票代码: 交易标的代码
            参数空间: 参数空间定义
            目标指标: 优化目标指标，默认为'收益回撤比'
            **kwargs: 特定优化算法的额外参数
            
        返回:
            按目标指标排序的参数组合和结果列表
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def _执行单次回测(self, 任务数据: Tuple) -> Dict[str, Any]:
        """
        执行单次回测任务。
        
        参数:
            任务数据: (策略实例, 市场数据, 庄家数据, 股票代码, 参数字典, 目标指标)的元组
            
        返回:
            包含参数和回测结果的字典
        """
        策略实例, 市场数据, 庄家数据, 股票代码, 参数, _ = 任务数据
        
        # 设置策略参数
        策略实例.设置参数(参数)
        
        # 生成信号
        信号数据 = 策略实例.生成信号(市场数据, 庄家数据)
        
        # 执行回测
        回测结果 = self.回测引擎.执行回测(市场数据, 信号数据, 股票代码)
        
        return {
            '参数': 参数,
            '回测结果': 回测结果
        }
    
    def _计算参数空间大小(self, 参数网格: Dict[str, List[Any]]) -> int:
        """计算参数空间的大小。"""
        大小 = 1
        for 值列表 in 参数网格.values():
            大小 *= len(值列表)
        return 大小