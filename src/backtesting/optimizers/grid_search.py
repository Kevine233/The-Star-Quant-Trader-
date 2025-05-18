"""
网格搜索优化器模块

实现网格搜索优化算法，遍历所有可能的参数组合以寻找最优解。

日期：2025-05-17
"""

import logging
import itertools
import multiprocessing
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd

from .optimizer_base import ParameterOptimizer, 参数优化器

# 配置日志
logger = logging.getLogger(__name__)

class GridSearchOptimizer(ParameterOptimizer):
    """网格搜索优化器，穷举所有可能的参数组合"""
    
    def optimize(self, 
                strategy,
                market_data: pd.DataFrame,
                smart_money_data: pd.DataFrame,
                symbol: str,
                parameter_grid: Dict[str, List[Any]],
                objective: str = 'calmar_ratio',
                parallel: bool = True,
                **kwargs) -> List[Dict[str, Any]]:
        """
        使用网格搜索方法优化策略参数。
        
        参数:
            strategy: 要优化的策略实例
            market_data: 市场数据DataFrame
            smart_money_data: 庄家指标数据DataFrame
            symbol: 交易标的代码
            parameter_grid: 参数名到可能值列表的映射
            objective: 优化目标指标，默认为'calmar_ratio'
            parallel: 是否使用并行计算
            
        返回:
            按目标指标排序的参数组合和结果列表
        """
        grid_size = self._calculate_parameter_space_size(parameter_grid)
        self.logger.info(f"开始网格搜索优化，参数空间大小: {grid_size}")
        
        # 生成所有参数组合
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        # 准备优化任务
        tasks = []
        for combination in param_combinations:
            param_dict = {param_names[i]: combination[i] for i in range(len(param_names))}
            tasks.append((strategy, market_data, smart_money_data, symbol, param_dict, objective))
        
        # 执行优化
        if parallel and len(tasks) > 1:
            with multiprocessing.Pool(self.parallel_processes) as pool:
                results = pool.map(self._execute_single_backtest, tasks)
        else:
            results = [self._execute_single_backtest(task) for task in tasks]
        
        # 按目标指标排序
        if objective.lower() in ['max_drawdown', '最大回撤']:
            # 对于回撤类指标，值越小越好
            results.sort(key=lambda x: x['backtest_results'].get(objective, x['backtest_results'].get('最大回撤', 1.0)), reverse=False)
        else:
            # 对于其他指标，值越大越好
            results.sort(key=lambda x: x['backtest_results'].get(objective, x['backtest_results'].get('收益回撤比', 0.0)), reverse=True)
        
        self.logger.info(f"网格搜索优化完成，共{len(results)}个结果")
        objective_value = results[0]['backtest_results'].get(objective, results[0]['backtest_results'].get('收益回撤比', 0.0))
        self.logger.info(f"最佳{objective}值: {objective_value:.4f}")
        
        return results


# 中文命名版本
class 网格搜索优化器(参数优化器):
    """网格搜索优化器，穷举所有可能的参数组合（中文命名版本）"""
    
    def 优化(self, 
           策略,
           市场数据: pd.DataFrame,
           庄家数据: pd.DataFrame,
           股票代码: str,
           参数网格: Dict[str, List[Any]],
           目标指标: str = '收益回撤比',
           并行: bool = True,
           **kwargs) -> List[Dict[str, Any]]:
        """
        使用网格搜索方法优化策略参数。
        
        参数:
            策略: 要优化的策略实例
            市场数据: 市场数据DataFrame
            庄家数据: 庄家指标数据DataFrame
            股票代码: 交易标的代码
            参数网格: 参数名到可能值列表的映射
            目标指标: 优化目标指标，默认为'收益回撤比'
            并行: 是否使用并行计算
            
        返回:
            按目标指标排序的参数组合和结果列表
        """
        网格大小 = self._计算参数空间大小(参数网格)
        self.日志器.info(f"开始网格搜索优化，参数空间大小: {网格大小}")
        
        # 生成所有参数组合
        参数名称 = list(参数网格.keys())
        参数值 = list(参数网格.values())
        参数组合 = list(itertools.product(*参数值))
        
        # 准备优化任务
        任务列表 = []
        for 组合 in 参数组合:
            参数字典 = {参数名称[i]: 组合[i] for i in range(len(参数名称))}
            任务列表.append((策略, 市场数据, 庄家数据, 股票代码, 参数字典, 目标指标))
        
        # 执行优化
        if 并行 and len(任务列表) > 1:
            with multiprocessing.Pool(self.并行进程数) as pool:
                结果 = pool.map(self._执行单次回测, 任务列表)
        else:
            结果 = [self._执行单次回测(任务) for 任务 in 任务列表]
        
        # 按目标指标排序
        if 目标指标.lower() in ['max_drawdown', '最大回撤']:
            # 对于回撤类指标，值越小越好
            结果.sort(key=lambda x: x['回测结果'].get(目标指标, x['回测结果'].get('最大回撤', 1.0)), reverse=False)
        else:
            # 对于其他指标，值越大越好
            结果.sort(key=lambda x: x['回测结果'].get(目标指标, x['回测结果'].get('收益回撤比', 0.0)), reverse=True)
        
        self.日志器.info(f"网格搜索优化完成，共{len(结果)}个结果")
        目标值 = 结果[0]['回测结果'].get(目标指标, 结果[0]['回测结果'].get('收益回撤比', 0.0))
        self.日志器.info(f"最佳{目标指标}值: {目标值:.4f}")
        
        return 结果 