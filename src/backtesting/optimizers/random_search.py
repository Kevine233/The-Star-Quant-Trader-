"""
随机搜索优化器模块

实现随机搜索优化算法，随机采样参数空间以寻找更优解，适合大型参数空间。

日期：2025-05-17
"""

import logging
import random
import multiprocessing
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

from .optimizer_base import ParameterOptimizer, 参数优化器

# 配置日志
logger = logging.getLogger(__name__)

class RandomSearchOptimizer(ParameterOptimizer):
    """随机搜索优化器，通过随机采样参数空间寻找最优组合"""
    
    def optimize(self, 
                strategy,
                market_data: pd.DataFrame,
                smart_money_data: pd.DataFrame,
                symbol: str,
                parameter_space: Dict[str, List[Any]],
                objective: str = 'calmar_ratio',
                n_iter: int = 20,
                parallel: bool = True,
                **kwargs) -> List[Dict[str, Any]]:
        """
        使用随机搜索方法优化策略参数。
        
        参数:
            strategy: 要优化的策略实例
            market_data: 市场数据DataFrame
            smart_money_data: 庄家指标数据DataFrame
            symbol: 交易标的代码
            parameter_space: 参数名到可能值列表的映射
            objective: 优化目标指标，默认为'calmar_ratio'
            n_iter: 随机迭代次数，默认为20
            parallel: 是否使用并行计算
            
        返回:
            按目标指标排序的参数组合和结果列表
        """
        total_space_size = self._calculate_parameter_space_size(parameter_space)
        actual_iter = min(n_iter, total_space_size)
        
        self.logger.info(f"开始随机搜索优化，参数空间大小: {total_space_size}, 计划迭代次数: {actual_iter}")
        
        # 准备随机任务
        tasks = []
        param_names = list(parameter_space.keys())
        
        for _ in range(actual_iter):
            # 随机采样参数
            random_params = {}
            for param_name in param_names:
                param_values = parameter_space[param_name]
                random_params[param_name] = random.choice(param_values)
            
            tasks.append((strategy, market_data, smart_money_data, symbol, random_params, objective))
        
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
        
        self.logger.info(f"随机搜索优化完成，共{len(results)}个结果")
        objective_value = results[0]['backtest_results'].get(objective, results[0]['backtest_results'].get('收益回撤比', 0.0))
        self.logger.info(f"最佳{objective}值: {objective_value:.4f}")
        
        return results


# 中文命名版本
class 随机搜索优化器(参数优化器):
    """随机搜索优化器，通过随机采样参数空间寻找最优组合（中文命名版本）"""
    
    def 优化(self, 
           策略,
           市场数据: pd.DataFrame,
           庄家数据: pd.DataFrame,
           股票代码: str,
           参数空间: Dict[str, List[Any]],
           目标指标: str = '收益回撤比',
           迭代次数: int = 20,
           并行: bool = True,
           **kwargs) -> List[Dict[str, Any]]:
        """
        使用随机搜索方法优化策略参数。
        
        参数:
            策略: 要优化的策略实例
            市场数据: 市场数据DataFrame
            庄家数据: 庄家指标数据DataFrame
            股票代码: 交易标的代码
            参数空间: 参数名到可能值列表的映射
            目标指标: 优化目标指标，默认为'收益回撤比'
            迭代次数: 随机迭代次数，默认为20
            并行: 是否使用并行计算
            
        返回:
            按目标指标排序的参数组合和结果列表
        """
        总空间大小 = self._计算参数空间大小(参数空间)
        实际迭代次数 = min(迭代次数, 总空间大小)
        
        self.日志器.info(f"开始随机搜索优化，参数空间大小: {总空间大小}, 计划迭代次数: {实际迭代次数}")
        
        # 准备随机任务
        任务列表 = []
        参数名称 = list(参数空间.keys())
        
        for _ in range(实际迭代次数):
            # 随机采样参数
            随机参数 = {}
            for 参数名 in 参数名称:
                参数值列表 = 参数空间[参数名]
                随机参数[参数名] = random.choice(参数值列表)
            
            任务列表.append((策略, 市场数据, 庄家数据, 股票代码, 随机参数, 目标指标))
        
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
        
        self.日志器.info(f"随机搜索优化完成，共{len(结果)}个结果")
        目标值 = 结果[0]['回测结果'].get(目标指标, 结果[0]['回测结果'].get('收益回撤比', 0.0))
        self.日志器.info(f"最佳{目标指标}值: {目标值:.4f}")
        
        return 结果 