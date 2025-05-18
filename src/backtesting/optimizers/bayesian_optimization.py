"""
贝叶斯优化器模块

实现基于贝叶斯优化的参数寻优算法，通过高斯过程建模目标函数，高效探索参数空间。

日期：2025-05-17
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import multiprocessing

from .optimizer_base import ParameterOptimizer, 参数优化器

# 配置日志
logger = logging.getLogger(__name__)

class BayesianOptimizer(ParameterOptimizer):
    """贝叶斯优化器，使用高斯过程回归建模目标函数"""
    
    def __init__(self, 
                engine,
                parallel_processes: int = None,
                logger_instance = None):
        """
        初始化贝叶斯优化器。
        
        参数:
            engine: 回测引擎实例，用于执行回测
            parallel_processes: 并行优化使用的进程数，默认为CPU核心数的一半
            logger_instance: 日志实例，默认使用模块级日志器
        """
        super().__init__(engine, parallel_processes, logger_instance)
        
        # 检查是否安装了scikit-optimize
        try:
            import skopt
            self.skopt = skopt
            self.logger.info("贝叶斯优化器初始化成功，使用scikit-optimize库")
        except ImportError:
            self.logger.warning("未安装scikit-optimize库，贝叶斯优化功能受限")
            self.logger.warning("请运行: pip install scikit-optimize")
            self.skopt = None
    
    def optimize(self, 
                strategy,
                market_data: pd.DataFrame,
                smart_money_data: pd.DataFrame,
                symbol: str,
                parameter_space: Dict[str, List[Any]],
                objective: str = 'calmar_ratio',
                n_calls: int = 20,
                n_random_starts: int = 5,
                acq_func: str = 'gp_hedge',
                **kwargs) -> List[Dict[str, Any]]:
        """
        使用贝叶斯优化方法优化策略参数。
        
        参数:
            strategy: 要优化的策略实例
            market_data: 市场数据DataFrame
            smart_money_data: 庄家指标数据DataFrame
            symbol: 交易标的代码
            parameter_space: 参数空间定义，格式为{参数名: [最小值, 最大值]}或{参数名: 可选值列表}
            objective: 优化目标指标，默认为'calmar_ratio'
            n_calls: 总的函数评估次数
            n_random_starts: 初始随机采样次数
            acq_func: 采集函数，可选'gp_hedge', 'EI', 'PI', 'LCB'
            
        返回:
            按目标指标排序的参数组合和结果列表
        """
        if self.skopt is None:
            self.logger.error("未安装scikit-optimize，无法执行贝叶斯优化")
            return []
        
        self.logger.info(f"开始贝叶斯优化，计划迭代次数: {n_calls}")
        
        # 将参数空间转换为贝叶斯优化需要的格式
        dimensions = []
        param_names = []
        
        for param_name, param_range in parameter_space.items():
            param_names.append(param_name)
            
            if isinstance(param_range, list) and len(param_range) == 2 and all(isinstance(x, (int, float)) for x in param_range):
                # 连续参数范围 [min, max]
                if all(isinstance(x, int) for x in param_range):
                    dimensions.append(self.skopt.space.Integer(*param_range))
                else:
                    dimensions.append(self.skopt.space.Real(*param_range))
            else:
                # 离散参数列表
                dimensions.append(self.skopt.space.Categorical(param_range))
        
        # 定义目标函数
        def objective_function(x):
            params = {param_names[i]: x[i] for i in range(len(param_names))}
            
            # 设置策略参数
            strategy_copy = strategy.clone() if hasattr(strategy, 'clone') else strategy
            strategy_copy.set_parameters(params)
            
            # 生成信号
            signal_data = strategy_copy.generate_signals(market_data, smart_money_data)
            
            # 执行回测
            backtest_results = self.engine.run_backtest(market_data, signal_data, symbol)
            
            # 获取目标指标值
            target_value = backtest_results.get(objective, backtest_results.get('calmar_ratio', 0.0))
            
            # 贝叶斯优化是最小化问题，所以对于需要最大化的指标，取负值
            if objective.lower() in ['max_drawdown', '最大回撤']:
                return target_value  # 最小化回撤
            else:
                return -target_value  # 最大化其他指标
        
        # 执行贝叶斯优化
        result = self.skopt.gp_minimize(
            objective_function,
            dimensions,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            acq_func=acq_func,
            verbose=True
        )
        
        # 将结果转换为标准格式
        results = []
        for i, x in enumerate(result.x_iters):
            params = {param_names[j]: x[j] for j in range(len(param_names))}
            
            strategy_copy = strategy.clone() if hasattr(strategy, 'clone') else strategy
            strategy_copy.set_parameters(params)
            signal_data = strategy_copy.generate_signals(market_data, smart_money_data)
            backtest_results = self.engine.run_backtest(market_data, signal_data, symbol)
            
            results.append({
                'parameters': params,
                'backtest_results': backtest_results
            })
        
        # 排序结果
        if objective.lower() in ['max_drawdown', '最大回撤']:
            # 对于回撤类指标，值越小越好
            results.sort(key=lambda x: x['backtest_results'].get(objective, x['backtest_results'].get('最大回撤', 1.0)), reverse=False)
        else:
            # 对于其他指标，值越大越好
            results.sort(key=lambda x: x['backtest_results'].get(objective, x['backtest_results'].get('收益回撤比', 0.0)), reverse=True)
        
        self.logger.info(f"贝叶斯优化完成，共{len(results)}个结果")
        if results:
            objective_value = results[0]['backtest_results'].get(objective, results[0]['backtest_results'].get('收益回撤比', 0.0))
            self.logger.info(f"最佳{objective}值: {objective_value:.4f}")
        
        return results


# 中文命名版本
class 贝叶斯优化器(参数优化器):
    """贝叶斯优化器，使用高斯过程回归建模目标函数（中文命名版本）"""
    
    def __init__(self, 
                回测引擎,
                并行进程数: int = None,
                日志器 = None):
        """
        初始化贝叶斯优化器。
        
        参数:
            回测引擎: 回测引擎实例，用于执行回测
            并行进程数: 并行优化使用的进程数，默认为CPU核心数的一半
            日志器: 日志实例，默认使用模块级日志器
        """
        super().__init__(回测引擎, 并行进程数, 日志器)
        
        # 检查是否安装了scikit-optimize
        try:
            import skopt
            self.skopt = skopt
            self.日志器.info("贝叶斯优化器初始化成功，使用scikit-optimize库")
        except ImportError:
            self.日志器.warning("未安装scikit-optimize库，贝叶斯优化功能受限")
            self.日志器.warning("请运行: pip install scikit-optimize")
            self.skopt = None
    
    def 优化(self, 
           策略,
           市场数据: pd.DataFrame,
           庄家数据: pd.DataFrame,
           股票代码: str,
           参数空间: Dict[str, List[Any]],
           目标指标: str = '收益回撤比',
           迭代次数: int = 20,
           随机起始次数: int = 5,
           采集函数: str = 'gp_hedge',
           **kwargs) -> List[Dict[str, Any]]:
        """
        使用贝叶斯优化方法优化策略参数。
        
        参数:
            策略: 要优化的策略实例
            市场数据: 市场数据DataFrame
            庄家数据: 庄家指标数据DataFrame
            股票代码: 交易标的代码
            参数空间: 参数空间定义，格式为{参数名: [最小值, 最大值]}或{参数名: 可选值列表}
            目标指标: 优化目标指标，默认为'收益回撤比'
            迭代次数: 总的函数评估次数
            随机起始次数: 初始随机采样次数
            采集函数: 采集函数，可选'gp_hedge', 'EI', 'PI', 'LCB'
            
        返回:
            按目标指标排序的参数组合和结果列表
        """
        if self.skopt is None:
            self.日志器.error("未安装scikit-optimize，无法执行贝叶斯优化")
            return []
        
        self.日志器.info(f"开始贝叶斯优化，计划迭代次数: {迭代次数}")
        
        # 将参数空间转换为贝叶斯优化需要的格式
        维度列表 = []
        参数名称 = []
        
        for 参数名, 参数范围 in 参数空间.items():
            参数名称.append(参数名)
            
            if isinstance(参数范围, list) and len(参数范围) == 2 and all(isinstance(x, (int, float)) for x in 参数范围):
                # 连续参数范围 [min, max]
                if all(isinstance(x, int) for x in 参数范围):
                    维度列表.append(self.skopt.space.Integer(*参数范围))
                else:
                    维度列表.append(self.skopt.space.Real(*参数范围))
            else:
                # 离散参数列表
                维度列表.append(self.skopt.space.Categorical(参数范围))
        
        # 定义目标函数
        def 目标函数(x):
            参数 = {参数名称[i]: x[i] for i in range(len(参数名称))}
            
            # 设置策略参数
            策略副本 = 策略.克隆() if hasattr(策略, '克隆') else 策略
            策略副本.设置参数(参数)
            
            # 生成信号
            信号数据 = 策略副本.生成信号(市场数据, 庄家数据)
            
            # 执行回测
            回测结果 = self.回测引擎.执行回测(市场数据, 信号数据, 股票代码)
            
            # 获取目标指标值
            目标值 = 回测结果.get(目标指标, 回测结果.get('收益回撤比', 0.0))
            
            # 贝叶斯优化是最小化问题，所以对于需要最大化的指标，取负值
            if 目标指标.lower() in ['max_drawdown', '最大回撤']:
                return 目标值  # 最小化回撤
            else:
                return -目标值  # 最大化其他指标
        
        # 执行贝叶斯优化
        结果 = self.skopt.gp_minimize(
            目标函数,
            维度列表,
            n_calls=迭代次数,
            n_random_starts=随机起始次数,
            acq_func=采集函数,
            verbose=True
        )
        
        # 将结果转换为标准格式
        优化结果 = []
        for i, x in enumerate(结果.x_iters):
            参数 = {参数名称[j]: x[j] for j in range(len(参数名称))}
            
            策略副本 = 策略.克隆() if hasattr(策略, '克隆') else 策略
            策略副本.设置参数(参数)
            信号数据 = 策略副本.生成信号(市场数据, 庄家数据)
            回测结果 = self.回测引擎.执行回测(市场数据, 信号数据, 股票代码)
            
            优化结果.append({
                '参数': 参数,
                '回测结果': 回测结果
            })
        
        # 排序结果
        if 目标指标.lower() in ['max_drawdown', '最大回撤']:
            # 对于回撤类指标，值越小越好
            优化结果.sort(key=lambda x: x['回测结果'].get(目标指标, x['回测结果'].get('最大回撤', 1.0)), reverse=False)
        else:
            # 对于其他指标，值越大越好
            优化结果.sort(key=lambda x: x['回测结果'].get(目标指标, x['回测结果'].get('收益回撤比', 0.0)), reverse=True)
        
        self.日志器.info(f"贝叶斯优化完成，共{len(优化结果)}个结果")
        if 优化结果:
            目标值 = 优化结果[0]['回测结果'].get(目标指标, 优化结果[0]['回测结果'].get('收益回撤比', 0.0))
            self.日志器.info(f"最佳{目标指标}值: {目标值:.4f}")
        
        return 优化结果 