"""
模拟退火优化器模块

实现基于模拟退火算法的参数优化方法，通过模拟金属退火过程搜索全局最优解。

日期：2025-05-17
"""

import logging
import random
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from copy import deepcopy

from .optimizer_base import ParameterOptimizer, 参数优化器

# 配置日志
logger = logging.getLogger(__name__)

class SimulatedAnnealingOptimizer(ParameterOptimizer):
    """模拟退火优化器，通过模拟退火过程寻找全局最优解"""
    
    def optimize(self, 
                strategy,
                market_data: pd.DataFrame,
                smart_money_data: pd.DataFrame,
                symbol: str,
                parameter_space: Dict[str, List[Any]],
                objective: str = 'calmar_ratio',
                initial_temp: float = 100.0,
                cooling_rate: float = 0.95,
                iterations: int = 100,
                **kwargs) -> List[Dict[str, Any]]:
        """
        使用模拟退火算法优化策略参数。
        
        参数:
            strategy: 要优化的策略实例
            market_data: 市场数据DataFrame
            smart_money_data: 庄家指标数据DataFrame
            symbol: 交易标的代码
            parameter_space: 参数名到可能值列表的映射
            objective: 优化目标指标，默认为'calmar_ratio'
            initial_temp: 初始温度，控制初始接受概率，默认为100.0
            cooling_rate: 冷却率，控制温度下降速度，默认为0.95
            iterations: 每个温度下的迭代次数，默认为100
            
        返回:
            按目标指标排序的参数组合和结果列表
        """
        param_names = list(parameter_space.keys())
        self.logger.info(f"开始模拟退火优化，初始温度: {initial_temp}, 冷却率: {cooling_rate}")
        
        # 生成初始解
        current_solution = self._generate_random_solution(parameter_space)
        current_params = self._solution_to_params(current_solution, parameter_space, param_names)
        
        # 评估初始解
        task = (strategy, market_data, smart_money_data, symbol, current_params, objective)
        current_result = self._execute_single_backtest(task)
        
        # 判断适应度计算方式
        is_minimize = objective.lower() in ['max_drawdown', '最大回撤']
        
        if is_minimize:
            # 对于回撤类指标，越小越好
            current_fitness = current_result['backtest_results'].get(objective, current_result['backtest_results'].get('最大回撤', 1.0))
        else:
            # 对于其他指标，越大越好
            current_fitness = current_result['backtest_results'].get(objective, current_result['backtest_results'].get('收益回撤比', 0.0))
        
        # 设置当前解为最优解
        best_solution = deepcopy(current_solution)
        best_params = deepcopy(current_params)
        best_fitness = current_fitness
        best_result = deepcopy(current_result)
        
        # 存储所有评估结果
        all_evaluations = [{
            'parameters': current_params,
            'fitness': current_fitness,
            'backtest_results': current_result['backtest_results']
        }]
        
        # 主循环
        temperature = initial_temp
        iteration = 0
        
        while temperature > 0.1:
            self.logger.info(f"当前温度: {temperature:.2f}")
            
            for _ in range(iterations):
                # 生成新解
                new_solution = self._generate_neighbor(current_solution, parameter_space)
                new_params = self._solution_to_params(new_solution, parameter_space, param_names)
                
                # 评估新解
                task = (strategy, market_data, smart_money_data, symbol, new_params, objective)
                new_result = self._execute_single_backtest(task)
                
                if is_minimize:
                    # 对于回撤类指标，越小越好
                    new_fitness = new_result['backtest_results'].get(objective, new_result['backtest_results'].get('最大回撤', 1.0))
                    delta_fitness = new_fitness - current_fitness
                else:
                    # 对于其他指标，越大越好
                    new_fitness = new_result['backtest_results'].get(objective, new_result['backtest_results'].get('收益回撤比', 0.0))
                    delta_fitness = current_fitness - new_fitness
                
                # 存储评估结果
                all_evaluations.append({
                    'parameters': new_params,
                    'fitness': new_fitness,
                    'backtest_results': new_result['backtest_results']
                })
                
                # Metropolis准则决定是否接受新解
                if (is_minimize and new_fitness < current_fitness) or (not is_minimize and new_fitness > current_fitness):
                    # 新解更好，接受
                    current_solution = deepcopy(new_solution)
                    current_params = deepcopy(new_params)
                    current_fitness = new_fitness
                    current_result = new_result
                    
                    # 更新全局最优解
                    if (is_minimize and new_fitness < best_fitness) or (not is_minimize and new_fitness > best_fitness):
                        best_solution = deepcopy(new_solution)
                        best_params = deepcopy(new_params)
                        best_fitness = new_fitness
                        best_result = deepcopy(new_result)
                        
                        if is_minimize:
                            self.logger.info(f"发现新的最优解，{objective}: {best_fitness:.4f}")
                        else:
                            self.logger.info(f"发现新的最优解，{objective}: {best_fitness:.4f}")
                else:
                    # 新解较差，按概率接受
                    try:
                        acceptance_probability = math.exp(-delta_fitness / temperature)
                    except OverflowError:
                        acceptance_probability = 0
                    
                    if random.random() < acceptance_probability:
                        current_solution = deepcopy(new_solution)
                        current_params = deepcopy(new_params)
                        current_fitness = new_fitness
                        current_result = new_result
                        
                        self.logger.debug(f"接受次优解，接受概率: {acceptance_probability:.4f}")
                
                iteration += 1
            
            # 降低温度
            temperature *= cooling_rate
        
        # 按适应度排序所有评估结果
        if is_minimize:
            # 对于回撤类指标，越小越好
            all_evaluations.sort(key=lambda x: x['fitness'])
        else:
            # 对于其他指标，越大越好
            all_evaluations.sort(key=lambda x: x['fitness'], reverse=True)
        
        # 转换为标准格式并移除重复结果
        unique_results = []
        unique_params = set()
        
        for eval_result in all_evaluations:
            param_tuple = tuple(sorted(eval_result['parameters'].items()))
            if param_tuple not in unique_params:
                unique_params.add(param_tuple)
                unique_results.append({
                    'parameters': eval_result['parameters'],
                    'backtest_results': eval_result['backtest_results']
                })
        
        # 重新按目标指标排序
        if objective.lower() in ['max_drawdown', '最大回撤']:
            unique_results.sort(key=lambda x: x['backtest_results'].get(objective, x['backtest_results'].get('最大回撤', 1.0)))
        else:
            unique_results.sort(key=lambda x: x['backtest_results'].get(objective, x['backtest_results'].get('收益回撤比', 0.0)), reverse=True)
        
        self.logger.info(f"模拟退火优化完成，找到 {len(unique_results)} 个唯一参数组合")
        
        best_params = unique_results[0]['parameters']
        best_value = unique_results[0]['backtest_results'].get(objective, unique_results[0]['backtest_results'].get('收益回撤比', 0.0))
        self.logger.info(f"最佳参数组合: {best_params}, {objective}值: {best_value:.4f}")
        
        return unique_results
    
    def _generate_random_solution(self, parameter_space: Dict[str, List[Any]]) -> List[int]:
        """生成随机解"""
        solution = []
        for param_values in parameter_space.values():
            solution.append(random.randint(0, len(param_values) - 1))
        return solution
    
    def _solution_to_params(self, solution: List[int], parameter_space: Dict[str, List[Any]], param_names: List[str]) -> Dict[str, Any]:
        """将解转换为参数字典"""
        params = {}
        for i, name in enumerate(param_names):
            param_values = parameter_space[name]
            idx = min(solution[i], len(param_values) - 1)
            params[name] = param_values[idx]
        return params
    
    def _generate_neighbor(self, solution: List[int], parameter_space: Dict[str, List[Any]]) -> List[int]:
        """生成邻域解"""
        neighbor = solution.copy()
        
        # 随机选择一个维度
        dim = random.randint(0, len(solution) - 1)
        
        # 获取该维度的参数空间大小
        param_values = list(parameter_space.values())[dim]
        param_size = len(param_values)
        
        if param_size <= 1:
            return neighbor
        
        # 生成新值
        new_value = random.randint(0, param_size - 1)
        while new_value == neighbor[dim] and param_size > 1:
            new_value = random.randint(0, param_size - 1)
        
        neighbor[dim] = new_value
        return neighbor


# 中文命名版本
class 模拟退火优化器(参数优化器):
    """模拟退火优化器，通过模拟退火过程寻找全局最优解（中文版）"""
    
    def 优化(self, 
           策略,
           市场数据: pd.DataFrame,
           庄家数据: pd.DataFrame,
           股票代码: str,
           参数空间: Dict[str, List[Any]],
           目标指标: str = '收益回撤比',
           初始温度: float = 100.0,
           冷却率: float = 0.95,
           迭代次数: int = 100,
           **kwargs) -> List[Dict[str, Any]]:
        """
        使用模拟退火算法优化策略参数。
        
        参数:
            策略: 要优化的策略实例
            市场数据: 市场数据DataFrame
            庄家数据: 庄家指标数据DataFrame
            股票代码: 交易标的代码
            参数空间: 参数名到可能值列表的映射
            目标指标: 优化目标指标，默认为'收益回撤比'
            初始温度: 初始温度，控制初始接受概率，默认为100.0
            冷却率: 冷却率，控制温度下降速度，默认为0.95
            迭代次数: 每个温度下的迭代次数，默认为100
            
        返回:
            按目标指标排序的参数组合和结果列表
        """
        参数名称 = list(参数空间.keys())
        self.日志器.info(f"开始模拟退火优化，初始温度: {初始温度}, 冷却率: {冷却率}")
        
        # 生成初始解
        当前解 = self._生成随机解(参数空间)
        当前参数 = self._解转换为参数(当前解, 参数空间, 参数名称)
        
        # 评估初始解
        任务 = (策略, 市场数据, 庄家数据, 股票代码, 当前参数, 目标指标)
        当前结果 = self._执行单次回测(任务)
        
        # 判断适应度计算方式
        是否最小化 = 目标指标.lower() in ['max_drawdown', '最大回撤']
        
        if 是否最小化:
            # 对于回撤类指标，越小越好
            当前适应度 = 当前结果['回测结果'].get(目标指标, 当前结果['回测结果'].get('最大回撤', 1.0))
        else:
            # 对于其他指标，越大越好
            当前适应度 = 当前结果['回测结果'].get(目标指标, 当前结果['回测结果'].get('收益回撤比', 0.0))
        
        # 设置当前解为最优解
        最优解 = deepcopy(当前解)
        最优参数 = deepcopy(当前参数)
        最优适应度 = 当前适应度
        最优结果 = deepcopy(当前结果)
        
        # 存储所有评估结果
        所有评估 = [{
            '参数': 当前参数,
            '适应度': 当前适应度,
            '回测结果': 当前结果['回测结果']
        }]
        
        # 主循环
        温度 = 初始温度
        迭代计数 = 0
        
        while 温度 > 0.1:
            self.日志器.info(f"当前温度: {温度:.2f}")
            
            for _ in range(迭代次数):
                # 生成新解
                新解 = self._生成邻域解(当前解, 参数空间)
                新参数 = self._解转换为参数(新解, 参数空间, 参数名称)
                
                # 评估新解
                任务 = (策略, 市场数据, 庄家数据, 股票代码, 新参数, 目标指标)
                新结果 = self._执行单次回测(任务)
                
                if 是否最小化:
                    # 对于回撤类指标，越小越好
                    新适应度 = 新结果['回测结果'].get(目标指标, 新结果['回测结果'].get('最大回撤', 1.0))
                    适应度差值 = 新适应度 - 当前适应度
                else:
                    # 对于其他指标，越大越好
                    新适应度 = 新结果['回测结果'].get(目标指标, 新结果['回测结果'].get('收益回撤比', 0.0))
                    适应度差值 = 当前适应度 - 新适应度
                
                # 存储评估结果
                所有评估.append({
                    '参数': 新参数,
                    '适应度': 新适应度,
                    '回测结果': 新结果['回测结果']
                })
                
                # Metropolis准则决定是否接受新解
                if (是否最小化 and 新适应度 < 当前适应度) or (not 是否最小化 and 新适应度 > 当前适应度):
                    # 新解更好，接受
                    当前解 = deepcopy(新解)
                    当前参数 = deepcopy(新参数)
                    当前适应度 = 新适应度
                    当前结果 = 新结果
                    
                    # 更新全局最优解
                    if (是否最小化 and 新适应度 < 最优适应度) or (not 是否最小化 and 新适应度 > 最优适应度):
                        最优解 = deepcopy(新解)
                        最优参数 = deepcopy(新参数)
                        最优适应度 = 新适应度
                        最优结果 = deepcopy(新结果)
                        
                        if 是否最小化:
                            self.日志器.info(f"发现新的最优解，{目标指标}: {最优适应度:.4f}")
                        else:
                            self.日志器.info(f"发现新的最优解，{目标指标}: {最优适应度:.4f}")
                else:
                    # 新解较差，按概率接受
                    try:
                        接受概率 = math.exp(-适应度差值 / 温度)
                    except OverflowError:
                        接受概率 = 0
                    
                    if random.random() < 接受概率:
                        当前解 = deepcopy(新解)
                        当前参数 = deepcopy(新参数)
                        当前适应度 = 新适应度
                        当前结果 = 新结果
                        
                        self.日志器.debug(f"接受次优解，接受概率: {接受概率:.4f}")
                
                迭代计数 += 1
            
            # 降低温度
            温度 *= 冷却率
        
        # 按适应度排序所有评估结果
        if 是否最小化:
            # 对于回撤类指标，越小越好
            所有评估.sort(key=lambda x: x['适应度'])
        else:
            # 对于其他指标，越大越好
            所有评估.sort(key=lambda x: x['适应度'], reverse=True)
        
        # 转换为标准格式并移除重复结果
        唯一结果 = []
        唯一参数集合 = set()
        
        for 评估结果 in 所有评估:
            参数元组 = tuple(sorted(评估结果['参数'].items()))
            if 参数元组 not in 唯一参数集合:
                唯一参数集合.add(参数元组)
                唯一结果.append({
                    '参数': 评估结果['参数'],
                    '回测结果': 评估结果['回测结果']
                })
        
        # 重新按目标指标排序
        if 目标指标.lower() in ['max_drawdown', '最大回撤']:
            唯一结果.sort(key=lambda x: x['回测结果'].get(目标指标, x['回测结果'].get('最大回撤', 1.0)))
        else:
            唯一结果.sort(key=lambda x: x['回测结果'].get(目标指标, x['回测结果'].get('收益回撤比', 0.0)), reverse=True)
        
        self.日志器.info(f"模拟退火优化完成，找到 {len(唯一结果)} 个唯一参数组合")
        
        最佳参数 = 唯一结果[0]['参数']
        最佳值 = 唯一结果[0]['回测结果'].get(目标指标, 唯一结果[0]['回测结果'].get('收益回撤比', 0.0))
        self.日志器.info(f"最佳参数组合: {最佳参数}, {目标指标}值: {最佳值:.4f}")
        
        return 唯一结果
    
    def _生成随机解(self, 参数空间: Dict[str, List[Any]]) -> List[int]:
        """生成随机解"""
        解 = []
        for 参数值列表 in 参数空间.values():
            解.append(random.randint(0, len(参数值列表) - 1))
        return 解
    
    def _解转换为参数(self, 解: List[int], 参数空间: Dict[str, List[Any]], 参数名称: List[str]) -> Dict[str, Any]:
        """将解转换为参数字典"""
        参数 = {}
        for i, 名称 in enumerate(参数名称):
            参数值列表 = 参数空间[名称]
            索引 = min(解[i], len(参数值列表) - 1)
            参数[名称] = 参数值列表[索引]
        return 参数
    
    def _生成邻域解(self, 解: List[int], 参数空间: Dict[str, List[Any]]) -> List[int]:
        """生成邻域解"""
        邻域解 = 解.copy()
        
        # 随机选择一个维度
        维度 = random.randint(0, len(解) - 1)
        
        # 获取该维度的参数空间大小
        参数值列表 = list(参数空间.values())[维度]
        参数空间大小 = len(参数值列表)
        
        if 参数空间大小 <= 1:
            return 邻域解
        
        # 生成新值
        新值 = random.randint(0, 参数空间大小 - 1)
        while 新值 == 邻域解[维度] and 参数空间大小 > 1:
            新值 = random.randint(0, 参数空间大小 - 1)
        
        邻域解[维度] = 新值
        return 邻域解 