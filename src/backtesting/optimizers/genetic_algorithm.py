"""
遗传算法优化器模块

实现基于遗传算法的参数优化方法，通过模拟自然选择过程来寻找最优参数组合。

日期：2025-05-17
"""

import logging
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import multiprocessing
from copy import deepcopy

from .optimizer_base import ParameterOptimizer, 参数优化器

# 配置日志
logger = logging.getLogger(__name__)

class GeneticOptimizer(ParameterOptimizer):
    """遗传算法优化器，通过进化算法搜索参数空间"""
    
    def optimize(self, 
                strategy,
                market_data: pd.DataFrame,
                smart_money_data: pd.DataFrame,
                symbol: str,
                parameter_space: Dict[str, List[Any]],
                objective: str = 'calmar_ratio',
                population_size: int = 20,
                generations: int = 10,
                mutation_rate: float = 0.1,
                crossover_rate: float = 0.8,
                elitism: int = 2,
                parallel: bool = True,
                **kwargs) -> List[Dict[str, Any]]:
        """
        使用遗传算法优化策略参数。
        
        参数:
            strategy: 要优化的策略实例
            market_data: 市场数据DataFrame
            smart_money_data: 庄家指标数据DataFrame
            symbol: 交易标的代码
            parameter_space: 参数名到可能值列表的映射
            objective: 优化目标指标，默认为'calmar_ratio'
            population_size: 种群大小，默认为20
            generations: 迭代代数，默认为10
            mutation_rate: 变异率，默认为0.1
            crossover_rate: 交叉率，默认为0.8
            elitism: 精英数量（每代保留的最佳个体数），默认为2
            parallel: 是否使用并行计算
            
        返回:
            按目标指标排序的参数组合和结果列表
        """
        param_names = list(parameter_space.keys())
        self.logger.info(f"开始遗传算法优化，种群大小: {population_size}, 代数: {generations}")
        
        # 创建初始种群
        population = self._create_initial_population(parameter_space, population_size)
        
        # 存储所有评估过的个体及其适应度
        all_individuals = []
        
        # 主循环
        for generation in range(generations):
            self.logger.info(f"开始第 {generation + 1}/{generations} 代进化")
            
            # 评估适应度
            tasks = []
            for individual in population:
                params = {param_names[i]: individual[i] for i in range(len(param_names))}
                tasks.append((strategy, market_data, smart_money_data, symbol, params, objective))
            
            # 并行或串行执行评估
            if parallel and len(tasks) > 1:
                with multiprocessing.Pool(self.parallel_processes) as pool:
                    fitness_results = pool.map(self._execute_single_backtest, tasks)
            else:
                fitness_results = [self._execute_single_backtest(task) for task in tasks]
            
            # 保存结果
            for individual, result in zip(population, fitness_results):
                params = {param_names[i]: individual[i] for i in range(len(param_names))}
                
                # 获取适应度值（目标指标）
                if objective.lower() in ['max_drawdown', '最大回撤']:
                    # 对于回撤类指标，越小越好，取相反数使之成为最大化问题
                    fitness = -result['backtest_results'].get(objective, result['backtest_results'].get('最大回撤', 1.0))
                else:
                    # 对于其他指标，越大越好
                    fitness = result['backtest_results'].get(objective, result['backtest_results'].get('收益回撤比', 0.0))
                
                all_individuals.append({
                    'individual': individual.copy(),
                    'parameters': params,
                    'fitness': fitness,
                    'backtest_results': result['backtest_results']
                })
            
            # 选择精英
            population_with_fitness = [{
                'individual': individual,
                'fitness': self._get_fitness(result['backtest_results'], objective)
            } for individual, result in zip(population, fitness_results)]
            
            population_with_fitness.sort(key=lambda x: x['fitness'], reverse=True)
            
            elite = [individual['individual'] for individual in population_with_fitness[:elitism]]
            
            # 如果已经达到最后一代，跳出循环
            if generation == generations - 1:
                break
                
            # 生成新种群
            new_population = elite.copy()  # 保留精英
            
            # 通过选择、交叉和变异生成剩余个体
            while len(new_population) < population_size:
                # 选择
                if random.random() < crossover_rate:
                    # 交叉
                    parent1 = self._selection(population_with_fitness)
                    parent2 = self._selection(population_with_fitness)
                    child1, child2 = self._crossover(parent1, parent2)
                    
                    # 变异
                    if random.random() < mutation_rate:
                        child1 = self._mutation(child1, parameter_space)
                    if random.random() < mutation_rate:
                        child2 = self._mutation(child2, parameter_space)
                    
                    new_population.append(child1)
                    if len(new_population) < population_size:
                        new_population.append(child2)
                else:
                    # 直接选择一个个体
                    selected = self._selection(population_with_fitness)
                    
                    # 变异
                    if random.random() < mutation_rate:
                        selected = self._mutation(selected, parameter_space)
                    
                    new_population.append(selected)
            
            # 更新种群
            population = new_population
        
        # 最终结果排序
        all_individuals.sort(key=lambda x: x['fitness'], reverse=True)
        
        # 转换为标准格式
        results = [{
            'parameters': individual['parameters'],
            'backtest_results': individual['backtest_results']
        } for individual in all_individuals]
        
        # 去除重复结果
        unique_results = []
        unique_params = set()
        
        for result in results:
            param_tuple = tuple(sorted(result['parameters'].items()))
            if param_tuple not in unique_params:
                unique_params.add(param_tuple)
                unique_results.append(result)
        
        self.logger.info(f"遗传算法优化完成，找到 {len(unique_results)} 个唯一参数组合")
        
        # 按目标指标排序
        if objective.lower() in ['max_drawdown', '最大回撤']:
            unique_results.sort(key=lambda x: x['backtest_results'].get(objective, x['backtest_results'].get('最大回撤', 1.0)), reverse=False)
        else:
            unique_results.sort(key=lambda x: x['backtest_results'].get(objective, x['backtest_results'].get('收益回撤比', 0.0)), reverse=True)
        
        best_params = unique_results[0]['parameters']
        best_value = unique_results[0]['backtest_results'].get(objective, unique_results[0]['backtest_results'].get('收益回撤比', 0.0))
        self.logger.info(f"最佳参数组合: {best_params}, {objective}值: {best_value:.4f}")
        
        return unique_results
    
    def _create_initial_population(self, parameter_space: Dict[str, List[Any]], population_size: int) -> List[List[Any]]:
        """创建初始种群"""
        param_values = list(parameter_space.values())
        population = []
        
        # 随机生成种群
        for _ in range(population_size):
            individual = [random.choice(values) for values in param_values]
            population.append(individual)
            
        return population
    
    def _get_fitness(self, backtest_results: Dict[str, Any], objective: str) -> float:
        """获取适应度值"""
        if objective.lower() in ['max_drawdown', '最大回撤']:
            # 对于回撤类指标，越小越好，取相反数使之成为最大化问题
            return -backtest_results.get(objective, backtest_results.get('最大回撤', 1.0))
        else:
            # 对于其他指标，越大越好
            return backtest_results.get(objective, backtest_results.get('收益回撤比', 0.0))
    
    def _selection(self, population_with_fitness: List[Dict]) -> List[Any]:
        """
        选择操作，使用轮盘赌方法选择个体
        """
        # 计算总适应度
        total_fitness = sum(max(0.0001, individual['fitness']) for individual in population_with_fitness)
        
        # 如果总适应度为0，则随机选择
        if total_fitness <= 0:
            return random.choice(population_with_fitness)['individual']
        
        # 轮盘赌选择
        pick = random.uniform(0, total_fitness)
        current = 0
        for individual in population_with_fitness:
            current += max(0.0001, individual['fitness'])
            if current >= pick:
                return individual['individual']
        
        # 防止浮点误差导致未选中任何个体
        return population_with_fitness[-1]['individual']
    
    def _crossover(self, parent1: List[Any], parent2: List[Any]) -> Tuple[List[Any], List[Any]]:
        """
        交叉操作，使用单点交叉
        """
        if len(parent1) <= 1:
            return parent1.copy(), parent2.copy()
        
        # 随机选择交叉点
        crossover_point = random.randint(1, len(parent1) - 1)
        
        # 交叉
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def _mutation(self, individual: List[Any], parameter_space: Dict[str, List[Any]]) -> List[Any]:
        """
        变异操作
        """
        # 复制个体
        mutated = individual.copy()
        
        # 随机选择一个位置进行变异
        mutation_point = random.randint(0, len(individual) - 1)
        
        # 获取对应参数的可能值
        param_values = list(parameter_space.values())[mutation_point]
        
        # 选择一个不同的值
        current_value = mutated[mutation_point]
        possible_values = [v for v in param_values if v != current_value]
        
        if possible_values:
            mutated[mutation_point] = random.choice(possible_values)
        
        return mutated


# 中文命名版本
class 遗传算法优化器(参数优化器):
    """遗传算法优化器，通过进化算法搜索参数空间（中文版）"""
    
    def 优化(self, 
           策略,
           市场数据: pd.DataFrame,
           庄家数据: pd.DataFrame,
           股票代码: str,
           参数空间: Dict[str, List[Any]],
           目标指标: str = '收益回撤比',
           种群大小: int = 20,
           代数: int = 10,
           变异率: float = 0.1,
           交叉率: float = 0.8,
           精英数量: int = 2,
           并行: bool = True,
           **kwargs) -> List[Dict[str, Any]]:
        """
        使用遗传算法优化策略参数。
        
        参数:
            策略: 要优化的策略实例
            市场数据: 市场数据DataFrame
            庄家数据: 庄家指标数据DataFrame
            股票代码: 交易标的代码
            参数空间: 参数名到可能值列表的映射
            目标指标: 优化目标指标，默认为'收益回撤比'
            种群大小: 种群大小，默认为20
            代数: 迭代代数，默认为10
            变异率: 变异率，默认为0.1
            交叉率: 交叉率，默认为0.8
            精英数量: 精英数量（每代保留的最佳个体数），默认为2
            并行: 是否使用并行计算
            
        返回:
            按目标指标排序的参数组合和结果列表
        """
        参数名称 = list(参数空间.keys())
        self.日志器.info(f"开始遗传算法优化，种群大小: {种群大小}, 代数: {代数}")
        
        # 创建初始种群
        种群 = self._创建初始种群(参数空间, 种群大小)
        
        # 存储所有评估过的个体及其适应度
        所有个体 = []
        
        # 主循环
        for 当前代数 in range(代数):
            self.日志器.info(f"开始第 {当前代数 + 1}/{代数} 代进化")
            
            # 评估适应度
            任务列表 = []
            for 个体 in 种群:
                参数 = {参数名称[i]: 个体[i] for i in range(len(参数名称))}
                任务列表.append((策略, 市场数据, 庄家数据, 股票代码, 参数, 目标指标))
            
            # 并行或串行执行评估
            if 并行 and len(任务列表) > 1:
                with multiprocessing.Pool(self.并行进程数) as pool:
                    适应度结果 = pool.map(self._执行单次回测, 任务列表)
            else:
                适应度结果 = [self._执行单次回测(任务) for 任务 in 任务列表]
            
            # 保存结果
            for 个体, 结果 in zip(种群, 适应度结果):
                参数 = {参数名称[i]: 个体[i] for i in range(len(参数名称))}
                
                # 获取适应度值（目标指标）
                if 目标指标.lower() in ['max_drawdown', '最大回撤']:
                    # 对于回撤类指标，越小越好，取相反数使之成为最大化问题
                    适应度 = -结果['回测结果'].get(目标指标, 结果['回测结果'].get('最大回撤', 1.0))
                else:
                    # 对于其他指标，越大越好
                    适应度 = 结果['回测结果'].get(目标指标, 结果['回测结果'].get('收益回撤比', 0.0))
                
                所有个体.append({
                    '个体': 个体.copy(),
                    '参数': 参数,
                    '适应度': 适应度,
                    '回测结果': 结果['回测结果']
                })
            
            # 选择精英
            带适应度种群 = [{
                '个体': 个体,
                '适应度': self._获取适应度(结果['回测结果'], 目标指标)
            } for 个体, 结果 in zip(种群, 适应度结果)]
            
            带适应度种群.sort(key=lambda x: x['适应度'], reverse=True)
            
            精英个体 = [个体信息['个体'] for 个体信息 in 带适应度种群[:精英数量]]
            
            # 如果已经达到最后一代，跳出循环
            if 当前代数 == 代数 - 1:
                break
                
            # 生成新种群
            新种群 = 精英个体.copy()  # 保留精英
            
            # 通过选择、交叉和变异生成剩余个体
            while len(新种群) < 种群大小:
                # 选择
                if random.random() < 交叉率:
                    # 交叉
                    父本 = self._选择(带适应度种群)
                    母本 = self._选择(带适应度种群)
                    子代1, 子代2 = self._交叉(父本, 母本)
                    
                    # 变异
                    if random.random() < 变异率:
                        子代1 = self._变异(子代1, 参数空间)
                    if random.random() < 变异率:
                        子代2 = self._变异(子代2, 参数空间)
                    
                    新种群.append(子代1)
                    if len(新种群) < 种群大小:
                        新种群.append(子代2)
                else:
                    # 直接选择一个个体
                    选中个体 = self._选择(带适应度种群)
                    
                    # 变异
                    if random.random() < 变异率:
                        选中个体 = self._变异(选中个体, 参数空间)
                    
                    新种群.append(选中个体)
            
            # 更新种群
            种群 = 新种群
        
        # 最终结果排序
        所有个体.sort(key=lambda x: x['适应度'], reverse=True)
        
        # 转换为标准格式
        结果列表 = [{
            '参数': 个体['参数'],
            '回测结果': 个体['回测结果']
        } for 个体 in 所有个体]
        
        # 去除重复结果
        唯一结果 = []
        唯一参数集合 = set()
        
        for 结果 in 结果列表:
            参数元组 = tuple(sorted(结果['参数'].items()))
            if 参数元组 not in 唯一参数集合:
                唯一参数集合.add(参数元组)
                唯一结果.append(结果)
        
        self.日志器.info(f"遗传算法优化完成，找到 {len(唯一结果)} 个唯一参数组合")
        
        # 按目标指标排序
        if 目标指标.lower() in ['max_drawdown', '最大回撤']:
            唯一结果.sort(key=lambda x: x['回测结果'].get(目标指标, x['回测结果'].get('最大回撤', 1.0)), reverse=False)
        else:
            唯一结果.sort(key=lambda x: x['回测结果'].get(目标指标, x['回测结果'].get('收益回撤比', 0.0)), reverse=True)
        
        最佳参数 = 唯一结果[0]['参数']
        最佳值 = 唯一结果[0]['回测结果'].get(目标指标, 唯一结果[0]['回测结果'].get('收益回撤比', 0.0))
        self.日志器.info(f"最佳参数组合: {最佳参数}, {目标指标}值: {最佳值:.4f}")
        
        return 唯一结果
    
    def _创建初始种群(self, 参数空间: Dict[str, List[Any]], 种群大小: int) -> List[List[Any]]:
        """创建初始种群"""
        参数值列表 = list(参数空间.values())
        种群 = []
        
        # 随机生成种群
        for _ in range(种群大小):
            个体 = [random.choice(值列表) for 值列表 in 参数值列表]
            种群.append(个体)
            
        return 种群
    
    def _获取适应度(self, 回测结果: Dict[str, Any], 目标指标: str) -> float:
        """获取适应度值"""
        if 目标指标.lower() in ['max_drawdown', '最大回撤']:
            # 对于回撤类指标，越小越好，取相反数使之成为最大化问题
            return -回测结果.get(目标指标, 回测结果.get('最大回撤', 1.0))
        else:
            # 对于其他指标，越大越好
            return 回测结果.get(目标指标, 回测结果.get('收益回撤比', 0.0))
    
    def _选择(self, 带适应度种群: List[Dict]) -> List[Any]:
        """
        选择操作，使用轮盘赌方法选择个体
        """
        # 计算总适应度
        总适应度 = sum(max(0.0001, 个体信息['适应度']) for 个体信息 in 带适应度种群)
        
        # 如果总适应度为0，则随机选择
        if 总适应度 <= 0:
            return random.choice(带适应度种群)['个体']
        
        # 轮盘赌选择
        选择值 = random.uniform(0, 总适应度)
        当前值 = 0
        for 个体信息 in 带适应度种群:
            当前值 += max(0.0001, 个体信息['适应度'])
            if 当前值 >= 选择值:
                return 个体信息['个体']
        
        # 防止浮点误差导致未选中任何个体
        return 带适应度种群[-1]['个体']
    
    def _交叉(self, 父本: List[Any], 母本: List[Any]) -> Tuple[List[Any], List[Any]]:
        """
        交叉操作，使用单点交叉
        """
        if len(父本) <= 1:
            return 父本.copy(), 母本.copy()
        
        # 随机选择交叉点
        交叉点 = random.randint(1, len(父本) - 1)
        
        # 交叉
        子代1 = 父本[:交叉点] + 母本[交叉点:]
        子代2 = 母本[:交叉点] + 父本[交叉点:]
        
        return 子代1, 子代2
    
    def _变异(self, 个体: List[Any], 参数空间: Dict[str, List[Any]]) -> List[Any]:
        """
        变异操作
        """
        # 复制个体
        变异后个体 = 个体.copy()
        
        # 随机选择一个位置进行变异
        变异点 = random.randint(0, len(个体) - 1)
        
        # 获取对应参数的可能值
        参数值列表 = list(参数空间.values())[变异点]
        
        # 选择一个不同的值
        当前值 = 变异后个体[变异点]
        可能值 = [值 for 值 in 参数值列表 if 值 != 当前值]
        
        if 可能值:
            变异后个体[变异点] = random.choice(可能值)
        
        return 变异后个体 