"""
多目标优化器模块

实现基于帕累托前沿的多目标优化算法，同时优化多个指标。

日期：2025-05-17
"""

import logging
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import multiprocessing
from copy import deepcopy

from .optimizer_base import ParameterOptimizer, 参数优化器

# 配置日志
logger = logging.getLogger(__name__)

class MultiObjectiveOptimizer(ParameterOptimizer):
    """多目标优化器，使用帕累托优化方法同时优化多个目标"""
    
    def optimize(self, 
                strategy,
                market_data: pd.DataFrame,
                smart_money_data: pd.DataFrame,
                symbol: str,
                parameter_space: Dict[str, List[Any]],
                objectives: List[str] = ['calmar_ratio', 'win_rate'],
                num_solutions: int = 50,
                generations: int = 10,
                mutation_rate: float = 0.1,
                crossover_rate: float = 0.8,
                parallel: bool = True,
                **kwargs) -> List[Dict[str, Any]]:
        """
        使用多目标优化算法同时优化多个指标。
        
        参数:
            strategy: 要优化的策略实例
            market_data: 市场数据DataFrame
            smart_money_data: 庄家指标数据DataFrame
            symbol: 交易标的代码
            parameter_space: 参数名到可能值列表的映射
            objectives: 优化目标指标列表，默认为['calmar_ratio', 'win_rate']
            num_solutions: 解的数量，默认为50
            generations: 迭代代数，默认为10
            mutation_rate: 变异率，默认为0.1
            crossover_rate: 交叉率，默认为0.8
            parallel: 是否使用并行计算
            
        返回:
            帕累托最优解集合
        """
        if len(objectives) < 2:
            self.logger.warning("多目标优化需要至少两个目标。使用calmar_ratio和win_rate作为默认目标。")
            objectives = ['calmar_ratio', 'win_rate']
        
        # 确定每个目标是最大化还是最小化问题
        is_minimize = [obj.lower() in ['max_drawdown', '最大回撤'] for obj in objectives]
        
        param_names = list(parameter_space.keys())
        self.logger.info(f"开始多目标优化，目标: {objectives}, 种群大小: {num_solutions}, 代数: {generations}")
        
        # 创建初始种群
        population = self._create_initial_population(parameter_space, num_solutions)
        
        # 存储所有解和评估结果
        all_solutions = []
        
        # 主循环
        for generation in range(generations):
            self.logger.info(f"开始第 {generation + 1}/{generations} 代进化")
            
            # 评估当前种群
            tasks = []
            for individual in population:
                params = self._solution_to_params(individual, parameter_space, param_names)
                tasks.append((strategy, market_data, smart_money_data, symbol, params, None))
            
            # 并行或串行执行评估
            if parallel and len(tasks) > 1:
                with multiprocessing.Pool(self.parallel_processes) as pool:
                    results = pool.map(self._execute_single_backtest, tasks)
            else:
                results = [self._execute_single_backtest(task) for task in tasks]
            
            # 收集评估结果
            evaluated_population = []
            for individual, result in zip(population, results):
                params = self._solution_to_params(individual, parameter_space, param_names)
                
                # 提取每个目标的值
                objective_values = []
                for i, obj in enumerate(objectives):
                    value = result['backtest_results'].get(obj, 0.0)
                    # 对于最小化问题，取负值使其成为最大化问题
                    if is_minimize[i]:
                        value = -value
                    objective_values.append(value)
                
                evaluated_population.append({
                    'solution': individual,
                    'parameters': params,
                    'objective_values': objective_values,
                    'backtest_results': result['backtest_results']
                })
                
                # 添加到所有解集合
                all_solutions.append({
                    'solution': individual.copy(),
                    'parameters': params.copy(),
                    'objective_values': objective_values.copy(),
                    'backtest_results': result['backtest_results'].copy()
                })
            
            # 找出当前种群的帕累托前沿
            pareto_front = self._find_pareto_front(evaluated_population)
            
            self.logger.info(f"第 {generation + 1} 代帕累托前沿大小: {len(pareto_front)}")
            
            # 如果已经到最后一代，跳出循环
            if generation == generations - 1:
                break
            
            # 生成新一代种群
            new_population = []
            
            # 保留帕累托前沿的解
            for solution in pareto_front:
                new_population.append(solution['solution'])
            
            # 通过选择、交叉和变异生成其余解
            while len(new_population) < num_solutions:
                if random.random() < crossover_rate:
                    # 选择两个父代
                    parent1 = self._tournament_selection(evaluated_population)
                    parent2 = self._tournament_selection(evaluated_population)
                    
                    # 交叉生成子代
                    child1, child2 = self._crossover(parent1['solution'], parent2['solution'])
                    
                    # 变异
                    if random.random() < mutation_rate:
                        child1 = self._mutation(child1, parameter_space)
                    if random.random() < mutation_rate:
                        child2 = self._mutation(child2, parameter_space)
                    
                    new_population.append(child1)
                    if len(new_population) < num_solutions:
                        new_population.append(child2)
                else:
                    # 直接选择一个个体
                    parent = self._tournament_selection(evaluated_population)
                    
                    # 变异
                    if random.random() < mutation_rate:
                        new_individual = self._mutation(parent['solution'].copy(), parameter_space)
                    else:
                        new_individual = parent['solution'].copy()
                    
                    new_population.append(new_individual)
            
            # 更新种群
            population = new_population
        
        # 找出所有解中的帕累托前沿
        final_pareto_front = self._find_pareto_front(all_solutions)
        
        # 转换为标准格式
        results = []
        for solution in final_pareto_front:
            # 将目标值转换回原始方向
            original_objective_values = {}
            for i, obj in enumerate(objectives):
                value = solution['objective_values'][i]
                if is_minimize[i]:
                    value = -value
                original_objective_values[obj] = value
            
            results.append({
                'parameters': solution['parameters'],
                'backtest_results': solution['backtest_results'],
                'objective_values': original_objective_values
            })
        
        self.logger.info(f"多目标优化完成，帕累托前沿包含 {len(results)} 个解")
        
        # 打印前沿上的一些代表性解
        if results:
            for i, result in enumerate(results[:min(3, len(results))]):
                obj_values_str = ", ".join([f"{obj}: {result['objective_values'][obj]:.4f}" for obj in objectives])
                self.logger.info(f"帕累托解 #{i+1}: {obj_values_str}")
        
        return results
    
    def _create_initial_population(self, parameter_space: Dict[str, List[Any]], num_solutions: int) -> List[List[int]]:
        """创建初始种群"""
        population = []
        for _ in range(num_solutions):
            individual = []
            for param_values in parameter_space.values():
                individual.append(random.randint(0, len(param_values) - 1))
            population.append(individual)
        return population
    
    def _solution_to_params(self, solution: List[int], parameter_space: Dict[str, List[Any]], param_names: List[str]) -> Dict[str, Any]:
        """将解转换为参数字典"""
        params = {}
        for i, name in enumerate(param_names):
            param_values = parameter_space[name]
            idx = min(solution[i], len(param_values) - 1)
            params[name] = param_values[idx]
        return params
    
    def _dominates(self, solution1: Dict, solution2: Dict) -> bool:
        """判断解1是否支配解2"""
        obj_values1 = solution1['objective_values']
        obj_values2 = solution2['objective_values']
        
        # 至少在一个目标上严格优于解2，且在其他目标上不劣于解2
        at_least_one_better = False
        for i in range(len(obj_values1)):
            if obj_values1[i] < obj_values2[i]:
                return False  # 解1在某个目标上劣于解2
            elif obj_values1[i] > obj_values2[i]:
                at_least_one_better = True  # 解1在至少一个目标上优于解2
        
        return at_least_one_better
    
    def _find_pareto_front(self, solutions: List[Dict]) -> List[Dict]:
        """找出帕累托前沿"""
        pareto_front = []
        
        for i, solution in enumerate(solutions):
            is_dominated = False
            
            # 检查是否被其他解支配
            for j, other_solution in enumerate(solutions):
                if i != j and self._dominates(other_solution, solution):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(solution)
        
        return pareto_front
    
    def _tournament_selection(self, evaluated_population: List[Dict], tournament_size: int = 3) -> Dict:
        """锦标赛选择"""
        if not evaluated_population:
            return None
        
        # 随机选择tournament_size个个体
        candidates = random.sample(evaluated_population, min(tournament_size, len(evaluated_population)))
        
        # 找出非支配解
        non_dominated = self._find_pareto_front(candidates)
        
        if non_dominated:
            # 从非支配解中随机选择一个
            return random.choice(non_dominated)
        else:
            # 如果没有非支配解，随机选择一个候选个体
            return random.choice(candidates)
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """交叉操作"""
        if len(parent1) <= 1:
            return parent1.copy(), parent2.copy()
        
        # 单点交叉
        crossover_point = random.randint(1, len(parent1) - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def _mutation(self, individual: List[int], parameter_space: Dict[str, List[Any]]) -> List[int]:
        """变异操作"""
        mutated = individual.copy()
        
        # 随机选择一个位置进行变异
        mutation_point = random.randint(0, len(individual) - 1)
        
        # 获取对应参数的可能值
        param_values = list(parameter_space.values())[mutation_point]
        
        # 选择一个不同的值
        current_value = mutated[mutation_point]
        
        if len(param_values) > 1:
            new_value = random.randint(0, len(param_values) - 1)
            while new_value == current_value and len(param_values) > 1:
                new_value = random.randint(0, len(param_values) - 1)
            
            mutated[mutation_point] = new_value
        
        return mutated


# 中文命名版本
class 多目标优化器(参数优化器):
    """多目标优化器，使用帕累托优化方法同时优化多个目标（中文版）"""
    
    def 优化(self, 
           策略,
           市场数据: pd.DataFrame,
           庄家数据: pd.DataFrame,
           股票代码: str,
           参数空间: Dict[str, List[Any]],
           目标指标列表: List[str] = ['收益回撤比', '胜率'],
           解数量: int = 50,
           代数: int = 10,
           变异率: float = 0.1,
           交叉率: float = 0.8,
           并行: bool = True,
           **kwargs) -> List[Dict[str, Any]]:
        """
        使用多目标优化算法同时优化多个指标。
        
        参数:
            策略: 要优化的策略实例
            市场数据: 市场数据DataFrame
            庄家数据: 庄家指标数据DataFrame
            股票代码: 交易标的代码
            参数空间: 参数名到可能值列表的映射
            目标指标列表: 优化目标指标列表，默认为['收益回撤比', '胜率']
            解数量: 解的数量，默认为50
            代数: 迭代代数，默认为10
            变异率: 变异率，默认为0.1
            交叉率: 交叉率，默认为0.8
            并行: 是否使用并行计算
            
        返回:
            帕累托最优解集合
        """
        if len(目标指标列表) < 2:
            self.日志器.warning("多目标优化需要至少两个目标。使用收益回撤比和胜率作为默认目标。")
            目标指标列表 = ['收益回撤比', '胜率']
        
        # 确定每个目标是最大化还是最小化问题
        是否最小化 = [目标.lower() in ['max_drawdown', '最大回撤'] for 目标 in 目标指标列表]
        
        参数名称 = list(参数空间.keys())
        self.日志器.info(f"开始多目标优化，目标: {目标指标列表}, 种群大小: {解数量}, 代数: {代数}")
        
        # 创建初始种群
        种群 = self._创建初始种群(参数空间, 解数量)
        
        # 存储所有解和评估结果
        所有解 = []
        
        # 主循环
        for 当前代数 in range(代数):
            self.日志器.info(f"开始第 {当前代数 + 1}/{代数} 代进化")
            
            # 评估当前种群
            任务列表 = []
            for 个体 in 种群:
                参数 = self._解转换为参数(个体, 参数空间, 参数名称)
                任务列表.append((策略, 市场数据, 庄家数据, 股票代码, 参数, None))
            
            # 并行或串行执行评估
            if 并行 and len(任务列表) > 1:
                with multiprocessing.Pool(self.并行进程数) as pool:
                    结果列表 = pool.map(self._执行单次回测, 任务列表)
            else:
                结果列表 = [self._执行单次回测(任务) for 任务 in 任务列表]
            
            # 收集评估结果
            已评估种群 = []
            for 个体, 结果 in zip(种群, 结果列表):
                参数 = self._解转换为参数(个体, 参数空间, 参数名称)
                
                # 提取每个目标的值
                目标值列表 = []
                for i, 目标 in enumerate(目标指标列表):
                    值 = 结果['回测结果'].get(目标, 0.0)
                    # 对于最小化问题，取负值使其成为最大化问题
                    if 是否最小化[i]:
                        值 = -值
                    目标值列表.append(值)
                
                已评估种群.append({
                    '解': 个体,
                    '参数': 参数,
                    '目标值列表': 目标值列表,
                    '回测结果': 结果['回测结果']
                })
                
                # 添加到所有解集合
                所有解.append({
                    '解': 个体.copy(),
                    '参数': 参数.copy(),
                    '目标值列表': 目标值列表.copy(),
                    '回测结果': 结果['回测结果'].copy()
                })
            
            # 找出当前种群的帕累托前沿
            帕累托前沿 = self._找出帕累托前沿(已评估种群)
            
            self.日志器.info(f"第 {当前代数 + 1} 代帕累托前沿大小: {len(帕累托前沿)}")
            
            # 如果已经到最后一代，跳出循环
            if 当前代数 == 代数 - 1:
                break
            
            # 生成新一代种群
            新种群 = []
            
            # 保留帕累托前沿的解
            for 解 in 帕累托前沿:
                新种群.append(解['解'])
            
            # 通过选择、交叉和变异生成其余解
            while len(新种群) < 解数量:
                if random.random() < 交叉率:
                    # 选择两个父代
                    父代1 = self._锦标赛选择(已评估种群)
                    父代2 = self._锦标赛选择(已评估种群)
                    
                    # 交叉生成子代
                    子代1, 子代2 = self._交叉(父代1['解'], 父代2['解'])
                    
                    # 变异
                    if random.random() < 变异率:
                        子代1 = self._变异(子代1, 参数空间)
                    if random.random() < 变异率:
                        子代2 = self._变异(子代2, 参数空间)
                    
                    新种群.append(子代1)
                    if len(新种群) < 解数量:
                        新种群.append(子代2)
                else:
                    # 直接选择一个个体
                    父代 = self._锦标赛选择(已评估种群)
                    
                    # 变异
                    if random.random() < 变异率:
                        新个体 = self._变异(父代['解'].copy(), 参数空间)
                    else:
                        新个体 = 父代['解'].copy()
                    
                    新种群.append(新个体)
            
            # 更新种群
            种群 = 新种群
        
        # 找出所有解中的帕累托前沿
        最终帕累托前沿 = self._找出帕累托前沿(所有解)
        
        # 转换为标准格式
        结果列表 = []
        for 解 in 最终帕累托前沿:
            # 将目标值转换回原始方向
            原始目标值 = {}
            for i, 目标 in enumerate(目标指标列表):
                值 = 解['目标值列表'][i]
                if 是否最小化[i]:
                    值 = -值
                原始目标值[目标] = 值
            
            结果列表.append({
                '参数': 解['参数'],
                '回测结果': 解['回测结果'],
                '目标值': 原始目标值
            })
        
        self.日志器.info(f"多目标优化完成，帕累托前沿包含 {len(结果列表)} 个解")
        
        # 打印前沿上的一些代表性解
        if 结果列表:
            for i, 结果 in enumerate(结果列表[:min(3, len(结果列表))]):
                目标值字符串 = ", ".join([f"{目标}: {结果['目标值'][目标]:.4f}" for 目标 in 目标指标列表])
                self.日志器.info(f"帕累托解 #{i+1}: {目标值字符串}")
        
        return 结果列表
    
    def _创建初始种群(self, 参数空间: Dict[str, List[Any]], 解数量: int) -> List[List[int]]:
        """创建初始种群"""
        种群 = []
        for _ in range(解数量):
            个体 = []
            for 参数值列表 in 参数空间.values():
                个体.append(random.randint(0, len(参数值列表) - 1))
            种群.append(个体)
        return 种群
    
    def _解转换为参数(self, 解: List[int], 参数空间: Dict[str, List[Any]], 参数名称: List[str]) -> Dict[str, Any]:
        """将解转换为参数字典"""
        参数 = {}
        for i, 名称 in enumerate(参数名称):
            参数值列表 = 参数空间[名称]
            索引 = min(解[i], len(参数值列表) - 1)
            参数[名称] = 参数值列表[索引]
        return 参数
    
    def _支配(self, 解1: Dict, 解2: Dict) -> bool:
        """判断解1是否支配解2"""
        目标值1 = 解1['目标值列表']
        目标值2 = 解2['目标值列表']
        
        # 至少在一个目标上严格优于解2，且在其他目标上不劣于解2
        至少一个更好 = False
        for i in range(len(目标值1)):
            if 目标值1[i] < 目标值2[i]:
                return False  # 解1在某个目标上劣于解2
            elif 目标值1[i] > 目标值2[i]:
                至少一个更好 = True  # 解1在至少一个目标上优于解2
        
        return 至少一个更好
    
    def _找出帕累托前沿(self, 解集合: List[Dict]) -> List[Dict]:
        """找出帕累托前沿"""
        帕累托前沿 = []
        
        for i, 解 in enumerate(解集合):
            被支配 = False
            
            # 检查是否被其他解支配
            for j, 其他解 in enumerate(解集合):
                if i != j and self._支配(其他解, 解):
                    被支配 = True
                    break
            
            if not 被支配:
                帕累托前沿.append(解)
        
        return 帕累托前沿
    
    def _锦标赛选择(self, 已评估种群: List[Dict], 锦标赛大小: int = 3) -> Dict:
        """锦标赛选择"""
        if not 已评估种群:
            return None
        
        # 随机选择锦标赛大小个个体
        候选个体 = random.sample(已评估种群, min(锦标赛大小, len(已评估种群)))
        
        # 找出非支配解
        非支配解 = self._找出帕累托前沿(候选个体)
        
        if 非支配解:
            # 从非支配解中随机选择一个
            return random.choice(非支配解)
        else:
            # 如果没有非支配解，随机选择一个候选个体
            return random.choice(候选个体)
    
    def _交叉(self, 父代1: List[int], 父代2: List[int]) -> Tuple[List[int], List[int]]:
        """交叉操作"""
        if len(父代1) <= 1:
            return 父代1.copy(), 父代2.copy()
        
        # 单点交叉
        交叉点 = random.randint(1, len(父代1) - 1)
        
        子代1 = 父代1[:交叉点] + 父代2[交叉点:]
        子代2 = 父代2[:交叉点] + 父代1[交叉点:]
        
        return 子代1, 子代2
    
    def _变异(self, 个体: List[int], 参数空间: Dict[str, List[Any]]) -> List[int]:
        """变异操作"""
        变异后个体 = 个体.copy()
        
        # 随机选择一个位置进行变异
        变异点 = random.randint(0, len(个体) - 1)
        
        # 获取对应参数的可能值
        参数值列表 = list(参数空间.values())[变异点]
        
        # 选择一个不同的值
        当前值 = 变异后个体[变异点]
        
        if len(参数值列表) > 1:
            新值 = random.randint(0, len(参数值列表) - 1)
            while 新值 == 当前值 and len(参数值列表) > 1:
                新值 = random.randint(0, len(参数值列表) - 1)
            
            变异后个体[变异点] = 新值
        
        return 变异后个体 