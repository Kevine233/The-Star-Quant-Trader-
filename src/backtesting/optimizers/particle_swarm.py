"""
粒子群优化器模块

实现基于粒子群算法的参数优化方法，通过模拟群体智能寻找最优参数组合。

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

class ParticleSwarmOptimizer(ParameterOptimizer):
    """粒子群优化器，通过模拟群体协作行为搜索参数空间"""
    
    def optimize(self, 
                strategy,
                market_data: pd.DataFrame,
                smart_money_data: pd.DataFrame,
                symbol: str,
                parameter_space: Dict[str, List[Any]],
                objective: str = 'calmar_ratio',
                num_particles: int = 20,
                iterations: int = 10,
                inertia_weight: float = 0.7,
                cognitive_weight: float = 1.5,
                social_weight: float = 1.5,
                parallel: bool = True,
                **kwargs) -> List[Dict[str, Any]]:
        """
        使用粒子群算法优化策略参数。
        
        参数:
            strategy: 要优化的策略实例
            market_data: 市场数据DataFrame
            smart_money_data: 庄家指标数据DataFrame
            symbol: 交易标的代码
            parameter_space: 参数名到可能值列表的映射
            objective: 优化目标指标，默认为'calmar_ratio'
            num_particles: 粒子数量，默认为20
            iterations: 迭代次数，默认为10
            inertia_weight: 惯性权重，控制粒子保持当前速度的程度，默认为0.7
            cognitive_weight: 个体认知权重，控制粒子向个体最优位置移动的程度，默认为1.5
            social_weight: 社会认知权重，控制粒子向群体最优位置移动的程度，默认为1.5
            parallel: 是否使用并行计算
            
        返回:
            按目标指标排序的参数组合和结果列表
        """
        param_names = list(parameter_space.keys())
        self.logger.info(f"开始粒子群优化，粒子数量: {num_particles}, 迭代次数: {iterations}")
        
        # 初始化粒子群
        particles = self._initialize_particles(parameter_space, num_particles)
        
        # 初始化粒子的历史最优解
        particle_best_positions = deepcopy(particles)
        particle_best_fitness = [float('-inf')] * num_particles
        
        # 初始化全局最优解
        global_best_position = None
        global_best_fitness = float('-inf')
        
        # 初始化粒子速度
        velocities = self._initialize_velocities(parameter_space, num_particles)
        
        # 存储所有评估过的参数组合
        all_evaluations = []
        
        # 主循环
        for iteration in range(iterations):
            self.logger.info(f"开始第 {iteration + 1}/{iterations} 次迭代")
            
            # 为每个粒子的位置构建参数字典
            parameter_dicts = []
            for particle in particles:
                params = {}
                for i, param_name in enumerate(param_names):
                    # 确保参数值在有效范围内
                    param_values = parameter_space[param_name]
                    if isinstance(param_values[0], (int, float)) and len(param_values) > 1:
                        # 连续参数空间，取离散值中最接近的
                        closest_idx = min(range(len(param_values)), 
                                          key=lambda i: abs(param_values[i] - particle[i]))
                        params[param_name] = param_values[closest_idx]
                    else:
                        # 离散参数空间，取索引对应的值
                        idx = min(int(particle[i]) % len(param_values), len(param_values) - 1)
                        params[param_name] = param_values[idx]
                parameter_dicts.append(params)
            
            # 评估粒子适应度
            tasks = [(strategy, market_data, smart_money_data, symbol, params, objective) 
                     for params in parameter_dicts]
            
            if parallel and len(tasks) > 1:
                with multiprocessing.Pool(self.parallel_processes) as pool:
                    fitness_results = pool.map(self._execute_single_backtest, tasks)
            else:
                fitness_results = [self._execute_single_backtest(task) for task in tasks]
            
            # 更新粒子的历史最优解和全局最优解
            for i, (params, result) in enumerate(zip(parameter_dicts, fitness_results)):
                # 计算适应度值
                if objective.lower() in ['max_drawdown', '最大回撤']:
                    # 对于回撤类指标，越小越好，取相反数使之成为最大化问题
                    fitness = -result['backtest_results'].get(objective, result['backtest_results'].get('最大回撤', 1.0))
                else:
                    # 对于其他指标，越大越好
                    fitness = result['backtest_results'].get(objective, result['backtest_results'].get('收益回撤比', 0.0))
                
                # 存储评估结果
                all_evaluations.append({
                    'parameters': params,
                    'fitness': fitness,
                    'backtest_results': result['backtest_results']
                })
                
                # 更新粒子的历史最优解
                if fitness > particle_best_fitness[i]:
                    particle_best_positions[i] = deepcopy(particles[i])
                    particle_best_fitness[i] = fitness
                
                # 更新全局最优解
                if fitness > global_best_fitness:
                    global_best_position = deepcopy(particles[i])
                    global_best_fitness = fitness
            
            # 更新粒子速度和位置
            for i in range(num_particles):
                for j in range(len(param_names)):
                    # 随机因子
                    r1 = random.random()
                    r2 = random.random()
                    
                    # 更新速度
                    velocities[i][j] = inertia_weight * velocities[i][j] + \
                                       cognitive_weight * r1 * (particle_best_positions[i][j] - particles[i][j]) + \
                                       social_weight * r2 * (global_best_position[j] - particles[i][j])
                    
                    # 更新位置
                    particles[i][j] += velocities[i][j]
                    
                    # 确保位置在参数空间范围内
                    param_values = parameter_space[param_names[j]]
                    if isinstance(param_values[0], (int, float)) and len(param_values) > 1:
                        # 连续参数空间
                        min_val = min(param_values)
                        max_val = max(param_values)
                        particles[i][j] = max(min_val, min(max_val, particles[i][j]))
                    else:
                        # 离散参数空间，保持在合理范围内
                        particles[i][j] = max(0, particles[i][j]) % len(param_values)
            
            self.logger.info(f"第 {iteration + 1} 次迭代完成，当前最佳适应度: {global_best_fitness:.4f}")
        
        # 按适应度排序结果
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
        
        # 按目标指标排序
        if objective.lower() in ['max_drawdown', '最大回撤']:
            unique_results.sort(key=lambda x: x['backtest_results'].get(objective, x['backtest_results'].get('最大回撤', 1.0)), reverse=False)
        else:
            unique_results.sort(key=lambda x: x['backtest_results'].get(objective, x['backtest_results'].get('收益回撤比', 0.0)), reverse=True)
        
        self.logger.info(f"粒子群优化完成，找到 {len(unique_results)} 个唯一参数组合")
        
        best_params = unique_results[0]['parameters']
        best_value = unique_results[0]['backtest_results'].get(objective, unique_results[0]['backtest_results'].get('收益回撤比', 0.0))
        self.logger.info(f"最佳参数组合: {best_params}, {objective}值: {best_value:.4f}")
        
        return unique_results
    
    def _initialize_particles(self, parameter_space: Dict[str, List[Any]], num_particles: int) -> List[List[float]]:
        """初始化粒子位置"""
        particles = []
        for _ in range(num_particles):
            particle = []
            for param_values in parameter_space.values():
                if isinstance(param_values[0], (int, float)) and len(param_values) > 1:
                    # 连续参数空间
                    min_val = min(param_values)
                    max_val = max(param_values)
                    particle.append(random.uniform(min_val, max_val))
                else:
                    # 离散参数空间
                    particle.append(random.randint(0, len(param_values) - 1))
            particles.append(particle)
        return particles
    
    def _initialize_velocities(self, parameter_space: Dict[str, List[Any]], num_particles: int) -> List[List[float]]:
        """初始化粒子速度"""
        velocities = []
        for _ in range(num_particles):
            velocity = []
            for param_values in parameter_space.values():
                if isinstance(param_values[0], (int, float)) and len(param_values) > 1:
                    # 连续参数空间
                    min_val = min(param_values)
                    max_val = max(param_values)
                    range_val = max_val - min_val
                    velocity.append(random.uniform(-range_val * 0.1, range_val * 0.1))
                else:
                    # 离散参数空间
                    velocity.append(random.uniform(-0.5, 0.5))
            velocities.append(velocity)
        return velocities


# 中文命名版本
class 粒子群优化器(参数优化器):
    """粒子群优化器，通过模拟群体协作行为搜索参数空间（中文版）"""
    
    def 优化(self, 
           策略,
           市场数据: pd.DataFrame,
           庄家数据: pd.DataFrame,
           股票代码: str,
           参数空间: Dict[str, List[Any]],
           目标指标: str = '收益回撤比',
           粒子数量: int = 20,
           迭代次数: int = 10,
           惯性权重: float = 0.7,
           个体认知权重: float = 1.5,
           社会认知权重: float = 1.5,
           并行: bool = True,
           **kwargs) -> List[Dict[str, Any]]:
        """
        使用粒子群算法优化策略参数。
        
        参数:
            策略: 要优化的策略实例
            市场数据: 市场数据DataFrame
            庄家数据: 庄家指标数据DataFrame
            股票代码: 交易标的代码
            参数空间: 参数名到可能值列表的映射
            目标指标: 优化目标指标，默认为'收益回撤比'
            粒子数量: 粒子数量，默认为20
            迭代次数: 迭代次数，默认为10
            惯性权重: 惯性权重，控制粒子保持当前速度的程度，默认为0.7
            个体认知权重: 个体认知权重，控制粒子向个体最优位置移动的程度，默认为1.5
            社会认知权重: 社会认知权重，控制粒子向群体最优位置移动的程度，默认为1.5
            并行: 是否使用并行计算
            
        返回:
            按目标指标排序的参数组合和结果列表
        """
        参数名称 = list(参数空间.keys())
        self.日志器.info(f"开始粒子群优化，粒子数量: {粒子数量}, 迭代次数: {迭代次数}")
        
        # 初始化粒子群
        粒子群 = self._初始化粒子位置(参数空间, 粒子数量)
        
        # 初始化粒子的历史最优解
        粒子最优位置 = deepcopy(粒子群)
        粒子最优适应度 = [float('-inf')] * 粒子数量
        
        # 初始化全局最优解
        全局最优位置 = None
        全局最优适应度 = float('-inf')
        
        # 初始化粒子速度
        速度 = self._初始化粒子速度(参数空间, 粒子数量)
        
        # 存储所有评估过的参数组合
        所有评估 = []
        
        # 主循环
        for 当前迭代 in range(迭代次数):
            self.日志器.info(f"开始第 {当前迭代 + 1}/{迭代次数} 次迭代")
            
            # 为每个粒子的位置构建参数字典
            参数字典列表 = []
            for 粒子 in 粒子群:
                参数 = {}
                for i, 参数名 in enumerate(参数名称):
                    # 确保参数值在有效范围内
                    参数值列表 = 参数空间[参数名]
                    if isinstance(参数值列表[0], (int, float)) and len(参数值列表) > 1:
                        # 连续参数空间，取离散值中最接近的
                        最接近索引 = min(range(len(参数值列表)), 
                                      key=lambda i: abs(参数值列表[i] - 粒子[i]))
                        参数[参数名] = 参数值列表[最接近索引]
                    else:
                        # 离散参数空间，取索引对应的值
                        索引 = min(int(粒子[i]) % len(参数值列表), len(参数值列表) - 1)
                        参数[参数名] = 参数值列表[索引]
                参数字典列表.append(参数)
            
            # 评估粒子适应度
            任务列表 = [(策略, 市场数据, 庄家数据, 股票代码, 参数, 目标指标) 
                     for 参数 in 参数字典列表]
            
            if 并行 and len(任务列表) > 1:
                with multiprocessing.Pool(self.并行进程数) as pool:
                    适应度结果 = pool.map(self._执行单次回测, 任务列表)
            else:
                适应度结果 = [self._执行单次回测(任务) for 任务 in 任务列表]
            
            # 更新粒子的历史最优解和全局最优解
            for i, (参数, 结果) in enumerate(zip(参数字典列表, 适应度结果)):
                # 计算适应度值
                if 目标指标.lower() in ['max_drawdown', '最大回撤']:
                    # 对于回撤类指标，越小越好，取相反数使之成为最大化问题
                    适应度 = -结果['回测结果'].get(目标指标, 结果['回测结果'].get('最大回撤', 1.0))
                else:
                    # 对于其他指标，越大越好
                    适应度 = 结果['回测结果'].get(目标指标, 结果['回测结果'].get('收益回撤比', 0.0))
                
                # 存储评估结果
                所有评估.append({
                    '参数': 参数,
                    '适应度': 适应度,
                    '回测结果': 结果['回测结果']
                })
                
                # 更新粒子的历史最优解
                if 适应度 > 粒子最优适应度[i]:
                    粒子最优位置[i] = deepcopy(粒子群[i])
                    粒子最优适应度[i] = 适应度
                
                # 更新全局最优解
                if 适应度 > 全局最优适应度:
                    全局最优位置 = deepcopy(粒子群[i])
                    全局最优适应度 = 适应度
            
            # 更新粒子速度和位置
            for i in range(粒子数量):
                for j in range(len(参数名称)):
                    # 随机因子
                    r1 = random.random()
                    r2 = random.random()
                    
                    # 更新速度
                    速度[i][j] = 惯性权重 * 速度[i][j] + \
                                 个体认知权重 * r1 * (粒子最优位置[i][j] - 粒子群[i][j]) + \
                                 社会认知权重 * r2 * (全局最优位置[j] - 粒子群[i][j])
                    
                    # 更新位置
                    粒子群[i][j] += 速度[i][j]
                    
                    # 确保位置在参数空间范围内
                    参数值列表 = 参数空间[参数名称[j]]
                    if isinstance(参数值列表[0], (int, float)) and len(参数值列表) > 1:
                        # 连续参数空间
                        最小值 = min(参数值列表)
                        最大值 = max(参数值列表)
                        粒子群[i][j] = max(最小值, min(最大值, 粒子群[i][j]))
                    else:
                        # 离散参数空间，保持在合理范围内
                        粒子群[i][j] = max(0, 粒子群[i][j]) % len(参数值列表)
            
            self.日志器.info(f"第 {当前迭代 + 1} 次迭代完成，当前最佳适应度: {全局最优适应度:.4f}")
        
        # 按适应度排序结果
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
        
        # 按目标指标排序
        if 目标指标.lower() in ['max_drawdown', '最大回撤']:
            唯一结果.sort(key=lambda x: x['回测结果'].get(目标指标, x['回测结果'].get('最大回撤', 1.0)), reverse=False)
        else:
            唯一结果.sort(key=lambda x: x['回测结果'].get(目标指标, x['回测结果'].get('收益回撤比', 0.0)), reverse=True)
        
        self.日志器.info(f"粒子群优化完成，找到 {len(唯一结果)} 个唯一参数组合")
        
        最佳参数 = 唯一结果[0]['参数']
        最佳值 = 唯一结果[0]['回测结果'].get(目标指标, 唯一结果[0]['回测结果'].get('收益回撤比', 0.0))
        self.日志器.info(f"最佳参数组合: {最佳参数}, {目标指标}值: {最佳值:.4f}")
        
        return 唯一结果
    
    def _初始化粒子位置(self, 参数空间: Dict[str, List[Any]], 粒子数量: int) -> List[List[float]]:
        """初始化粒子位置"""
        粒子群 = []
        for _ in range(粒子数量):
            粒子 = []
            for 参数值列表 in 参数空间.values():
                if isinstance(参数值列表[0], (int, float)) and len(参数值列表) > 1:
                    # 连续参数空间
                    最小值 = min(参数值列表)
                    最大值 = max(参数值列表)
                    粒子.append(random.uniform(最小值, 最大值))
                else:
                    # 离散参数空间
                    粒子.append(random.randint(0, len(参数值列表) - 1))
            粒子群.append(粒子)
        return 粒子群
    
    def _初始化粒子速度(self, 参数空间: Dict[str, List[Any]], 粒子数量: int) -> List[List[float]]:
        """初始化粒子速度"""
        速度列表 = []
        for _ in range(粒子数量):
            速度 = []
            for 参数值列表 in 参数空间.values():
                if isinstance(参数值列表[0], (int, float)) and len(参数值列表) > 1:
                    # 连续参数空间
                    最小值 = min(参数值列表)
                    最大值 = max(参数值列表)
                    范围 = 最大值 - 最小值
                    速度.append(random.uniform(-范围 * 0.1, 范围 * 0.1))
                else:
                    # 离散参数空间
                    速度.append(random.uniform(-0.5, 0.5))
            速度列表.append(速度)
        return 速度列表 