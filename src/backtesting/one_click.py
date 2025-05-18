"""
一键回测模块

提供简化的回测流程，自动处理数据获取、清洗、回测和结果生成。

日期：2025-05-17
"""

import logging
import datetime
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .engines import BacktestEngine, 回测引擎
from .optimizers import ParameterOptimizer, 参数优化器
from ..utils.data_utils import DataUtils

# 配置日志
logger = logging.getLogger(__name__)


class OneClickBacktest:
    """一键回测工具类，提供简化的回测流程"""
    
    def __init__(self, 
                data_source_factory,
                strategy_factory,
                backtest_engine = None,
                parameter_optimizer = None):
        """
        初始化一键回测器。
        
        参数:
            data_source_factory: 用于创建数据源的工厂
            strategy_factory: 用于创建策略的工厂
            backtest_engine: 回测引擎实例，如果为None则创建新实例
            parameter_optimizer: 参数优化器实例，如果为None则创建新实例
        """
        self.data_source_factory = data_source_factory
        self.strategy_factory = strategy_factory
        self.backtest_engine = backtest_engine if backtest_engine is not None else BacktestEngine()
        self.parameter_optimizer = parameter_optimizer if parameter_optimizer is not None else ParameterOptimizer(self.backtest_engine)
        
    def run_backtest(self, 
                    symbol: str,
                    strategy_name: str,
                    start_date: str,
                    end_date: str,
                    market_type: str = 'stock',
                    strategy_parameters: Optional[Dict[str, Any]] = None,
                    data_cleaning_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        一键执行回测流程。
        
        参数:
            symbol: 交易标的代码
            strategy_name: 策略名称
            start_date: 回测开始日期，格式为YYYY-MM-DD
            end_date: 回测结束日期，格式为YYYY-MM-DD
            market_type: 市场类型，如'stock'或'crypto'
            strategy_parameters: 策略参数字典，如果为None则使用默认参数
            data_cleaning_config: 数据清洗配置，如果为None则使用默认配置
            
        返回:
            包含回测结果的字典
        """
        logger.info(f"开始一键回测，股票代码: {symbol}，策略: {strategy_name}，时间段: {start_date} 到 {end_date}")
        
        # 1. 获取数据
        data_source = self.data_source_factory.create_data_source(market_type)
        
        market_data = data_source.get_market_data(symbol, start_date, end_date)
        smart_money_data = data_source.get_smart_money_indicators(symbol, start_date, end_date)
        
        # 2. 清洗数据
        if data_cleaning_config is not None:
            market_data = DataUtils.clean_data(market_data, data_cleaning_config)
            smart_money_data = DataUtils.clean_data(smart_money_data, data_cleaning_config)
        
        # 3. 创建策略
        strategy = self.strategy_factory.create_strategy(strategy_name, strategy_parameters)
        
        # 4. 生成信号
        signal_data = strategy.generate_signals(market_data, smart_money_data)
        
        # 5. 执行回测
        backtest_results = self.backtest_engine.run_backtest(market_data, signal_data, symbol)
        
        # 6. 生成回测报告
        backtest_report = self._generate_backtest_report(backtest_results, market_data, signal_data, symbol, strategy)
        
        logger.info(f"一键回测完成，最终资产: {backtest_results['final_equity']:.2f}，收益率: {backtest_results['return']*100:.2f}%")
        return backtest_report
    
    def run_parameter_optimization(self, 
                                 symbol: str,
                                 strategy_name: str,
                                 start_date: str,
                                 end_date: str,
                                 parameter_space: Dict[str, Any],
                                 optimization_method: str = 'grid_search',
                                 objective: str = 'calmar_ratio',
                                 market_type: str = 'stock',
                                 data_cleaning_config: Optional[Dict[str, Any]] = None,
                                 optimization_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        一键执行参数优化流程。
        
        参数:
            symbol: 交易标的代码
            strategy_name: 策略名称
            start_date: 回测开始日期，格式为YYYY-MM-DD
            end_date: 回测结束日期，格式为YYYY-MM-DD
            parameter_space: 参数搜索空间
            optimization_method: 优化方法，支持'grid_search', 'random_search'等
            objective: 优化目标指标，默认为'calmar_ratio'
            market_type: 市场类型，如'stock'或'crypto'
            data_cleaning_config: 数据清洗配置，如果为None则使用默认配置
            optimization_params: 优化方法的特定参数
            
        返回:
            包含优化结果的字典
        """
        logger.info(f"开始一键参数优化，股票代码: {symbol}，策略: {strategy_name}，优化方法: {optimization_method}")
        
        # 1. 获取数据
        data_source = self.data_source_factory.create_data_source(market_type)
        
        market_data = data_source.get_market_data(symbol, start_date, end_date)
        smart_money_data = data_source.get_smart_money_indicators(symbol, start_date, end_date)
        
        # 2. 清洗数据
        if data_cleaning_config is not None:
            market_data = DataUtils.clean_data(market_data, data_cleaning_config)
            smart_money_data = DataUtils.clean_data(smart_money_data, data_cleaning_config)
        
        # 3. 创建策略
        strategy = self.strategy_factory.create_strategy(strategy_name)
        
        # 4. 执行参数优化
        optimization_params = optimization_params if optimization_params is not None else {}
        
        # 基于优化方法选择合适的优化器
        if optimization_method == 'grid_search':
            from .optimizers.grid_search import GridSearchOptimizer
            optimizer = GridSearchOptimizer(self.backtest_engine)
            optimization_results = optimizer.optimize(
                strategy, market_data, smart_money_data, symbol, parameter_space, objective, **optimization_params
            )
        elif optimization_method == 'random_search':
            from .optimizers.random_search import RandomSearchOptimizer
            optimizer = RandomSearchOptimizer(self.backtest_engine)
            optimization_results = optimizer.optimize(
                strategy, market_data, smart_money_data, symbol, parameter_space, objective, **optimization_params
            )
        elif optimization_method == 'bayesian_optimization':
            from .optimizers.bayesian_optimization import BayesianOptimizer
            optimizer = BayesianOptimizer(self.backtest_engine)
            optimization_results = optimizer.optimize(
                strategy, market_data, smart_money_data, symbol, parameter_space, objective, **optimization_params
            )
        else:
            raise ValueError(f"不支持的优化方法: {optimization_method}")
        
        # 5. 生成优化报告
        optimization_report = self._generate_optimization_report(
            optimization_results, market_data, symbol, strategy_name, optimization_method, objective
        )
        
        logger.info(f"一键参数优化完成，找到 {len(optimization_results)} 个结果")
        return optimization_report
    
    def _generate_backtest_report(self, 
                               backtest_results: Dict[str, Any],
                               market_data: pd.DataFrame,
                               signal_data: pd.DataFrame,
                               symbol: str,
                               strategy) -> Dict[str, Any]:
        """生成详细的回测报告。"""
        # 基本信息
        report = {
            'basic_info': {
                'symbol': symbol,
                'strategy_name': strategy.name,
                'strategy_description': strategy.description,
                'strategy_parameters': strategy.parameters,
                'backtest_start_date': market_data.index[0].strftime('%Y-%m-%d'),
                'backtest_end_date': market_data.index[-1].strftime('%Y-%m-%d'),
                'backtest_days': len(market_data)
            },
            'backtest_results': {
                'initial_equity': backtest_results['initial_equity'],
                'final_equity': backtest_results['final_equity'],
                'total_return': backtest_results['return'],
                'annual_return': backtest_results['annual_return'],
                'max_drawdown': backtest_results['max_drawdown'],
                'calmar_ratio': backtest_results['calmar_ratio'],
                'sharpe_ratio': backtest_results['sharpe_ratio'],
                'trade_count': backtest_results['trade_count'],
                'profit_count': backtest_results.get('profit_count', 0),
                'loss_count': backtest_results.get('loss_count', 0),
                'win_rate': backtest_results.get('win_rate', 0),
                'profit_loss_ratio': backtest_results.get('profit_loss_ratio', 0)
            },
            'trade_records': backtest_results['trade_records'],
            'equity_curve': backtest_results['equity_curve'].to_dict('records')
        }
        
        # 生成图表
        chart_path = self._generate_backtest_charts(backtest_results, market_data, signal_data, symbol)
        report['chart_path'] = chart_path
        
        return report
    
    def _generate_optimization_report(self, 
                                   optimization_results: List[Dict[str, Any]],
                                   market_data: pd.DataFrame,
                                   symbol: str,
                                   strategy_name: str,
                                   optimization_method: str,
                                   objective: str) -> Dict[str, Any]:
        """生成详细的参数优化报告。"""
        # 基本信息
        report = {
            'basic_info': {
                'symbol': symbol,
                'strategy_name': strategy_name,
                'optimization_method': optimization_method,
                'objective': objective,
                'backtest_start_date': market_data.index[0].strftime('%Y-%m-%d'),
                'backtest_end_date': market_data.index[-1].strftime('%Y-%m-%d'),
                'backtest_days': len(market_data),
                'result_count': len(optimization_results)
            },
            'best_result': optimization_results[0] if optimization_results else None,
            'all_results': optimization_results
        }
        
        # 生成参数分布图表
        chart_path = self._generate_parameter_distribution_charts(optimization_results, objective)
        report['chart_path'] = chart_path
        
        return report
    
    def _generate_backtest_charts(self, 
                               backtest_results: Dict[str, Any],
                               market_data: pd.DataFrame,
                               signal_data: pd.DataFrame,
                               symbol: str) -> str:
        """生成回测结果图表。"""
        # 创建图表目录
        chart_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'charts')
        os.makedirs(chart_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{symbol}_backtest_{timestamp}.png"
        filepath = os.path.join(chart_dir, filename)
        
        # 创建图表
        plt.figure(figsize=(12, 10))
        
        # 1. 资金曲线
        plt.subplot(3, 1, 1)
        equity_curve = backtest_results['equity_curve']
        plt.plot(equity_curve.index, equity_curve['equity'], label='Total Equity')
        plt.plot(equity_curve.index, equity_curve['cash'], label='Cash', alpha=0.7)
        plt.plot(equity_curve.index, equity_curve['position_value'], label='Position Value', alpha=0.7)
        
        # 添加买入卖出标记
        for trade in backtest_results['trade_records']:
            trade_date = trade.get('date', trade.get('日期'))
            if trade.get('type', trade.get('类型', '')).lower() in ['buy', '买入']:
                plt.scatter(trade_date, trade.get('remaining_capital', trade.get('剩余资金', 0)) + trade.get('amount', trade.get('交易金额', 0)), 
                         marker='^', color='red', s=100)
            elif trade.get('type', trade.get('类型', '')).lower() in ['sell', '卖出']:
                plt.scatter(trade_date, trade.get('remaining_capital', trade.get('剩余资金', 0)), 
                         marker='v', color='green', s=100)
        
        plt.title(f"{symbol} Backtest Equity Curve")
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.legend()
        plt.grid(True)
        
        # 2. 价格和信号
        plt.subplot(3, 1, 2)
        plt.plot(market_data.index, market_data['close'], label='Close Price')
        
        # 添加买入卖出信号
        buy_points = signal_data[signal_data['信号'] == 1].index
        sell_points = signal_data[signal_data['信号'] == -1].index
        
        plt.scatter(buy_points, market_data.loc[buy_points]['close'], 
                 marker='^', color='red', s=100, label='Buy Signal')
        plt.scatter(sell_points, market_data.loc[sell_points]['close'], 
                 marker='v', color='green', s=100, label='Sell Signal')
        
        plt.title(f"{symbol} Price and Trading Signals")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # 3. 智能资金指标
        if '操纵分数' in signal_data.columns:
            plt.subplot(3, 1, 3)
            plt.plot(signal_data.index, signal_data['操纵分数'], label='Smart Money Score')
            
            if '成交量异常分数' in signal_data.columns:
                plt.plot(signal_data.index, signal_data['成交量异常分数'], 
                       label='Volume Anomaly Score', alpha=0.7)
            
            if '价格模式分数' in signal_data.columns:
                plt.plot(signal_data.index, signal_data['价格模式分数'], 
                       label='Price Pattern Score', alpha=0.7)
            
            if '机构活动分数' in signal_data.columns:
                plt.plot(signal_data.index, signal_data['机构活动分数'], 
                       label='Institutional Activity Score', alpha=0.7)
            
            plt.title(f"{symbol} Smart Money Indicators")
            plt.xlabel('Date')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        return filepath
    
    def _generate_parameter_distribution_charts(self, 
                                            optimization_results: List[Dict[str, Any]],
                                            objective: str) -> str:
        """生成参数分布图表。"""
        if not optimization_results:
            return ""
        
        # 创建图表目录
        chart_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'charts')
        os.makedirs(chart_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"parameter_optimization_{timestamp}.png"
        filepath = os.path.join(chart_dir, filename)
        
        # 提取参数和目标指标值
        param_names = list(optimization_results[0]['parameters'].keys())
        objective_key = objective
        
        # 兼容中英文指标名
        if objective not in optimization_results[0]['backtest_results']:
            for key in optimization_results[0]['backtest_results'].keys():
                if key.lower() == objective.lower():
                    objective_key = key
                    break
        
        objective_values = [result['backtest_results'][objective_key] for result in optimization_results]
        
        # 创建图表
        param_count = len(param_names)
        chart_rows = (param_count + 1) // 2
        
        plt.figure(figsize=(12, 4 * chart_rows))
        
        # 为每个参数创建散点图
        for i, param_name in enumerate(param_names):
            plt.subplot(chart_rows, 2, i + 1)
            
            param_values = [result['parameters'][param_name] for result in optimization_results]
            
            plt.scatter(param_values, objective_values, alpha=0.7)
            plt.title(f"{param_name} vs {objective}")
            plt.xlabel(param_name)
            plt.ylabel(objective)
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        return filepath


# 中文命名版本
class 一键回测器:
    """一键回测工具类，提供简化的回测流程（中文版）"""
    
    def __init__(self, 
                数据源工厂,
                策略工厂,
                回测引擎 = None,
                参数优化器 = None):
        """
        初始化一键回测器。
        
        参数:
            数据源工厂: 用于创建数据源的工厂
            策略工厂: 用于创建策略的工厂
            回测引擎: 回测引擎实例，如果为None则创建新实例
            参数优化器: 参数优化器实例，如果为None则创建新实例
        """
        self.数据源工厂 = 数据源工厂
        self.策略工厂 = 策略工厂
        self.回测引擎 = 回测引擎 if 回测引擎 is not None else 回测引擎()
        self.参数优化器 = 参数优化器 if 参数优化器 is not None else 参数优化器(self.回测引擎)
        
    def 执行回测(self, 
               股票代码: str,
               策略名称: str,
               开始日期: str,
               结束日期: str,
               市场类型: str = 'a股',
               策略参数: Optional[Dict[str, Any]] = None,
               数据清洗配置: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        一键执行回测流程。
        
        参数:
            股票代码: 交易标的代码
            策略名称: 策略名称
            开始日期: 回测开始日期，格式为YYYY-MM-DD
            结束日期: 回测结束日期，格式为YYYY-MM-DD
            市场类型: 市场类型，如'a股'或'虚拟货币'
            策略参数: 策略参数字典，如果为None则使用默认参数
            数据清洗配置: 数据清洗配置，如果为None则使用默认配置
            
        返回:
            包含回测结果的字典
        """
        logger.info(f"开始一键回测，股票代码: {股票代码}，策略: {策略名称}，时间段: {开始日期} 到 {结束日期}")
        
        # 1. 获取数据
        数据源 = self.数据源工厂.创建数据源(市场类型)
        
        市场数据 = 数据源.获取市场数据(股票代码, 开始日期, 结束日期)
        庄家数据 = 数据源.获取庄家指标(股票代码, 开始日期, 结束日期)
        
        # 2. 清洗数据
        if 数据清洗配置 is not None:
            市场数据 = DataUtils.清洗数据(市场数据, 数据清洗配置)
            庄家数据 = DataUtils.清洗数据(庄家数据, 数据清洗配置)
        
        # 3. 创建策略
        策略 = self.策略工厂.创建策略(策略名称, 策略参数)
        
        # 4. 生成信号
        信号数据 = 策略.生成信号(市场数据, 庄家数据)
        
        # 5. 执行回测
        回测结果 = self.回测引擎.执行回测(市场数据, 信号数据, 股票代码)
        
        # 6. 生成回测报告
        回测报告 = self._生成回测报告(回测结果, 市场数据, 信号数据, 股票代码, 策略)
        
        logger.info(f"一键回测完成，最终资产: {回测结果['最终资产']:.2f}，收益率: {回测结果['收益率']*100:.2f}%")
        return 回测报告
    
    def 执行参数优化(self, 
                 股票代码: str,
                 策略名称: str,
                 开始日期: str,
                 结束日期: str,
                 参数空间: Dict[str, Any],
                 优化方法: str = '网格搜索',
                 目标指标: str = '收益回撤比',
                 市场类型: str = 'a股',
                 数据清洗配置: Optional[Dict[str, Any]] = None,
                 优化参数: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        一键执行参数优化流程。
        
        参数:
            股票代码: 交易标的代码
            策略名称: 策略名称
            开始日期: 回测开始日期，格式为YYYY-MM-DD
            结束日期: 回测结束日期，格式为YYYY-MM-DD
            参数空间: 参数搜索空间
            优化方法: 优化方法，支持'网格搜索', '随机搜索'等
            目标指标: 优化目标指标，默认为'收益回撤比'
            市场类型: 市场类型，如'a股'或'虚拟货币'
            数据清洗配置: 数据清洗配置，如果为None则使用默认配置
            优化参数: 优化方法的特定参数
            
        返回:
            包含优化结果的字典
        """
        # 转换中文优化方法到英文
        优化方法映射 = {
            '网格搜索': 'grid_search',
            '随机搜索': 'random_search',
            '贝叶斯优化': 'bayesian_optimization',
            '遗传算法': 'genetic_algorithm',
            '粒子群优化': 'particle_swarm',
            '模拟退火': 'simulated_annealing',
            '多目标优化': 'multi_objective'
        }
        英文优化方法 = 优化方法映射.get(优化方法, 优化方法)
        
        # 转换中文指标到英文
        指标映射 = {
            '收益回撤比': 'calmar_ratio',
            '夏普比率': 'sharpe_ratio',
            '最大回撤': 'max_drawdown',
            '年化收益率': 'annual_return',
            '收益率': 'total_return',
            '胜率': 'win_rate',
            '盈亏比': 'profit_loss_ratio'
        }
        英文指标 = 指标映射.get(目标指标, 目标指标)
        
        # 创建一键回测器实例并调用英文方法
        one_click = OneClickBacktest(self.数据源工厂, self.策略工厂, self.回测引擎, self.参数优化器)
        return one_click.run_parameter_optimization(
            股票代码, 策略名称, 开始日期, 结束日期, 参数空间, 英文优化方法, 英文指标, 
            市场类型, 数据清洗配置, 优化参数
        )
    
    def _生成回测报告(self, 
                  回测结果: Dict[str, Any],
                  市场数据: pd.DataFrame,
                  信号数据: pd.DataFrame,
                  股票代码: str,
                  策略) -> Dict[str, Any]:
        """生成详细的回测报告。"""
        # 基本信息
        报告 = {
            '基本信息': {
                '股票代码': 股票代码,
                '策略名称': 策略.名称,
                '策略描述': 策略.描述,
                '策略参数': 策略.参数,
                '回测开始日期': 市场数据.index[0].strftime('%Y-%m-%d'),
                '回测结束日期': 市场数据.index[-1].strftime('%Y-%m-%d'),
                '回测天数': len(市场数据)
            },
            '回测结果': {
                '初始资产': 回测结果['初始资产'],
                '最终资产': 回测结果['最终资产'],
                '收益率': 回测结果['收益率'],
                '年化收益率': 回测结果['年化收益率'],
                '最大回撤': 回测结果['最大回撤'],
                '收益回撤比': 回测结果['收益回撤比'],
                '夏普比率': 回测结果['夏普比率'],
                '交易次数': 回测结果['交易次数'],
                '盈利次数': 回测结果.get('盈利次数', 0),
                '亏损次数': 回测结果.get('亏损次数', 0),
                '胜率': 回测结果.get('胜率', 0),
                '盈亏比': 回测结果.get('盈亏比', 0)
            },
            '交易记录': 回测结果['交易记录'],
            '资金曲线': 回测结果['资金曲线'].to_dict('records')
        }
        
        # 生成图表
        图表路径 = self._生成回测图表(回测结果, 市场数据, 信号数据, 股票代码)
        报告['图表路径'] = 图表路径
        
        return 报告
    
    def _生成回测图表(self, 
                  回测结果: Dict[str, Any],
                  市场数据: pd.DataFrame,
                  信号数据: pd.DataFrame,
                  股票代码: str) -> str:
        """生成回测结果图表。"""
        # 创建一键回测器实例并调用英文方法
        one_click = OneClickBacktest(self.数据源工厂, self.策略工厂, self.回测引擎, self.参数优化器)
        return one_click._generate_backtest_charts(回测结果, 市场数据, 信号数据, 股票代码) 