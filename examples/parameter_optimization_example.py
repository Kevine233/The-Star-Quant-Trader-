"""
参数优化示例脚本

演示如何使用不同的参数优化器进行策略参数优化。

日期：2025-05-17
"""

import sys
import os
import pandas as pd
import numpy as np
import datetime
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入回测和优化模块
from src.backtesting.engines import BacktestEngine
from src.backtesting.optimizers import (
    GridSearchOptimizer, 
    RandomSearchOptimizer, 
    BayesianOptimizer
)

# 导入中文版模块（可选）
from src.backtesting.engines import 回测引擎
from src.backtesting.optimizers import (
    网格搜索优化器,
    随机搜索优化器,
    贝叶斯优化器
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 简单移动平均策略
class SmaStrategy:
    """简单移动平均策略示例"""
    
    def __init__(self, params=None):
        """初始化策略"""
        self.name = "SMA交叉策略"
        self.description = "使用短期和长期移动平均线交叉生成交易信号"
        self.parameters = params or {
            'short_window': 20,
            'long_window': 50
        }
    
    def set_parameters(self, params):
        """设置策略参数"""
        self.parameters.update(params)
    
    def clone(self):
        """复制策略实例"""
        return SmaStrategy(self.parameters.copy())
    
    def generate_signals(self, market_data, smart_money_data=None):
        """生成交易信号"""
        short_window = self.parameters['short_window']
        long_window = self.parameters['long_window']
        
        # 复制数据，避免修改原始数据
        data = market_data.copy()
        
        # 计算移动平均线
        data['short_ma'] = data['close'].rolling(window=short_window).mean()
        data['long_ma'] = data['close'].rolling(window=long_window).mean()
        
        # 生成交易信号
        data['信号'] = 0  # 0表示无信号
        
        # 短期均线上穿长期均线，买入信号
        data.loc[(data['short_ma'] > data['long_ma']) & 
                (data['short_ma'].shift(1) <= data['long_ma'].shift(1)), '信号'] = 1
        
        # 短期均线下穿长期均线，卖出信号
        data.loc[(data['short_ma'] < data['long_ma']) & 
                (data['short_ma'].shift(1) >= data['long_ma'].shift(1)), '信号'] = -1
        
        # 只保留有信号的行
        signal_data = data[['信号']].copy()
        
        return signal_data


# 为测试创建示例数据
def create_sample_data(start_date='2020-01-01', end_date='2021-01-01'):
    """创建样本数据"""
    # 创建日期范围
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 模拟股票价格
    n = len(dates)
    np.random.seed(42)  # 设置随机种子，确保结果可重现
    
    # 生成随机价格变动
    price_changes = np.random.normal(0.0005, 0.015, n)
    
    # 添加趋势
    trend = np.linspace(0, 0.5, n)
    price_changes = price_changes + trend
    
    # 计算价格
    prices = 100 * (1 + price_changes).cumprod()
    
    # 创建OHLCV数据
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.003, n)),
        'high': prices * (1 + np.random.normal(0.005, 0.003, n)),
        'low': prices * (1 - np.random.normal(0.005, 0.003, n)),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n)
    }, index=dates)
    
    # 过滤掉周末
    data = data[data.index.dayofweek < 5]
    
    return data


def main():
    """主函数，运行参数优化示例"""
    logger.info("开始参数优化示例")
    
    # 创建示例数据
    market_data = create_sample_data('2020-01-01', '2021-01-01')
    logger.info(f"创建了示例数据，共 {len(market_data)} 个交易日")
    
    # 创建策略实例
    strategy = SmaStrategy()
    
    # 创建回测引擎
    engine = BacktestEngine(initial_capital=100000.0, commission_rate=0.0003)
    
    # 定义参数空间
    parameter_space = {
        'short_window': list(range(5, 30, 5)),  # [5, 10, 15, 20, 25]
        'long_window': list(range(30, 101, 10))  # [30, 40, 50, 60, 70, 80, 90, 100]
    }
    
    # 执行网格搜索优化
    logger.info("执行网格搜索优化")
    grid_optimizer = GridSearchOptimizer(engine)
    grid_results = grid_optimizer.optimize(
        strategy,
        market_data,
        None,  # 没有smart_money_data
        'EXAMPLE',
        parameter_space,
        objective='calmar_ratio',
        parallel=True
    )
    
    logger.info(f"网格搜索找到 {len(grid_results)} 个结果")
    best_grid_params = grid_results[0]['parameters']
    best_grid_calmar = grid_results[0]['backtest_results']['calmar_ratio']
    logger.info(f"最佳参数: {best_grid_params}, 收益回撤比: {best_grid_calmar:.4f}")
    
    # 执行随机搜索优化
    logger.info("执行随机搜索优化")
    random_optimizer = RandomSearchOptimizer(engine)
    random_results = random_optimizer.optimize(
        strategy,
        market_data,
        None,
        'EXAMPLE',
        parameter_space,
        objective='calmar_ratio',
        n_iter=10
    )
    
    logger.info(f"随机搜索找到 {len(random_results)} 个结果")
    best_random_params = random_results[0]['parameters']
    best_random_calmar = random_results[0]['backtest_results']['calmar_ratio']
    logger.info(f"最佳参数: {best_random_params}, 收益回撤比: {best_random_calmar:.4f}")
    
    # 尝试执行贝叶斯优化（如果安装了scikit-optimize）
    try:
        logger.info("执行贝叶斯优化")
        bayesian_optimizer = BayesianOptimizer(engine)
        bayesian_results = bayesian_optimizer.optimize(
            strategy,
            market_data,
            None,
            'EXAMPLE',
            parameter_space,
            objective='calmar_ratio',
            n_calls=10
        )
        
        if bayesian_results:
            logger.info(f"贝叶斯优化找到 {len(bayesian_results)} 个结果")
            best_bayesian_params = bayesian_results[0]['parameters']
            best_bayesian_calmar = bayesian_results[0]['backtest_results']['calmar_ratio']
            logger.info(f"最佳参数: {best_bayesian_params}, 收益回撤比: {best_bayesian_calmar:.4f}")
    except Exception as e:
        logger.warning(f"贝叶斯优化失败: {e}")
    
    # 比较不同方法的结果
    logger.info("优化结果比较:")
    logger.info(f"网格搜索: 短期={best_grid_params['short_window']}, 长期={best_grid_params['long_window']}, 收益回撤比={best_grid_calmar:.4f}")
    logger.info(f"随机搜索: 短期={best_random_params['short_window']}, 长期={best_random_params['long_window']}, 收益回撤比={best_random_calmar:.4f}")
    
    logger.info("参数优化示例完成")


# 使用中文版API的示例
def 中文示例():
    """使用中文API的示例函数"""
    logger.info("开始中文API参数优化示例")
    
    # 创建示例数据
    市场数据 = create_sample_data('2020-01-01', '2021-01-01')
    logger.info(f"创建了示例数据，共 {len(市场数据)} 个交易日")
    
    # 创建策略实例
    class 均线策略:
        def __init__(self, 参数=None):
            self.名称 = "均线交叉策略"
            self.描述 = "使用短期和长期移动平均线交叉生成交易信号"
            self.参数 = 参数 or {'短期窗口': 20, '长期窗口': 50}
            
        def 设置参数(self, 参数):
            self.参数.update(参数)
            
        def 克隆(self):
            return 均线策略(self.参数.copy())
            
        def 生成信号(self, 市场数据, 庄家数据=None):
            短期窗口 = self.参数['短期窗口']
            长期窗口 = self.参数['长期窗口']
            
            数据 = 市场数据.copy()
            数据['短期均线'] = 数据['close'].rolling(window=短期窗口).mean()
            数据['长期均线'] = 数据['close'].rolling(window=长期窗口).mean()
            
            数据['信号'] = 0
            数据.loc[(数据['短期均线'] > 数据['长期均线']) & (数据['短期均线'].shift(1) <= 数据['长期均线'].shift(1)), '信号'] = 1
            数据.loc[(数据['短期均线'] < 数据['长期均线']) & (数据['短期均线'].shift(1) >= 数据['长期均线'].shift(1)), '信号'] = -1
            
            信号数据 = 数据[['信号']].copy()
            return 信号数据
    
    策略 = 均线策略()
    
    # 创建回测引擎
    引擎 = 回测引擎(初始资金=100000.0, 手续费率=0.0003)
    
    # 定义参数空间
    参数空间 = {
        '短期窗口': list(range(5, 30, 5)),
        '长期窗口': list(range(30, 101, 10))
    }
    
    # 执行网格搜索优化
    logger.info("执行网格搜索优化（中文API）")
    网格优化器 = 网格搜索优化器(引擎)
    网格结果 = 网格优化器.优化(
        策略,
        市场数据,
        None,
        'EXAMPLE',
        参数空间,
        目标指标='收益回撤比',
        并行=True
    )
    
    logger.info(f"网格搜索找到 {len(网格结果)} 个结果")
    最佳网格参数 = 网格结果[0]['参数']
    最佳网格收益回撤比 = 网格结果[0]['回测结果']['收益回撤比']
    logger.info(f"最佳参数: {最佳网格参数}, 收益回撤比: {最佳网格收益回撤比:.4f}")
    
    logger.info("中文API参数优化示例完成")


if __name__ == "__main__":
    main()
    # 取消下面的注释以运行中文版示例
    # 中文示例()