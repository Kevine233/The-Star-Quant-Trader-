"""
回测模块，用于评估交易策略的历史表现。
本模块支持一键回测和自动参数优化功能。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import datetime
import logging
import itertools
import multiprocessing
import os
import json
import matplotlib.pyplot as plt
from ..strategies import 策略
from ..data_sources import 数据源
from ..utils.data_cleaner import 数据清洗器

# 配置日志
logger = logging.getLogger(__name__)

class 回测引擎:
    """
    回测引擎类，用于评估交易策略在历史数据上的表现。
    支持一键回测和自动参数优化功能。
    """
    
    def __init__(self, 
               初始资金: float = 100000.0,
               手续费率: float = 0.0003,
               滑点: float = 0.0001):
        """
        初始化回测引擎。
        
        参数:
            初始资金: 回测初始资金
            手续费率: 交易手续费率（买入和卖出都适用）
            滑点: 交易滑点率
        """
        self.初始资金 = 初始资金
        self.手续费率 = 手续费率
        self.滑点 = 滑点
        self.重置()
    
    def 重置(self):
        """重置回测状态。"""
        self.资金 = self.初始资金
        self.持仓 = {}  # 股票代码 -> 持仓数量
        self.持仓成本 = {}  # 股票代码 -> 平均成本
        self.交易记录 = []
        self.资金曲线 = []
        self.当前日期 = None
    
    def 执行回测(self, 
              市场数据: pd.DataFrame, 
              信号数据: pd.DataFrame,
              股票代码: str) -> Dict[str, Any]:
        """
        执行回测过程。
        
        参数:
            市场数据: 包含OHLCV数据的DataFrame
            信号数据: 包含交易信号的DataFrame
            股票代码: 交易标的代码
            
        返回:
            包含回测结果的字典
        """
        logger.info(f"开始回测 {股票代码}")
        
        # 重置回测状态
        self.重置()
        
        # 确保数据索引对齐
        共同索引 = 市场数据.index.intersection(信号数据.index)
        市场数据 = 市场数据.loc[共同索引]
        信号数据 = 信号数据.loc[共同索引]
        
        # 初始化结果记录
        self.资金曲线 = [{'日期': 市场数据.index[0], '资产': self.初始资金, '现金': self.初始资金, '持仓': 0.0}]
        
        # 遍历每个交易日
        for i in range(len(市场数据)):
            当前日期 = 市场数据.index[i]
            self.当前日期 = 当前日期
            
            # 获取当日价格和信号
            开盘价 = 市场数据['open'].iloc[i]
            最高价 = 市场数据['high'].iloc[i]
            最低价 = 市场数据['low'].iloc[i]
            收盘价 = 市场数据['close'].iloc[i]
            信号 = 信号数据['信号'].iloc[i]
            
            # 处理交易信号
            if 信号 == 1:  # 买入信号
                self._执行买入(股票代码, 收盘价, 当前日期, 信号数据.iloc[i])
            elif 信号 == -1:  # 卖出信号
                self._执行卖出(股票代码, 收盘价, 当前日期, 信号数据.iloc[i])
            
            # 更新资金曲线
            总资产 = self._计算总资产(股票代码, 收盘价)
            持仓价值 = 总资产 - self.资金
            self.资金曲线.append({
                '日期': 当前日期,
                '资产': 总资产,
                '现金': self.资金,
                '持仓': 持仓价值
            })
        
        # 计算回测结果指标
        结果 = self._计算回测指标()
        
        logger.info(f"回测完成，最终资产: {结果['最终资产']:.2f}，收益率: {结果['收益率']*100:.2f}%")
        return 结果
    
    def _执行买入(self, 股票代码: str, 价格: float, 日期: datetime.datetime, 信号行: pd.Series):
        """执行买入操作。"""
        # 考虑滑点
        实际价格 = 价格 * (1 + self.滑点)
        
        # 计算可买入数量（使用可用资金的90%，避免资金不足）
        可用资金 = self.资金 * 0.9
        数量 = int(可用资金 / 实际价格)
        
        if 数量 <= 0:
            logger.warning(f"{日期}: 资金不足，无法买入 {股票代码}")
            return
        
        # 计算交易成本
        交易金额 = 数量 * 实际价格
        手续费 = 交易金额 * self.手续费率
        总成本 = 交易金额 + 手续费
        
        # 检查资金是否足够
        if 总成本 > self.资金:
            数量 = int((self.资金 / (实际价格 * (1 + self.手续费率))))
            交易金额 = 数量 * 实际价格
            手续费 = 交易金额 * self.手续费率
            总成本 = 交易金额 + 手续费
        
        if 数量 <= 0:
            logger.warning(f"{日期}: 资金不足，无法买入 {股票代码}")
            return
        
        # 更新持仓和资金
        if 股票代码 in self.持仓:
            # 计算新的平均成本
            原持仓 = self.持仓[股票代码]
            原成本 = self.持仓成本[股票代码]
            新持仓 = 原持仓 + 数量
            新成本 = (原持仓 * 原成本 + 交易金额) / 新持仓
            self.持仓[股票代码] = 新持仓
            self.持仓成本[股票代码] = 新成本
        else:
            self.持仓[股票代码] = 数量
            self.持仓成本[股票代码] = 实际价格
        
        self.资金 -= 总成本
        
        # 记录交易
        交易记录 = {
            '日期': 日期,
            '类型': '买入',
            '股票代码': 股票代码,
            '价格': 实际价格,
            '数量': 数量,
            '交易金额': 交易金额,
            '手续费': 手续费,
            '剩余资金': self.资金,
            '信号解释': 信号行.get('信号解释', '')
        }
        self.交易记录.append(交易记录)
        
        logger.debug(f"{日期}: 买入 {股票代码} {数量}股，价格 {实际价格:.2f}，总成本 {总成本:.2f}")
    
    def _执行卖出(self, 股票代码: str, 价格: float, 日期: datetime.datetime, 信号行: pd.Series):
        """执行卖出操作。"""
        if 股票代码 not in self.持仓 or self.持仓[股票代码] <= 0:
            logger.warning(f"{日期}: 没有持仓，无法卖出 {股票代码}")
            return
        
        # 考虑滑点
        实际价格 = 价格 * (1 - self.滑点)
        
        # 卖出全部持仓
        数量 = self.持仓[股票代码]
        交易金额 = 数量 * 实际价格
        手续费 = 交易金额 * self.手续费率
        净收入 = 交易金额 - 手续费
        
        # 计算盈亏
        成本 = self.持仓成本[股票代码] * 数量
        盈亏 = 净收入 - 成本
        
        # 更新持仓和资金
        self.持仓[股票代码] = 0
        self.资金 += 净收入
        
        # 记录交易
        交易记录 = {
            '日期': 日期,
            '类型': '卖出',
            '股票代码': 股票代码,
            '价格': 实际价格,
            '数量': 数量,
            '交易金额': 交易金额,
            '手续费': 手续费,
            '净收入': 净收入,
            '盈亏': 盈亏,
            '剩余资金': self.资金,
            '信号解释': 信号行.get('信号解释', '')
        }
        self.交易记录.append(交易记录)
        
        logger.debug(f"{日期}: 卖出 {股票代码} {数量}股，价格 {实际价格:.2f}，净收入 {净收入:.2f}，盈亏 {盈亏:.2f}")
    
    def _计算总资产(self, 股票代码: str, 当前价格: float) -> float:
        """计算当前总资产。"""
        持仓价值 = 0.0
        if 股票代码 in self.持仓:
            持仓价值 = self.持仓[股票代码] * 当前价格
        
        return self.资金 + 持仓价值
    
    def _计算回测指标(self) -> Dict[str, Any]:
        """计算回测结果指标。"""
        # 转换资金曲线为DataFrame
        资金曲线 = pd.DataFrame(self.资金曲线)
        资金曲线['日期'] = pd.to_datetime(资金曲线['日期'])
        资金曲线 = 资金曲线.set_index('日期')
        
        # 计算基本指标
        初始资产 = 资金曲线['资产'].iloc[0]
        最终资产 = 资金曲线['资产'].iloc[-1]
        收益率 = (最终资产 / 初始资产) - 1
        
        # 计算年化收益率
        开始日期 = 资金曲线.index[0]
        结束日期 = 资金曲线.index[-1]
        交易天数 = (结束日期 - 开始日期).days
        if 交易天数 > 0:
            年化收益率 = (1 + 收益率) ** (365 / 交易天数) - 1
        else:
            年化收益率 = 0
        
        # 计算最大回撤
        资金曲线['累计最大值'] = 资金曲线['资产'].cummax()
        资金曲线['回撤'] = (资金曲线['资产'] - 资金曲线['累计最大值']) / 资金曲线['累计最大值']
        最大回撤 = abs(资金曲线['回撤'].min())
        
        # 计算收益回撤比
        收益回撤比 = 年化收益率 / 最大回撤 if 最大回撤 > 0 else float('inf')
        
        # 计算夏普比率（假设无风险利率为0）
        日收益率 = 资金曲线['资产'].pct_change().dropna()
        if len(日收益率) > 1:
            夏普比率 = np.sqrt(252) * 日收益率.mean() / 日收益率.std() if 日收益率.std() > 0 else 0
        else:
            夏普比率 = 0
        
        # 计算交易统计
        交易次数 = len(self.交易记录)
        if 交易次数 > 0:
            盈利交易 = [交易 for 交易 in self.交易记录 if 交易.get('类型') == '卖出' and 交易.get('盈亏', 0) > 0]
            亏损交易 = [交易 for 交易 in self.交易记录 if 交易.get('类型') == '卖出' and 交易.get('盈亏', 0) <= 0]
            
            盈利次数 = len(盈利交易)
            亏损次数 = len(亏损交易)
            
            胜率 = 盈利次数 / (盈利次数 + 亏损次数) if (盈利次数 + 亏损次数) > 0 else 0
            
            平均盈利 = sum(交易['盈亏'] for 交易 in 盈利交易) / 盈利次数 if 盈利次数 > 0 else 0
            平均亏损 = sum(交易['盈亏'] for 交易 in 亏损交易) / 亏损次数 if 亏损次数 > 0 else 0
            
            盈亏比 = abs(平均盈利 / 平均亏损) if 平均亏损 != 0 else float('inf')
        else:
            胜率 = 0
            盈亏比 = 0
            盈利次数 = 0
            亏损次数 = 0
        
        # 汇总结果
        结果 = {
            '初始资产': 初始资产,
            '最终资产': 最终资产,
            '收益率': 收益率,
            '年化收益率': 年化收益率,
            '最大回撤': 最大回撤,
            '收益回撤比': 收益回撤比,
            '夏普比率': 夏普比率,
            '交易次数': 交易次数,
            '盈利次数': 盈利次数,
            '亏损次数': 亏损次数,
            '胜率': 胜率,
            '盈亏比': 盈亏比,
            '资金曲线': 资金曲线,
            '交易记录': self.交易记录
        }
        
        return 结果


class 参数优化器:
    """
    参数优化器类，用于自动寻找策略的最优参数。
    支持多种优化方法和目标函数。
    """
    
    def __init__(self, 
               回测引擎: 回测引擎,
               并行进程数: int = None):
        """
        初始化参数优化器。
        
        参数:
            回测引擎: 用于执行回测的回测引擎实例
            并行进程数: 并行优化使用的进程数，默认为CPU核心数的一半
        """
        self.回测引擎 = 回测引擎
        self.并行进程数 = 并行进程数 if 并行进程数 is not None else max(1, multiprocessing.cpu_count() // 2)
        logger.info(f"参数优化器初始化，并行进程数: {self.并行进程数}")
    
    def 网格搜索(self, 
              策略: 策略,
              市场数据: pd.DataFrame,
              庄家数据: pd.DataFrame,
              股票代码: str,
              参数网格: Dict[str, List[Any]],
              目标指标: str = '收益回撤比',
              并行: bool = True) -> List[Dict[str, Any]]:
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
        logger.info(f"开始网格搜索优化，参数空间大小: {self._计算参数空间大小(参数网格)}")
        
        # 生成所有参数组合
        参数名列表 = list(参数网格.keys())
        参数值列表 = list(参数网格.values())
        参数组合列表 = list(itertools.product(*参数值列表))
        
        # 准备优化任务
        任务列表 = []
        for 组合 in 参数组合列表:
            参数字典 = {参数名列表[i]: 组合[i] for i in range(len(参数名列表))}
            任务列表.append((策略, 市场数据, 庄家数据, 股票代码, 参数字典, 目标指标))
        
        # 执行优化
        if 并行 and len(任务列表) > 1:
            with multiprocessing.Pool(self.并行进程数) as 进程池:
                结果列表 = 进程池.map(self._执行单次回测, 任务列表)
        else:
            结果列表 = [self._执行单次回测(任务) for 任务 in 任务列表]
        
        # 按目标指标排序
        结果列表.sort(key=lambda x: x['回测结果'][目标指标], reverse=True)
        
        logger.info(f"网格搜索优化完成，最佳{目标指标}: {结果列表[0]['回测结果'][目标指标]:.4f}")
        return 结果列表
    
    def 随机搜索(self, 
              策略: 策略,
              市场数据: pd.DataFrame,
              庄家数据: pd.DataFrame,
              股票代码: str,
              参数空间: Dict[str, Tuple[Any, Any]],
              迭代次数: int = 100,
              目标指标: str = '收益回撤比',
              并行: bool = True) -> List[Dict[str, Any]]:
        """
        使用随机搜索方法优化策略参数。
        
        参数:
            策略: 要优化的策略实例
            市场数据: 市场数据DataFrame
            庄家数据: 庄家指标数据DataFrame
            股票代码: 交易标的代码
            参数空间: 参数名到(最小值,最大值)元组的映射
            迭代次数: 随机搜索的迭代次数
            目标指标: 优化目标指标，默认为'收益回撤比'
            并行: 是否使用并行计算
            
        返回:
            按目标指标排序的参数组合和结果列表
        """
        logger.info(f"开始随机搜索优化，迭代次数: {迭代次数}")
        
        # 生成随机参数组合
        任务列表 = []
        for _ in range(迭代次数):
            参数字典 = {}
            for 参数名, (最小值, 最大值) in 参数空间.items():
                if isinstance(最小值, int) and isinstance(最大值, int):
                    参数字典[参数名] = np.random.randint(最小值, 最大值 + 1)
                elif isinstance(最小值, float) or isinstance(最大值, float):
                    参数字典[参数名] = np.random.uniform(最小值, 最大值)
                else:
                    raise ValueError(f"不支持的参数类型: {type(最小值)}")
            
            任务列表.append((策略, 市场数据, 庄家数据, 股票代码, 参数字典, 目标指标))
        
        # 执行优化
        if 并行 and len(任务列表) > 1:
            with multiprocessing.Pool(self.并行进程数) as 进程池:
                结果列表 = 进程池.map(self._执行单次回测, 任务列表)
        else:
            结果列表 = [self._执行单次回测(任务) for 任务 in 任务列表]
        
        # 按目标指标排序
        结果列表.sort(key=lambda x: x['回测结果'][目标指标], reverse=True)
        
        logger.info(f"随机搜索优化完成，最佳{目标指标}: {结果列表[0]['回测结果'][目标指标]:.4f}")
        return 结果列表
    
    def 贝叶斯优化(self, 
               策略: 策略,
               市场数据: pd.DataFrame,
               庄家数据: pd.DataFrame,
               股票代码: str,
               参数空间: Dict[str, Tuple[Any, Any]],
               迭代次数: int = 50,
               初始点数: int = 10,
               目标指标: str = '收益回撤比') -> List[Dict[str, Any]]:
        """
        使用贝叶斯优化方法优化策略参数。
        
        参数:
            策略: 要优化的策略实例
            市场数据: 市场数据DataFrame
            庄家数据: 庄家指标数据DataFrame
            股票代码: 交易标的代码
            参数空间: 参数名到(最小值,最大值)元组的映射
            迭代次数: 贝叶斯优化的迭代次数
            初始点数: 初始随机点的数量
            目标指标: 优化目标指标，默认为'收益回撤比'
            
        返回:
            按目标指标排序的参数组合和结果列表
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer
            from skopt.utils import use_named_args
        except ImportError:
            logger.error("无法导入scikit-optimize，请安装: pip install scikit-optimize")
            raise
        
        logger.info(f"开始贝叶斯优化，迭代次数: {迭代次数}")
        
        # 定义参数空间
        空间 = []
        参数名列表 = []
        
        for 参数名, (最小值, 最大值) in 参数空间.items():
            参数名列表.append(参数名)
            if isinstance(最小值, int) and isinstance(最大值, int):
                空间.append(Integer(最小值, 最大值, name=参数名))
            elif isinstance(最小值, float) or isinstance(最大值, float):
                空间.append(Real(最小值, 最大值, name=参数名))
            else:
                raise ValueError(f"不支持的参数类型: {type(最小值)}")
        
        # 定义目标函数
        @use_named_args(空间)
        def 目标函数(**参数):
            # 设置策略参数
            策略.设置参数(参数)
            
            # 生成信号
            信号数据 = 策略.生成信号(市场数据, 庄家数据)
            
            # 执行回测
            回测结果 = self.回测引擎.执行回测(市场数据, 信号数据, 股票代码)
            
            # 返回负的目标指标值（因为gp_minimize是最小化函数）
            return -回测结果[目标指标]
        
        # 执行贝叶斯优化
        结果 = gp_minimize(
            目标函数,
            空间,
            n_calls=迭代次数,
            n_random_starts=初始点数,
            random_state=42
        )
        
        # 提取结果
        最佳参数 = {参数名列表[i]: 结果.x[i] for i in range(len(参数名列表))}
        
        # 使用最佳参数执行一次回测以获取完整结果
        策略.设置参数(最佳参数)
        信号数据 = 策略.生成信号(市场数据, 庄家数据)
        回测结果 = self.回测引擎.执行回测(市场数据, 信号数据, 股票代码)
        
        结果列表 = [{
            '参数': 最佳参数,
            '回测结果': 回测结果
        }]
        
        logger.info(f"贝叶斯优化完成，最佳{目标指标}: {回测结果[目标指标]:.4f}")
        return 结果列表
    
    def 遗传算法优化(self, 
                策略: 策略,
                市场数据: pd.DataFrame,
                庄家数据: pd.DataFrame,
                股票代码: str,
                参数空间: Dict[str, Tuple[Any, Any]],
                种群大小: int = 50,
                代数: int = 20,
                变异率: float = 0.1,
                目标指标: str = '收益回撤比',
                并行: bool = True) -> List[Dict[str, Any]]:
        """
        使用遗传算法优化策略参数。
        
        参数:
            策略: 要优化的策略实例
            市场数据: 市场数据DataFrame
            庄家数据: 庄家指标数据DataFrame
            股票代码: 交易标的代码
            参数空间: 参数名到(最小值,最大值)元组的映射
            种群大小: 遗传算法的种群大小
            代数: 遗传算法的代数
            变异率: 基因变异的概率
            目标指标: 优化目标指标，默认为'收益回撤比'
            并行: 是否使用并行计算
            
        返回:
            按目标指标排序的参数组合和结果列表
        """
        logger.info(f"开始遗传算法优化，种群大小: {种群大小}，代数: {代数}")
        
        # 参数名列表
        参数名列表 = list(参数空间.keys())
        
        # 生成初始种群
        种群 = []
        for _ in range(种群大小):
            个体 = {}
            for 参数名, (最小值, 最大值) in 参数空间.items():
                if isinstance(最小值, int) and isinstance(最大值, int):
                    个体[参数名] = np.random.randint(最小值, 最大值 + 1)
                elif isinstance(最小值, float) or isinstance(最大值, float):
                    个体[参数名] = np.random.uniform(最小值, 最大值)
                else:
                    raise ValueError(f"不支持的参数类型: {type(最小值)}")
            种群.append(个体)
        
        # 所有结果的列表
        所有结果 = []
        
        # 遗传算法主循环
        for 代 in range(代数):
            logger.info(f"遗传算法第 {代+1}/{代数} 代")
            
            # 评估当前种群
            任务列表 = [(策略, 市场数据, 庄家数据, 股票代码, 个体, 目标指标) for 个体 in 种群]
            
            if 并行 and len(任务列表) > 1:
                with multiprocessing.Pool(self.并行进程数) as 进程池:
                    结果列表 = 进程池.map(self._执行单次回测, 任务列表)
            else:
                结果列表 = [self._执行单次回测(任务) for 任务 in 任务列表]
            
            # 添加到所有结果
            所有结果.extend(结果列表)
            
            # 按适应度（目标指标）排序
            结果列表.sort(key=lambda x: x['回测结果'][目标指标], reverse=True)
            
            # 如果是最后一代，直接返回结果
            if 代 == 代数 - 1:
                所有结果.sort(key=lambda x: x['回测结果'][目标指标], reverse=True)
                return 所有结果
            
            # 选择精英个体
            精英数量 = max(1, 种群大小 // 10)
            新种群 = [结果['参数'] for 结果 in 结果列表[:精英数量]]
            
            # 通过锦标赛选择和交叉生成新个体
            while len(新种群) < 种群大小:
                # 锦标赛选择父母
                父亲索引 = np.random.randint(0, len(结果列表))
                母亲索引 = np.random.randint(0, len(结果列表))
                for _ in range(3):  # 锦标赛大小为3
                    候选索引 = np.random.randint(0, len(结果列表))
                    if 结果列表[候选索引]['回测结果'][目标指标] > 结果列表[父亲索引]['回测结果'][目标指标]:
                        父亲索引 = 候选索引
                
                for _ in range(3):
                    候选索引 = np.random.randint(0, len(结果列表))
                    if 结果列表[候选索引]['回测结果'][目标指标] > 结果列表[母亲索引]['回测结果'][目标指标]:
                        母亲索引 = 候选索引
                
                父亲 = 结果列表[父亲索引]['参数']
                母亲 = 结果列表[母亲索引]['参数']
                
                # 交叉
                子代 = {}
                for 参数名 in 参数名列表:
                    # 均匀交叉
                    if np.random.random() < 0.5:
                        子代[参数名] = 父亲[参数名]
                    else:
                        子代[参数名] = 母亲[参数名]
                    
                    # 变异
                    if np.random.random() < 变异率:
                        最小值, 最大值 = 参数空间[参数名]
                        if isinstance(最小值, int) and isinstance(最大值, int):
                            子代[参数名] = np.random.randint(最小值, 最大值 + 1)
                        else:
                            子代[参数名] = np.random.uniform(最小值, 最大值)
                
                新种群.append(子代)
            
            # 更新种群
            种群 = 新种群
        
        # 这里不应该到达，但为了完整性
        所有结果.sort(key=lambda x: x['回测结果'][目标指标], reverse=True)
        return 所有结果
    
    def 粒子群优化(self, 
               策略: 策略,
               市场数据: pd.DataFrame,
               庄家数据: pd.DataFrame,
               股票代码: str,
               参数空间: Dict[str, Tuple[Any, Any]],
               粒子数: int = 30,
               迭代次数: int = 20,
               目标指标: str = '收益回撤比',
               并行: bool = True) -> List[Dict[str, Any]]:
        """
        使用粒子群优化方法优化策略参数。
        
        参数:
            策略: 要优化的策略实例
            市场数据: 市场数据DataFrame
            庄家数据: 庄家指标数据DataFrame
            股票代码: 交易标的代码
            参数空间: 参数名到(最小值,最大值)元组的映射
            粒子数: 粒子群中的粒子数量
            迭代次数: 迭代次数
            目标指标: 优化目标指标，默认为'收益回撤比'
            并行: 是否使用并行计算
            
        返回:
            按目标指标排序的参数组合和结果列表
        """
        logger.info(f"开始粒子群优化，粒子数: {粒子数}，迭代次数: {迭代次数}")
        
        # 参数名和范围
        参数名列表 = list(参数空间.keys())
        参数最小值 = np.array([参数空间[参数名][0] for 参数名 in 参数名列表])
        参数最大值 = np.array([参数空间[参数名][1] for 参数名 in 参数名列表])
        参数范围 = 参数最大值 - 参数最小值
        
        # 参数类型（整数或浮点数）
        参数类型 = [
            'int' if isinstance(参数空间[参数名][0], int) and isinstance(参数空间[参数名][1], int) else 'float'
            for 参数名 in 参数名列表
        ]
        
        # 初始化粒子位置和速度
        位置 = np.random.rand(粒子数, len(参数名列表)) * 参数范围 + 参数最小值
        速度 = np.random.randn(粒子数, len(参数名列表)) * 0.1 * 参数范围
        
        # 对整数参数取整
        for i, 类型 in enumerate(参数类型):
            if 类型 == 'int':
                位置[:, i] = np.round(位置[:, i])
        
        # 初始化个体最优和全局最优
        个体最优位置 = 位置.copy()
        个体最优适应度 = np.zeros(粒子数) - float('inf')
        全局最优位置 = 位置[0].copy()
        全局最优适应度 = -float('inf')
        
        # 所有结果的列表
        所有结果 = []
        
        # PSO参数
        w = 0.7  # 惯性权重
        c1 = 1.5  # 认知参数
        c2 = 1.5  # 社会参数
        
        # 主循环
        for 迭代 in range(迭代次数):
            logger.info(f"粒子群优化第 {迭代+1}/{迭代次数} 次迭代")
            
            # 将位置转换为参数字典列表
            参数字典列表 = []
            for 粒子位置 in 位置:
                参数字典 = {}
                for i, 参数名 in enumerate(参数名列表):
                    值 = 粒子位置[i]
                    if 参数类型[i] == 'int':
                        值 = int(round(值))
                    参数字典[参数名] = 值
                参数字典列表.append(参数字典)
            
            # 评估粒子
            任务列表 = [(策略, 市场数据, 庄家数据, 股票代码, 参数字典, 目标指标) for 参数字典 in 参数字典列表]
            
            if 并行 and len(任务列表) > 1:
                with multiprocessing.Pool(self.并行进程数) as 进程池:
                    结果列表 = 进程池.map(self._执行单次回测, 任务列表)
            else:
                结果列表 = [self._执行单次回测(任务) for 任务 in 任务列表]
            
            # 添加到所有结果
            所有结果.extend(结果列表)
            
            # 更新个体最优和全局最优
            for i, 结果 in enumerate(结果列表):
                适应度 = 结果['回测结果'][目标指标]
                
                if 适应度 > 个体最优适应度[i]:
                    个体最优适应度[i] = 适应度
                    个体最优位置[i] = 位置[i].copy()
                
                if 适应度 > 全局最优适应度:
                    全局最优适应度 = 适应度
                    全局最优位置 = 位置[i].copy()
            
            # 更新速度和位置
            r1 = np.random.rand(粒子数, len(参数名列表))
            r2 = np.random.rand(粒子数, len(参数名列表))
            
            速度 = (w * 速度 + 
                  c1 * r1 * (个体最优位置 - 位置) + 
                  c2 * r2 * (全局最优位置 - 位置))
            
            位置 = 位置 + 速度
            
            # 确保位置在边界内
            位置 = np.maximum(位置, 参数最小值)
            位置 = np.minimum(位置, 参数最大值)
            
            # 对整数参数取整
            for i, 类型 in enumerate(参数类型):
                if 类型 == 'int':
                    位置[:, i] = np.round(位置[:, i])
        
        # 返回排序后的结果
        所有结果.sort(key=lambda x: x['回测结果'][目标指标], reverse=True)
        return 所有结果
    
    def 模拟退火优化(self, 
                策略: 策略,
                市场数据: pd.DataFrame,
                庄家数据: pd.DataFrame,
                股票代码: str,
                参数空间: Dict[str, Tuple[Any, Any]],
                初始温度: float = 100.0,
                冷却率: float = 0.95,
                迭代次数: int = 100,
                目标指标: str = '收益回撤比') -> List[Dict[str, Any]]:
        """
        使用模拟退火算法优化策略参数。
        
        参数:
            策略: 要优化的策略实例
            市场数据: 市场数据DataFrame
            庄家数据: 庄家指标数据DataFrame
            股票代码: 交易标的代码
            参数空间: 参数名到(最小值,最大值)元组的映射
            初始温度: 初始温度
            冷却率: 温度冷却率
            迭代次数: 每个温度的迭代次数
            目标指标: 优化目标指标，默认为'收益回撤比'
            
        返回:
            按目标指标排序的参数组合和结果列表
        """
        logger.info(f"开始模拟退火优化，初始温度: {初始温度}，冷却率: {冷却率}")
        
        # 参数名列表
        参数名列表 = list(参数空间.keys())
        
        # 生成初始解
        当前解 = {}
        for 参数名, (最小值, 最大值) in 参数空间.items():
            if isinstance(最小值, int) and isinstance(最大值, int):
                当前解[参数名] = np.random.randint(最小值, 最大值 + 1)
            elif isinstance(最小值, float) or isinstance(最大值, float):
                当前解[参数名] = np.random.uniform(最小值, 最大值)
            else:
                raise ValueError(f"不支持的参数类型: {type(最小值)}")
        
        # 评估初始解
        策略.设置参数(当前解)
        信号数据 = 策略.生成信号(市场数据, 庄家数据)
        当前结果 = self.回测引擎.执行回测(市场数据, 信号数据, 股票代码)
        当前适应度 = 当前结果[目标指标]
        
        # 初始化最优解
        最优解 = 当前解.copy()
        最优结果 = 当前结果
        最优适应度 = 当前适应度
        
        # 所有结果的列表
        所有结果 = [{
            '参数': 当前解.copy(),
            '回测结果': 当前结果
        }]
        
        # 当前温度
        温度 = 初始温度
        
        # 模拟退火主循环
        while 温度 > 0.1:
            for _ in range(迭代次数):
                # 生成邻居解
                邻居解 = 当前解.copy()
                
                # 随机选择一个参数进行扰动
                参数名 = np.random.choice(参数名列表)
                最小值, 最大值 = 参数空间[参数名]
                
                # 扰动大小随温度变化
                扰动比例 = 温度 / 初始温度 * 0.3  # 最大扰动为范围的30%
                
                if isinstance(最小值, int) and isinstance(最大值, int):
                    扰动大小 = max(1, int((最大值 - 最小值) * 扰动比例))
                    邻居解[参数名] = int(邻居解[参数名] + np.random.randint(-扰动大小, 扰动大小 + 1))
                    邻居解[参数名] = max(最小值, min(最大值, 邻居解[参数名]))
                else:
                    扰动大小 = (最大值 - 最小值) * 扰动比例
                    邻居解[参数名] = 邻居解[参数名] + np.random.uniform(-扰动大小, 扰动大小)
                    邻居解[参数名] = max(最小值, min(最大值, 邻居解[参数名]))
                
                # 评估邻居解
                策略.设置参数(邻居解)
                信号数据 = 策略.生成信号(市场数据, 庄家数据)
                邻居结果 = self.回测引擎.执行回测(市场数据, 信号数据, 股票代码)
                邻居适应度 = 邻居结果[目标指标]
                
                # 记录结果
                所有结果.append({
                    '参数': 邻居解.copy(),
                    '回测结果': 邻居结果
                })
                
                # 决定是否接受新解
                if 邻居适应度 > 当前适应度:
                    # 如果新解更好，总是接受
                    当前解 = 邻居解.copy()
                    当前结果 = 邻居结果
                    当前适应度 = 邻居适应度
                    
                    # 更新最优解
                    if 邻居适应度 > 最优适应度:
                        最优解 = 邻居解.copy()
                        最优结果 = 邻居结果
                        最优适应度 = 邻居适应度
                else:
                    # 如果新解更差，以一定概率接受
                    接受概率 = np.exp((邻居适应度 - 当前适应度) / 温度)
                    if np.random.random() < 接受概率:
                        当前解 = 邻居解.copy()
                        当前结果 = 邻居结果
                        当前适应度 = 邻居适应度
            
            # 降低温度
            温度 *= 冷却率
            logger.info(f"温度降低到 {温度:.4f}，当前最优{目标指标}: {最优适应度:.4f}")
        
        # 返回排序后的结果
        所有结果.sort(key=lambda x: x['回测结果'][目标指标], reverse=True)
        return 所有结果
    
    def 多目标优化(self, 
               策略: 策略,
               市场数据: pd.DataFrame,
               庄家数据: pd.DataFrame,
               股票代码: str,
               参数网格: Dict[str, List[Any]],
               目标指标列表: List[str] = ['收益回撤比', '夏普比率'],
               并行: bool = True) -> List[Dict[str, Any]]:
        """
        执行多目标优化，找到Pareto最优解集。
        
        参数:
            策略: 要优化的策略实例
            市场数据: 市场数据DataFrame
            庄家数据: 庄家指标数据DataFrame
            股票代码: 交易标的代码
            参数网格: 参数名到可能值列表的映射
            目标指标列表: 要优化的目标指标列表
            并行: 是否使用并行计算
            
        返回:
            Pareto最优解集
        """
        logger.info(f"开始多目标优化，目标指标: {目标指标列表}")
        
        # 生成所有参数组合
        参数名列表 = list(参数网格.keys())
        参数值列表 = list(参数网格.values())
        参数组合列表 = list(itertools.product(*参数值列表))
        
        # 准备优化任务
        任务列表 = []
        for 组合 in 参数组合列表:
            参数字典 = {参数名列表[i]: 组合[i] for i in range(len(参数名列表))}
            # 使用第一个目标指标作为任务的目标指标（实际上会计算所有指标）
            任务列表.append((策略, 市场数据, 庄家数据, 股票代码, 参数字典, 目标指标列表[0]))
        
        # 执行优化
        if 并行 and len(任务列表) > 1:
            with multiprocessing.Pool(self.并行进程数) as 进程池:
                结果列表 = 进程池.map(self._执行单次回测, 任务列表)
        else:
            结果列表 = [self._执行单次回测(任务) for 任务 in 任务列表]
        
        # 找到Pareto最优解集
        pareto集 = self._找到Pareto最优解(结果列表, 目标指标列表)
        
        logger.info(f"多目标优化完成，找到 {len(pareto集)} 个Pareto最优解")
        return pareto集
    
    def _找到Pareto最优解(self, 结果列表: List[Dict[str, Any]], 目标指标列表: List[str]) -> List[Dict[str, Any]]:
        """找到Pareto最优解集。"""
        pareto集 = []
        
        for i, 结果 in enumerate(结果列表):
            被支配 = False
            
            for j, 其他结果 in enumerate(结果列表):
                if i == j:
                    continue
                
                # 检查是否被支配
                支配 = True
                至少一个更好 = False
                
                for 指标 in 目标指标列表:
                    if 结果['回测结果'][指标] > 其他结果['回测结果'][指标]:
                        支配 = False
                        break
                    elif 结果['回测结果'][指标] < 其他结果['回测结果'][指标]:
                        至少一个更好 = True
                
                if 支配 and 至少一个更好:
                    被支配 = True
                    break
            
            if not 被支配:
                pareto集.append(结果)
        
        return pareto集
    
    def _执行单次回测(self, 任务: Tuple) -> Dict[str, Any]:
        """
        执行单次回测任务。
        
        参数:
            任务: (策略, 市场数据, 庄家数据, 股票代码, 参数字典, 目标指标)的元组
            
        返回:
            包含参数和回测结果的字典
        """
        策略实例, 市场数据, 庄家数据, 股票代码, 参数字典, _ = 任务
        
        # 设置策略参数
        策略实例.设置参数(参数字典)
        
        # 生成信号
        信号数据 = 策略实例.生成信号(市场数据, 庄家数据)
        
        # 执行回测
        回测结果 = self.回测引擎.执行回测(市场数据, 信号数据, 股票代码)
        
        return {
            '参数': 参数字典,
            '回测结果': 回测结果
        }
    
    def _计算参数空间大小(self, 参数网格: Dict[str, List[Any]]) -> int:
        """计算参数空间的大小。"""
        大小 = 1
        for 值列表 in 参数网格.values():
            大小 *= len(值列表)
        return 大小


class 一键回测器:
    """
    一键回测工具类，提供简化的回测流程。
    自动获取数据、清洗数据、执行回测和参数优化。
    """
    
    def __init__(self, 
               数据源工厂,
               策略工厂,
               回测引擎: 回测引擎 = None,
               参数优化器: 参数优化器 = None):
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
    
    def 一键回测(self, 
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
            市场数据 = 数据清洗器.一键清洗(市场数据, 数据清洗配置)
            庄家数据 = 数据清洗器.一键清洗(庄家数据, 数据清洗配置)
        
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
    
    def 一键参数优化(self, 
                股票代码: str,
                策略名称: str,
                开始日期: str,
                结束日期: str,
                参数网格: Dict[str, List[Any]],
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
            参数网格: 参数名到可能值列表的映射（网格搜索）或参数名到(最小值,最大值)的映射（其他方法）
            优化方法: 优化方法，可选值包括：
                      '网格搜索', '随机搜索', '贝叶斯优化', '遗传算法', '粒子群优化', '模拟退火', '多目标优化'
            目标指标: 优化目标指标，默认为'收益回撤比'
            市场类型: 市场类型，如'a股'或'虚拟货币'
            数据清洗配置: 数据清洗配置，如果为None则使用默认配置
            优化参数: 优化方法的特定参数
            
        返回:
            包含优化结果的字典
        """
        logger.info(f"开始一键参数优化，股票代码: {股票代码}，策略: {策略名称}，优化方法: {优化方法}")
        
        # 1. 获取数据
        数据源 = self.数据源工厂.创建数据源(市场类型)
        
        市场数据 = 数据源.获取市场数据(股票代码, 开始日期, 结束日期)
        庄家数据 = 数据源.获取庄家指标(股票代码, 开始日期, 结束日期)
        
        # 2. 清洗数据
        if 数据清洗配置 is not None:
            市场数据 = 数据清洗器.一键清洗(市场数据, 数据清洗配置)
            庄家数据 = 数据清洗器.一键清洗(庄家数据, 数据清洗配置)
        
        # 3. 创建策略
        策略 = self.策略工厂.创建策略(策略名称)
        
        # 4. 执行参数优化
        优化参数 = 优化参数 if 优化参数 is not None else {}
        
        if 优化方法 == '网格搜索':
            优化结果 = self.参数优化器.网格搜索(
                策略, 市场数据, 庄家数据, 股票代码, 参数网格, 目标指标, **优化参数
            )
        elif 优化方法 == '随机搜索':
            优化结果 = self.参数优化器.随机搜索(
                策略, 市场数据, 庄家数据, 股票代码, 参数网格, **优化参数
            )
        elif 优化方法 == '贝叶斯优化':
            优化结果 = self.参数优化器.贝叶斯优化(
                策略, 市场数据, 庄家数据, 股票代码, 参数网格, **优化参数
            )
        elif 优化方法 == '遗传算法':
            优化结果 = self.参数优化器.遗传算法优化(
                策略, 市场数据, 庄家数据, 股票代码, 参数网格, **优化参数
            )
        elif 优化方法 == '粒子群优化':
            优化结果 = self.参数优化器.粒子群优化(
                策略, 市场数据, 庄家数据, 股票代码, 参数网格, **优化参数
            )
        elif 优化方法 == '模拟退火':
            优化结果 = self.参数优化器.模拟退火优化(
                策略, 市场数据, 庄家数据, 股票代码, 参数网格, **优化参数
            )
        elif 优化方法 == '多目标优化':
            目标指标列表 = 优化参数.get('目标指标列表', ['收益回撤比', '夏普比率'])
            优化结果 = self.参数优化器.多目标优化(
                策略, 市场数据, 庄家数据, 股票代码, 参数网格, 目标指标列表, **优化参数
            )
        else:
            raise ValueError(f"未知的优化方法: {优化方法}")
        
        # 5. 生成优化报告
        优化报告 = self._生成优化报告(优化结果, 市场数据, 股票代码, 策略名称, 优化方法, 目标指标)
        
        logger.info(f"一键参数优化完成，找到 {len(优化结果)} 个结果")
        return 优化报告
    
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
                '盈利次数': 回测结果['盈利次数'],
                '亏损次数': 回测结果['亏损次数'],
                '胜率': 回测结果['胜率'],
                '盈亏比': 回测结果['盈亏比']
            },
            '交易记录': 回测结果['交易记录'],
            '资金曲线': 回测结果['资金曲线'].to_dict('records')
        }
        
        # 生成图表
        图表路径 = self._生成回测图表(回测结果, 市场数据, 信号数据, 股票代码)
        报告['图表路径'] = 图表路径
        
        return 报告
    
    def _生成优化报告(self, 
                 优化结果: List[Dict[str, Any]],
                 市场数据: pd.DataFrame,
                 股票代码: str,
                 策略名称: str,
                 优化方法: str,
                 目标指标: str) -> Dict[str, Any]:
        """生成详细的参数优化报告。"""
        # 基本信息
        报告 = {
            '基本信息': {
                '股票代码': 股票代码,
                '策略名称': 策略名称,
                '优化方法': 优化方法,
                '目标指标': 目标指标,
                '回测开始日期': 市场数据.index[0].strftime('%Y-%m-%d'),
                '回测结束日期': 市场数据.index[-1].strftime('%Y-%m-%d'),
                '回测天数': len(市场数据),
                '优化结果数量': len(优化结果)
            },
            '最佳结果': 优化结果[0] if 优化结果 else None,
            '所有结果': 优化结果
        }
        
        # 生成参数分布图表
        图表路径 = self._生成参数分布图表(优化结果, 目标指标)
        报告['图表路径'] = 图表路径
        
        return 报告
    
    def _生成回测图表(self, 
                 回测结果: Dict[str, Any],
                 市场数据: pd.DataFrame,
                 信号数据: pd.DataFrame,
                 股票代码: str) -> str:
        """生成回测结果图表。"""
        # 创建图表目录
        图表目录 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'charts')
        os.makedirs(图表目录, exist_ok=True)
        
        # 生成文件名
        时间戳 = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        文件名 = f"{股票代码}_回测_{时间戳}.png"
        文件路径 = os.path.join(图表目录, 文件名)
        
        # 创建图表
        plt.figure(figsize=(12, 10))
        
        # 1. 资金曲线
        plt.subplot(3, 1, 1)
        资金曲线 = 回测结果['资金曲线']
        plt.plot(资金曲线.index, 资金曲线['资产'], label='总资产')
        plt.plot(资金曲线.index, 资金曲线['现金'], label='现金', alpha=0.7)
        plt.plot(资金曲线.index, 资金曲线['持仓'], label='持仓价值', alpha=0.7)
        
        # 添加买入卖出标记
        for 交易 in 回测结果['交易记录']:
            if 交易['类型'] == '买入':
                plt.scatter(交易['日期'], 交易['剩余资金'] + 交易['交易金额'], 
                         marker='^', color='red', s=100)
            elif 交易['类型'] == '卖出':
                plt.scatter(交易['日期'], 交易['剩余资金'], 
                         marker='v', color='green', s=100)
        
        plt.title(f"{股票代码} 回测资金曲线")
        plt.xlabel('日期')
        plt.ylabel('资产')
        plt.legend()
        plt.grid(True)
        
        # 2. 价格和信号
        plt.subplot(3, 1, 2)
        plt.plot(市场数据.index, 市场数据['close'], label='收盘价')
        
        # 添加买入卖出信号
        买入点 = 信号数据[信号数据['信号'] == 1].index
        卖出点 = 信号数据[信号数据['信号'] == -1].index
        
        plt.scatter(买入点, 市场数据.loc[买入点]['close'], 
                 marker='^', color='red', s=100, label='买入信号')
        plt.scatter(卖出点, 市场数据.loc[卖出点]['close'], 
                 marker='v', color='green', s=100, label='卖出信号')
        
        plt.title(f"{股票代码} 价格和交易信号")
        plt.xlabel('日期')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True)
        
        # 3. 操纵分数
        if '操纵分数' in 信号数据.columns:
            plt.subplot(3, 1, 3)
            plt.plot(信号数据.index, 信号数据['操纵分数'], label='操纵分数')
            
            if '成交量异常分数' in 信号数据.columns:
                plt.plot(信号数据.index, 信号数据['成交量异常分数'], 
                       label='成交量异常分数', alpha=0.7)
            
            if '价格模式分数' in 信号数据.columns:
                plt.plot(信号数据.index, 信号数据['价格模式分数'], 
                       label='价格模式分数', alpha=0.7)
            
            if '机构活动分数' in 信号数据.columns:
                plt.plot(信号数据.index, 信号数据['机构活动分数'], 
                       label='机构活动分数', alpha=0.7)
            
            plt.title(f"{股票代码} 操纵分数")
            plt.xlabel('日期')
            plt.ylabel('分数')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(文件路径)
        plt.close()
        
        return 文件路径
    
    def _生成参数分布图表(self, 
                   优化结果: List[Dict[str, Any]],
                   目标指标: str) -> str:
        """生成参数分布图表。"""
        if not 优化结果:
            return ""
        
        # 创建图表目录
        图表目录 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'charts')
        os.makedirs(图表目录, exist_ok=True)
        
        # 生成文件名
        时间戳 = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        文件名 = f"参数优化_{时间戳}.png"
        文件路径 = os.path.join(图表目录, 文件名)
        
        # 提取参数和目标指标值
        参数名列表 = list(优化结果[0]['参数'].keys())
        目标值列表 = [结果['回测结果'][目标指标] for 结果 in 优化结果]
        
        # 创建图表
        参数数量 = len(参数名列表)
        图表行数 = (参数数量 + 1) // 2
        
        plt.figure(figsize=(12, 4 * 图表行数))
        
        # 为每个参数创建散点图
        for i, 参数名 in enumerate(参数名列表):
            plt.subplot(图表行数, 2, i + 1)
            
            参数值列表 = [结果['参数'][参数名] for 结果 in 优化结果]
            
            plt.scatter(参数值列表, 目标值列表, alpha=0.7)
            plt.title(f"{参数名} vs {目标指标}")
            plt.xlabel(参数名)
            plt.ylabel(目标指标)
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(文件路径)
        plt.close()
        
        return 文件路径
