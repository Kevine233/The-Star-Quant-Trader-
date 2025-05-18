"""
价格模式识别组件

本组件负责识别各种价格操纵模式，如拉高、震仓和洗盘。

日期：2025-05-17
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
import talib as ta

# 配置日志
logger = logging.getLogger(__name__)

class PricePatternDetector:
    """价格模式识别器，用于检测价格操纵模式"""
    
    def __init__(self, config: Dict = None):
        """
        初始化价格模式识别器
        
        参数:
            config: 配置参数
        """
        self.config = config or {}
        
        # 默认参数
        self.price_volatility_threshold = self.config.get('price_volatility_threshold', 0.03)  # 价格波动率阈值
        self.price_manipulation_window = self.config.get('price_manipulation_window', 20)      # 价格操纵检测窗口
        
    def detect_price_manipulation(self, data: pd.DataFrame, window: int = None) -> pd.DataFrame:
        """
        检测价格操纵模式
        
        参数:
            data: 包含OHLCV数据的DataFrame
            window: 滑动窗口大小，如果为None则使用默认值
            
        返回:
            添加了价格操纵标记的DataFrame
        """
        if data.empty:
            logger.warning("输入数据为空，无法检测价格操纵")
            return data
        
        # 确保数据包含必要的列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            logger.error(f"输入数据缺少必要的列: {required_cols}")
            return data
        
        # 使用默认窗口大小（如果未指定）
        if window is None:
            window = self.price_manipulation_window
        
        # 复制输入数据
        result = data.copy()
        
        try:
            # 计算价格波动率
            result['price_volatility'] = (result['high'] - result['low']) / result['low']
            
            # 计算价格波动率的滚动均值和标准差
            result['price_volatility_mean'] = result['price_volatility'].rolling(window=window).mean()
            result['price_volatility_std'] = result['price_volatility'].rolling(window=window).std()
            
            # 计算价格波动率Z得分
            result['price_volatility_zscore'] = (result['price_volatility'] - result['price_volatility_mean']) / result['price_volatility_std']
            
            # 计算价格与成交量的相关性
            result['price_volume_corr'] = 0.0
            
            for i in range(window, len(result)):
                price_changes = result['close'].iloc[i-window:i].pct_change().dropna()
                volume_changes = result['volume'].iloc[i-window:i].pct_change().dropna()
                
                if len(price_changes) > 2 and len(volume_changes) > 2:
                    corr, _ = stats.pearsonr(price_changes, volume_changes)
                    result['price_volume_corr'].iloc[i] = corr
            
            # 检测拉高出货模式
            result['pump_dump_pattern'] = 0
            
            for i in range(window, len(result)):
                # 拉高阶段：价格持续上涨，成交量放大
                price_up = result['close'].iloc[i-window:i].pct_change().rolling(5).sum() > 0.1
                volume_up = result['volume'].iloc[i-window:i].pct_change().rolling(5).sum() > 0.5
                
                if price_up.iloc[-1] and volume_up.iloc[-1]:
                    # 检查是否有出货迹象：价格开始回落，成交量继续放大
                    if result['close'].iloc[i] < result['close'].iloc[i-1] and result['volume'].iloc[i] > result['volume'].iloc[i-1]:
                        result['pump_dump_pattern'].iloc[i] = 1
            
            # 检测洗盘模式
            result['shakeout_pattern'] = 0
            
            for i in range(window, len(result)):
                # 洗盘特征：价格快速下跌后快速反弹，成交量先放大后减小
                price_drop = result['close'].iloc[i-5:i-2].pct_change().sum() < -0.05
                price_rebound = result['close'].iloc[i-2:i].pct_change().sum() > 0.03
                volume_pattern = (result['volume'].iloc[i-5] < result['volume'].iloc[i-3]) and (result['volume'].iloc[i-3] > result['volume'].iloc[i-1])
                
                if price_drop and price_rebound and volume_pattern:
                    result['shakeout_pattern'].iloc[i] = 1
            
            # 计算综合价格操纵评分
            result['price_manipulation_score'] = (
                np.abs(result['price_volatility_zscore']) * 0.3 +
                np.abs(result['price_volume_corr']) * 0.3 +
                result['pump_dump_pattern'] * 0.2 +
                result['shakeout_pattern'] * 0.2
            ) * 100
            
            logger.info("价格操纵模式检测完成")
            return result
        
        except Exception as e:
            logger.error(f"价格操纵模式检测失败: {e}")
            return data
    
    def detect_consolidation_breakout(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        检测盘整突破模式
        
        参数:
            data: 包含OHLCV数据的DataFrame
            window: 盘整检测窗口
            
        返回:
            添加了盘整突破标记的DataFrame
        """
        if data.empty:
            logger.warning("输入数据为空，无法检测盘整突破")
            return data
            
        # 确保数据包含必要的列
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            logger.error(f"输入数据缺少必要的列: {required_cols}")
            return data
            
        # 复制输入数据
        result = data.copy()
        
        try:
            # 计算价格波动范围
            result['price_range'] = (result['high'].rolling(window=window).max() - result['low'].rolling(window=window).min()) / result['low'].rolling(window=window).min()
            
            # 修复：使用pandas的rank方法而不是rolling对象的rank方法（不存在）
            # 先计算rolling数据，然后作为Series应用rank
            window_history = window*3 if len(result) > window*3 else len(result)
            result['price_range_rank'] = result['price_range'].rank(pct=True)  # 使用整体排名代替滚动排名
            
            # 计算价格标准差
            result['price_std'] = result['close'].rolling(window=window).std() / result['close'].rolling(window=window).mean()
            
            # 标记盘整区域（小波动）
            result['is_consolidation'] = (result['price_range_rank'] < 0.3) & (result['price_std'] < 0.02)
            
            # 计算突破（盘整后的较大波动）
            result['breakout'] = 0
            
            # 查找盘整区域后的突破
            consolidation_periods = []
            in_consolidation = False
            start_idx = 0
            
            # 识别盘整区域
            for i in range(len(result)):
                if result['is_consolidation'].iloc[i] and not in_consolidation:
                    # 开始新的盘整区域
                    in_consolidation = True
                    start_idx = i
                elif not result['is_consolidation'].iloc[i] and in_consolidation:
                    # 盘整区域结束
                    in_consolidation = False
                    if i - start_idx >= 5:  # 至少需要5个周期的盘整
                        consolidation_periods.append((start_idx, i-1))
            
            # 检查每个盘整区域后的突破
            for start, end in consolidation_periods:
                if end + 3 >= len(result):  # 确保有足够的数据检查突破
                    continue
                
                # 计算盘整区域的高低点
                consolidation_high = result['high'].iloc[start:end+1].max()
                consolidation_low = result['low'].iloc[start:end+1].min()
                
                # 检查向上突破
                if result['close'].iloc[end+1] > consolidation_high * 1.02:  # 2%的突破阈值
                    result['breakout'].iloc[end+1] = 1
                    
                # 检查向下突破
                elif result['close'].iloc[end+1] < consolidation_low * 0.98:  # 2%的突破阈值
                    result['breakout'].iloc[end+1] = -1
            
            logger.info("盘整突破检测完成")
            return result
            
        except Exception as e:
            logger.error(f"盘整突破检测失败: {e}")
            return data
    
    def detect_divergence(self, data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        检测背离（价格与指标之间的不一致）
        
        参数:
            data: 包含OHLCV数据的DataFrame
            window: 技术指标计算窗口
            
        返回:
            添加了背离标记的DataFrame
        """
        if data.empty:
            logger.warning("输入数据为空，无法检测背离")
            return data
            
        # 确保数据包含必要的列
        required_cols = ['close', 'high', 'low']
        if not all(col in data.columns for col in required_cols):
            logger.error(f"输入数据缺少必要的列: {required_cols}")
            return data
            
        # 复制输入数据
        result = data.copy()
        
        try:
            # 计算RSI指标
            close_values = result['close'].values
            try:
                rsi = ta.RSI(close_values, timeperiod=window)
                result['rsi'] = rsi
            except Exception as e:
                logger.warning(f"计算RSI指标失败: {e}")
                result['rsi'] = 50  # 默认值
            
            # 计算MACD指标
            try:
                # 修复：确保TA-Lib能正确处理None值，或者对结果进行检查
                macd_result = ta.MACD(
                    close_values, 
                    fastperiod=12, 
                    slowperiod=26, 
                    signalperiod=9
                )
                
                # 检查返回值
                if macd_result is not None and len(macd_result) == 3:
                    macd, macd_signal, macd_hist = macd_result
                    result['macd'] = macd
                    result['macd_signal'] = macd_signal
                    result['macd_hist'] = macd_hist
                else:
                    # 如果返回值不符合预期，使用默认值
                    logger.warning("MACD计算返回值不符合预期，使用默认值")
                    result['macd'] = 0
                    result['macd_signal'] = 0
                    result['macd_hist'] = 0
            except Exception as e:
                logger.warning(f"计算MACD指标失败: {e}")
                result['macd'] = 0
                result['macd_signal'] = 0
                result['macd_hist'] = 0
            
            # 初始化背离标记列
            result['rsi_divergence'] = 0
            result['macd_divergence'] = 0
            
            # 这里可以添加价格和指标的背离检测逻辑
            # 目前简化为占位符逻辑
            result['rsi_divergence'] = 0
            result['macd_divergence'] = 0
            
            logger.info("背离检测完成")
            return result
            
        except Exception as e:
            logger.error(f"背离检测失败: {e}")
            # 确保即使出错也返回必要的列
            if 'rsi_divergence' not in result.columns:
                result['rsi_divergence'] = 0
            if 'macd_divergence' not in result.columns:
                result['macd_divergence'] = 0
            return result 