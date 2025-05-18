"""
成交量分析组件

本组件负责分析交易量模式，识别异常成交量。

日期：2025-05-17
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# 配置日志
logger = logging.getLogger(__name__)

class VolumeAnalyzer:
    """成交量分析器，识别异常成交量模式"""
    
    def __init__(self, config: Dict = None):
        """
        初始化成交量分析器
        
        参数:
            config: 配置参数
        """
        self.config = config or {}
        
        # 默认参数
        self.volume_threshold = self.config.get('volume_threshold', 3.0)  # 成交量异常阈值（标准差倍数）
        
    def detect_anomalies(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        检测成交量异常
        
        参数:
            data: 包含OHLCV数据的DataFrame
            window: 滑动窗口大小
            
        返回:
            添加了成交量异常标记的DataFrame
        """
        if data.empty:
            logger.warning("输入数据为空，无法检测成交量异常")
            return data
        
        # 确保数据包含必要的列
        required_cols = ['volume']
        if not all(col in data.columns for col in required_cols):
            logger.error(f"输入数据缺少必要的列: {required_cols}")
            return data
        
        # 复制输入数据
        result = data.copy()
        
        try:
            # 计算成交量的对数收益率
            result['volume_log_return'] = np.log(result['volume'] / result['volume'].shift(1))
            
            # 计算成交量对数收益率的滚动均值和标准差
            result['volume_log_return_mean'] = result['volume_log_return'].rolling(window=window).mean()
            result['volume_log_return_std'] = result['volume_log_return'].rolling(window=window).std()
            
            # 计算成交量Z得分
            result['volume_zscore'] = (result['volume_log_return'] - result['volume_log_return_mean']) / result['volume_log_return_std']
            
            # 标记异常成交量
            result['volume_anomaly'] = np.where(
                result['volume_zscore'] > self.volume_threshold,
                1,  # 异常增加
                np.where(
                    result['volume_zscore'] < -self.volume_threshold,
                    -1,  # 异常减少
                    0  # 正常
                )
            )
            
            # 计算连续异常天数
            result['consecutive_anomalies'] = 0
            anomaly_count = 0
            
            for i in range(len(result)):
                if result['volume_anomaly'].iloc[i] != 0:
                    anomaly_count += 1
                else:
                    anomaly_count = 0
                result['consecutive_anomalies'].iloc[i] = anomaly_count
            
            # 删除中间计算列
            result = result.drop(['volume_log_return', 'volume_log_return_mean', 'volume_log_return_std'], axis=1)
            
            logger.info("成交量异常检测完成")
            return result
        
        except Exception as e:
            logger.error(f"成交量异常检测失败: {e}")
            return data
            
    def calculate_volume_concentration(self, data: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        计算成交量集中度
        
        参数:
            data: 包含OHLCV数据的DataFrame
            window: 滑动窗口大小
            
        返回:
            添加了成交量集中度的DataFrame
        """
        if data.empty:
            logger.warning("输入数据为空，无法计算成交量集中度")
            return data
            
        # 确保数据包含必要的列
        required_cols = ['volume']
        if not all(col in data.columns for col in required_cols):
            logger.error(f"输入数据缺少必要的列: {required_cols}")
            return data
            
        # 复制输入数据
        result = data.copy()
        
        try:
            # 计算成交量百分比
            result['volume_pct'] = result['volume'] / result['volume'].rolling(window=window).sum()
            
            # 计算成交量占比的标准差，作为集中度指标
            result['volume_concentration'] = result['volume_pct'].rolling(window=window).std() * 100
            
            logger.info("成交量集中度计算完成")
            return result
            
        except Exception as e:
            logger.error(f"成交量集中度计算失败: {e}")
            return data
    
    def analyze_volume_price_relation(self, data: pd.DataFrame, window: int = 10) -> Tuple[pd.DataFrame, Dict]:
        """
        分析成交量与价格的关系
        
        参数:
            data: 包含OHLCV数据的DataFrame
            window: 分析窗口大小
            
        返回:
            (添加了分析指标的DataFrame, 分析结果字典)
        """
        if data.empty:
            logger.warning("输入数据为空，无法分析成交量与价格关系")
            return data, {}
            
        # 确保数据包含必要的列
        required_cols = ['volume', 'close']
        if not all(col in data.columns for col in required_cols):
            logger.error(f"输入数据缺少必要的列: {required_cols}")
            return data, {}
            
        # 复制输入数据
        result = data.copy()
        
        try:
            # 计算价格变化率
            result['price_change'] = result['close'].pct_change()
            
            # 计算成交量变化率
            result['volume_change'] = result['volume'].pct_change()
            
            # 计算价格和成交量变化的相关性（滚动窗口）
            result['price_volume_corr'] = 0.0
            
            for i in range(window, len(result)):
                price_changes = result['price_change'].iloc[i-window:i]
                volume_changes = result['volume_change'].iloc[i-window:i]
                
                if not price_changes.isna().any() and not volume_changes.isna().any():
                    result['price_volume_corr'].iloc[i] = price_changes.corr(volume_changes)
            
            # 分析结果
            analysis_result = {}
            
            # 计算价涨量增比例
            price_up_volume_up = ((result['price_change'] > 0) & (result['volume_change'] > 0)).sum()
            price_down_volume_down = ((result['price_change'] < 0) & (result['volume_change'] < 0)).sum()
            price_up_volume_down = ((result['price_change'] > 0) & (result['volume_change'] < 0)).sum()
            price_down_volume_up = ((result['price_change'] < 0) & (result['volume_change'] > 0)).sum()
            
            total_days = price_up_volume_up + price_down_volume_down + price_up_volume_down + price_down_volume_up
            
            if total_days > 0:
                analysis_result['price_up_volume_up_pct'] = price_up_volume_up / total_days * 100
                analysis_result['price_down_volume_down_pct'] = price_down_volume_down / total_days * 100
                analysis_result['price_up_volume_down_pct'] = price_up_volume_down / total_days * 100
                analysis_result['price_down_volume_up_pct'] = price_down_volume_up / total_days * 100
                
                # 整体相关性
                analysis_result['overall_correlation'] = result['price_change'].corr(result['volume_change'])
                
                # 最近相关性
                analysis_result['recent_correlation'] = result['price_volume_corr'].iloc[-1] if len(result) > 0 else 0
            
            logger.info("成交量与价格关系分析完成")
            return result, analysis_result
            
        except Exception as e:
            logger.error(f"成交量与价格关系分析失败: {e}")
            return data, {} 