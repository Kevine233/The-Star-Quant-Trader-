"""
数据处理工具函数

提供各种数据处理、格式转换和验证的通用函数。

日期：2025-05-17
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime, timedelta

# 配置日志
logger = logging.getLogger(__name__)

def ensure_datetime_index(df: pd.DataFrame, date_col: str = None) -> pd.DataFrame:
    """
    确保DataFrame具有日期时间索引
    
    参数:
        df: 输入数据框
        date_col: 包含日期的列名，如果为None则假设索引已经是日期时间类型
        
    返回:
        具有日期时间索引的DataFrame
    """
    if df.empty:
        return df
        
    result = df.copy()
    
    try:
        if date_col is not None and date_col in result.columns:
            # 转换日期列并设置为索引
            result[date_col] = pd.to_datetime(result[date_col])
            result.set_index(date_col, inplace=True)
        elif not isinstance(result.index, pd.DatetimeIndex):
            # 尝试将现有索引转换为日期时间
            result.index = pd.to_datetime(result.index)
        
        # 排序索引
        result.sort_index(inplace=True)
        
        return result
    
    except Exception as e:
        logger.error(f"转换日期时间索引失败: {e}")
        return df
        
def normalize_ohlcv_data(df: pd.DataFrame, rename_mapping: Dict = None) -> pd.DataFrame:
    """
    标准化OHLCV数据列名
    
    参数:
        df: 输入数据框
        rename_mapping: 列重命名映射，如果为None则使用默认映射
        
    返回:
        标准化列名的DataFrame
    """
    if df.empty:
        return df
        
    result = df.copy()
    
    try:
        # 默认映射
        default_mapping = {
            # 开盘价可能的列名
            'Open': 'open', 'OPEN': 'open', 'open_price': 'open', 'OpenPrice': 'open',
            # 最高价可能的列名
            'High': 'high', 'HIGH': 'high', 'high_price': 'high', 'HighPrice': 'high',
            # 最低价可能的列名
            'Low': 'low', 'LOW': 'low', 'low_price': 'low', 'LowPrice': 'low',
            # 收盘价可能的列名
            'Close': 'close', 'CLOSE': 'close', 'close_price': 'close', 'ClosePrice': 'close',
            # 成交量可能的列名
            'Volume': 'volume', 'VOLUME': 'volume', 'vol': 'volume', 'Vol': 'volume',
            # 其他常见列
            'Date': 'date', 'DATE': 'date', 'Timestamp': 'date', 'Time': 'date',
            'Amount': 'amount', 'Turnover': 'amount', 'turnover': 'amount'
        }
        
        # 使用用户自定义映射（如果提供）
        mapping = rename_mapping or default_mapping
        
        # 仅重命名存在的列
        rename_cols = {old: new for old, new in mapping.items() if old in result.columns}
        if rename_cols:
            result.rename(columns=rename_cols, inplace=True)
        
        return result
    
    except Exception as e:
        logger.error(f"标准化OHLCV数据列名失败: {e}")
        return df
        
def resample_ohlcv_data(df: pd.DataFrame, timeframe: str = '1d') -> pd.DataFrame:
    """
    重采样OHLCV数据到指定时间周期
    
    参数:
        df: 具有日期时间索引的OHLCV数据框
        timeframe: 目标时间周期，如 '1m', '5m', '1h', '1d', '1w', '1M'
        
    返回:
        重采样后的DataFrame
    """
    if df.empty:
        return df
        
    # 确保数据具有日期时间索引
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("输入数据必须具有日期时间索引")
        return df
        
    # 确保数据具有必要的OHLCV列
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"输入数据缺少必要的OHLCV列: {required_cols}")
        return df
        
    try:
        # 定义重采样规则
        resampled = df.resample(timeframe)
        
        # 应用OHLCV聚合函数
        result = pd.DataFrame({
            'open': resampled['open'].first(),
            'high': resampled['high'].max(),
            'low': resampled['low'].min(),
            'close': resampled['close'].last(),
            'volume': resampled['volume'].sum()
        })
        
        # 处理可能的额外列
        if 'amount' in df.columns:
            result['amount'] = resampled['amount'].sum()
            
        # 删除空行
        result.dropna(inplace=True)
        
        return result
    
    except Exception as e:
        logger.error(f"重采样OHLCV数据失败: {e}")
        return df
        
def calculate_returns(df: pd.DataFrame, price_col: str = 'close', periods: List[int] = None) -> pd.DataFrame:
    """
    计算不同周期的收益率
    
    参数:
        df: 输入数据框
        price_col: 价格列名
        periods: 计算收益率的周期列表，如[1, 5, 20]代表日收益率、周收益率和月收益率
        
    返回:
        添加了收益率列的DataFrame
    """
    if df.empty or price_col not in df.columns:
        return df
        
    result = df.copy()
    periods = periods or [1, 5, 20, 60]
    
    try:
        for period in periods:
            col_name = f'return_{period}'
            result[col_name] = result[price_col].pct_change(period)
            
        return result
    
    except Exception as e:
        logger.error(f"计算收益率失败: {e}")
        return df
        
def load_market_data(filepath: str) -> pd.DataFrame:
    """
    加载市场数据文件（支持CSV、Excel和Parquet格式）
    
    参数:
        filepath: 文件路径
        
    返回:
        加载的数据DataFrame
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(filepath):
            logger.error(f"文件不存在: {filepath}")
            return pd.DataFrame()
            
        # 根据文件扩展名选择加载方法
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.csv':
            df = pd.read_csv(filepath)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        elif ext == '.parquet':
            df = pd.read_parquet(filepath)
        else:
            logger.error(f"不支持的文件格式: {ext}")
            return pd.DataFrame()
            
        # 标准化数据
        df = normalize_ohlcv_data(df)
        
        # 确保有日期时间索引
        df = ensure_datetime_index(df, 'date' if 'date' in df.columns else None)
        
        return df
    
    except Exception as e:
        logger.error(f"加载市场数据失败: {e}")
        return pd.DataFrame()
        
def save_market_data(df: pd.DataFrame, filepath: str) -> bool:
    """
    保存市场数据到文件
    
    参数:
        df: 要保存的DataFrame
        filepath: 文件路径
        
    返回:
        是否保存成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # 根据文件扩展名选择保存方法
        ext = os.path.splitext(filepath)[1].lower()
        
        # 如果索引是日期时间，重置索引以便保存
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        
        if ext == '.csv':
            df.to_csv(filepath, index=False)
        elif ext in ['.xlsx', '.xls']:
            df.to_excel(filepath, index=False)
        elif ext == '.parquet':
            df.to_parquet(filepath, index=False)
        else:
            logger.error(f"不支持的文件格式: {ext}")
            return False
            
        logger.info(f"数据已保存到: {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"保存市场数据失败: {e}")
        return False

# 添加DataUtils类，封装数据处理函数
class DataUtils:
    """
    数据工具类
    
    提供各种数据处理、清洗和转换的方法
    """
    
    @staticmethod
    def clean_data(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
        """
        清洗数据
        
        参数:
            df: 输入数据框
            config: 清洗配置
            
        返回:
            清洗后的DataFrame
        """
        if df.empty:
            return df
            
        result = df.copy()
        config = config or {}
        
        try:
            # 1. 处理缺失值
            if config.get('fill_missing', True):
                fill_method = config.get('fill_method', 'ffill')
                if fill_method == 'ffill':
                    result.fillna(method='ffill', inplace=True)
                elif fill_method == 'bfill':
                    result.fillna(method='bfill', inplace=True)
                elif fill_method == 'mean':
                    result.fillna(result.mean(), inplace=True)
                elif fill_method == 'median':
                    result.fillna(result.median(), inplace=True)
                elif fill_method == 'zero':
                    result.fillna(0, inplace=True)
            
            # 2. 处理异常值
            if config.get('handle_outliers', False):
                outlier_method = config.get('outlier_method', 'clip')
                outlier_threshold = config.get('outlier_threshold', 3.0)
                
                if outlier_method == 'clip':
                    # 使用标准差的倍数来裁剪异常值
                    for col in result.select_dtypes(include=[np.number]).columns:
                        mean = result[col].mean()
                        std = result[col].std()
                        result[col] = result[col].clip(
                            lower=mean - outlier_threshold * std,
                            upper=mean + outlier_threshold * std
                        )
                elif outlier_method == 'remove':
                    # 移除异常行
                    for col in result.select_dtypes(include=[np.number]).columns:
                        mean = result[col].mean()
                        std = result[col].std()
                        lower_bound = mean - outlier_threshold * std
                        upper_bound = mean + outlier_threshold * std
                        outlier_mask = (result[col] < lower_bound) | (result[col] > upper_bound)
                        result = result[~outlier_mask]
            
            # 3. 标准化数据列名
            if config.get('normalize_columns', True):
                result = normalize_ohlcv_data(result)
                
            # 4. 确保日期时间索引
            if config.get('ensure_datetime_index', True):
                date_col = config.get('date_column', 'date')
                result = ensure_datetime_index(result, date_col if date_col in result.columns else None)
                
            # 5. 重采样数据
            if config.get('resample', False):
                timeframe = config.get('timeframe', '1d')
                result = resample_ohlcv_data(result, timeframe)
                
            # 6. 移除重复行
            if config.get('remove_duplicates', True):
                result = result[~result.index.duplicated(keep='first')]
                
            # 7. 排序数据
            if config.get('sort_index', True):
                result.sort_index(inplace=True)
                
            return result
            
        except Exception as e:
            logger.error(f"清洗数据失败: {e}")
            return df
    
    @staticmethod
    def 清洗数据(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
        """
        清洗数据（中文别名）
        
        参数:
            df: 输入数据框
            config: 清洗配置
            
        返回:
            清洗后的DataFrame
        """
        return DataUtils.clean_data(df, config) 