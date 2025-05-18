"""
股票数据源基类

定义所有股票数据源需要实现的通用接口。

日期：2025-05-17
"""

import os
import logging
import time
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any

# 配置日志
logger = logging.getLogger(__name__)

class StockDataSource:
    """股票数据源基类，定义通用接口"""
    
    def __init__(self, config: Dict = None):
        """
        初始化数据源
        
        参数:
            config: 数据源配置，包含API密钥等信息
        """
        self.config = config or {}
        self.name = "基础数据源"
        self.cache_dir = os.path.join("data", "stock", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"初始化数据源: {self.name}")
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        raise NotImplementedError("子类必须实现此方法")
    
    def get_daily_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取日线数据
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            
        返回:
            包含OHLCV数据的DataFrame
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_minute_data(self, stock_code: str, start_date: str, end_date: str, freq: str = '1min') -> pd.DataFrame:
        """
        获取分钟线数据
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            freq: 频率，如'1min', '5min', '15min', '30min', '60min'
            
        返回:
            包含OHLCV数据的DataFrame
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_dragon_tiger_list(self, date: str = None) -> pd.DataFrame:
        """
        获取龙虎榜数据
        
        参数:
            date: 日期，格式：YYYY-MM-DD，默认为最近一个交易日
            
        返回:
            龙虎榜数据DataFrame
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_big_deal(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取大单交易数据
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            
        返回:
            大单交易数据DataFrame
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_shareholders_change(self, stock_code: str, quarter: str = None) -> pd.DataFrame:
        """
        获取股东变化数据
        
        参数:
            stock_code: 股票代码
            quarter: 季度，格式：YYYYQ1, YYYYQ2, YYYYQ3, YYYYQ4，默认为最近一个季度
            
        返回:
            股东变化数据DataFrame
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_financial_data(self, stock_code: str, report_type: str = 'income') -> pd.DataFrame:
        """
        获取财务数据
        
        参数:
            stock_code: 股票代码
            report_type: 报表类型，可选 'income'(利润表), 'balance'(资产负债表), 'cash'(现金流量表)
            
        返回:
            财务数据DataFrame
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_news_sentiment(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取新闻舆情数据
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            
        返回:
            新闻舆情数据DataFrame
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def _cache_file_path(self, data_type: str, params: Dict) -> str:
        """
        生成缓存文件路径
        
        参数:
            data_type: 数据类型
            params: 请求参数
            
        返回:
            缓存文件路径
        """
        # 将参数转换为字符串作为文件名的一部分
        param_str = "_".join([f"{k}_{v}" for k, v in params.items()])
        return os.path.join(self.cache_dir, f"{data_type}_{param_str}.csv")
    
    def _load_from_cache(self, data_type: str, params: Dict, cache_days: int = 1) -> Optional[pd.DataFrame]:
        """
        从缓存加载数据
        
        参数:
            data_type: 数据类型
            params: 请求参数
            cache_days: 缓存有效天数
            
        返回:
            缓存的DataFrame或None（如果缓存不存在或已过期）
        """
        cache_file = self._cache_file_path(data_type, params)
        if os.path.exists(cache_file):
            # 检查文件修改时间
            file_time = os.path.getmtime(cache_file)
            if (time.time() - file_time) / 86400 < cache_days:  # 转换为天数
                try:
                    return pd.read_csv(cache_file, parse_dates=True)
                except Exception as e:
                    logger.warning(f"读取缓存文件失败: {e}")
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, data_type: str, params: Dict) -> None:
        """
        保存数据到缓存
        
        参数:
            data: 要缓存的DataFrame
            data_type: 数据类型
            params: 请求参数
        """
        if data is None or data.empty:
            return
        
        cache_file = self._cache_file_path(data_type, params)
        try:
            data.to_csv(cache_file, index=False)
            logger.debug(f"数据已缓存到: {cache_file}")
        except Exception as e:
            logger.warning(f"保存缓存文件失败: {e}") 