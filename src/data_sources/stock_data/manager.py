"""
股票数据管理器

负责管理和协调多个数据源，提供统一的数据访问接口。

日期：2025-05-17
"""

import logging
from typing import Dict, List, Union, Optional, Tuple, Any
import pandas as pd

from .base import StockDataSource
from .akshare import AKShareDataSource
from .tushare import TushareDataSource
from .baostock import BaostockDataSource

# 配置日志
logger = logging.getLogger(__name__)

class StockDataManager:
    """股票数据管理器，统一管理多个数据源"""
    
    def __init__(self, config: Dict = None):
        """
        初始化数据管理器
        
        参数:
            config: 数据源配置，格式为 {数据源名称: 配置}
        """
        self.config = config or {}
        self.data_sources = {}
        self.default_source = None
        self.logger = logger
        
        # 初始化数据源
        self._init_data_sources()
        
        logger.info(f"股票数据管理器初始化成功，可用数据源: {list(self.data_sources.keys())}")
    
    def _init_data_sources(self):
        """初始化所有配置的数据源"""
        # 初始化AKShare数据源（最简单和免费的数据源，优先使用）
        if 'akshare' in self.config:
            try:
                self.data_sources['akshare'] = AKShareDataSource(self.config['akshare'])
                # 设置为默认数据源（如果未设置）
                if self.default_source is None:
                    self.default_source = 'akshare'
                logger.info("AKShare数据源初始化成功")
            except Exception as e:
                logger.error(f"AKShare数据源初始化失败: {e}")
        
        # 初始化Tushare数据源
        if 'tushare' in self.config and self.config['tushare'].get('api_key'):
            try:
                self.data_sources['tushare'] = TushareDataSource(self.config['tushare'])
                # 如果没有AKShare，设置为默认数据源
                if self.default_source is None:
                    self.default_source = 'tushare'
                logger.info("Tushare数据源初始化成功")
            except Exception as e:
                logger.error(f"Tushare数据源初始化失败: {e}")
        
        # 初始化Baostock数据源
        if 'baostock' in self.config:
            try:
                self.data_sources['baostock'] = BaostockDataSource(self.config['baostock'])
                # 如果没有其他数据源，设置为默认数据源
                if self.default_source is None:
                    self.default_source = 'baostock'
                logger.info("Baostock数据源初始化成功")
            except Exception as e:
                logger.error(f"Baostock数据源初始化失败: {e}")
        
        # 如果没有配置任何数据源，添加默认的AKShare数据源
        if not self.data_sources:
            try:
                self.data_sources['akshare'] = AKShareDataSource({})
                self.default_source = 'akshare'
                logger.info("添加默认AKShare数据源")
            except Exception as e:
                logger.error(f"默认AKShare数据源初始化失败: {e}")
    
    def get_data_source(self, source_name: str = None) -> StockDataSource:
        """
        获取指定数据源
        
        参数:
            source_name: 数据源名称，如果为None则使用默认数据源
            
        返回:
            数据源实例
        """
        if source_name is None:
            source_name = self.default_source
            
        if source_name not in self.data_sources:
            available_sources = list(self.data_sources.keys())
            if available_sources:
                source_name = available_sources[0]
                logger.warning(f"指定的数据源不可用，使用 {source_name} 作为备选")
            else:
                logger.error("没有可用的数据源")
                return None
        
        return self.data_sources[source_name]
    
    def get_stock_list(self, source_name: str = None) -> pd.DataFrame:
        """
        获取股票列表
        
        参数:
            source_name: 数据源名称，如果为None则使用默认数据源
            
        返回:
            股票列表DataFrame
        """
        data_source = self.get_data_source(source_name)
        return data_source.get_stock_list()
    
    def get_daily_data(self, stock_code: str, start_date: str, end_date: str, source_name: str = None) -> pd.DataFrame:
        """
        获取日线数据
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            source_name: 数据源名称，如果为None则使用默认数据源
            
        返回:
            包含OHLCV数据的DataFrame
        """
        data_source = self.get_data_source(source_name)
        return data_source.get_daily_data(stock_code, start_date, end_date)
    
    def get_minute_data(self, stock_code: str, start_date: str, end_date: str, freq: str = '1min', source_name: str = None) -> pd.DataFrame:
        """
        获取分钟线数据
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            freq: 频率，如'1min', '5min', '15min', '30min', '60min'
            source_name: 数据源名称，如果为None则使用默认数据源
            
        返回:
            包含OHLCV数据的DataFrame
        """
        data_source = self.get_data_source(source_name)
        return data_source.get_minute_data(stock_code, start_date, end_date, freq)
    
    def get_dragon_tiger_list(self, date: str = None, source_name: str = None) -> pd.DataFrame:
        """
        获取龙虎榜数据
        
        参数:
            date: 日期，格式：YYYY-MM-DD，默认为最近一个交易日
            source_name: 数据源名称，如果为None则使用AKShare（优先支持龙虎榜数据）
            
        返回:
            龙虎榜数据DataFrame
        """
        # 龙虎榜数据优先使用AKShare
        if source_name is None and 'akshare' in self.data_sources:
            source_name = 'akshare'
            
        data_source = self.get_data_source(source_name)
        return data_source.get_dragon_tiger_list(date)
    
    def get_big_deal(self, stock_code: str, start_date: str, end_date: str, source_name: str = None) -> pd.DataFrame:
        """
        获取大单交易数据
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            source_name: 数据源名称，如果为None则使用AKShare
            
        返回:
            大单交易数据DataFrame
        """
        if source_name is None and 'akshare' in self.data_sources:
            source_name = 'akshare'
            
        data_source = self.get_data_source(source_name)
        return data_source.get_big_deal(stock_code, start_date, end_date)
    
    def get_shareholders_change(self, stock_code: str, quarter: str = None, source_name: str = None) -> pd.DataFrame:
        """
        获取股东变化数据
        
        参数:
            stock_code: 股票代码
            quarter: 季度，格式：YYYYQ1, YYYYQ2, YYYYQ3, YYYYQ4，默认为最近一个季度
            source_name: 数据源名称，如果为None则使用AKShare
            
        返回:
            股东变化数据DataFrame
        """
        if source_name is None and 'akshare' in self.data_sources:
            source_name = 'akshare'
            
        data_source = self.get_data_source(source_name)
        return data_source.get_shareholders_change(stock_code, quarter)
    
    def get_financial_data(self, stock_code: str, report_type: str = 'income', source_name: str = None) -> pd.DataFrame:
        """
        获取财务数据
        
        参数:
            stock_code: 股票代码
            report_type: 报表类型，可选 'income'(利润表), 'balance'(资产负债表), 'cash'(现金流量表)
            source_name: 数据源名称，如果为None则使用优先级：Tushare > AKShare > Baostock
            
        返回:
            财务数据DataFrame
        """
        # 财务数据优先使用Tushare（质量最高）
        if source_name is None:
            if 'tushare' in self.data_sources:
                source_name = 'tushare'
            elif 'akshare' in self.data_sources:
                source_name = 'akshare'
            
        data_source = self.get_data_source(source_name)
        return data_source.get_financial_data(stock_code, report_type)
    
    def get_news_sentiment(self, stock_code: str, start_date: str, end_date: str, source_name: str = None) -> pd.DataFrame:
        """
        获取新闻舆情数据
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            source_name: 数据源名称，如果为None则使用Tushare
            
        返回:
            新闻舆情数据DataFrame
        """
        # 舆情数据优先使用Tushare
        if source_name is None and 'tushare' in self.data_sources:
            source_name = 'tushare'
            
        data_source = self.get_data_source(source_name)
        return data_source.get_news_sentiment(stock_code, start_date, end_date)
    
    def get_available_data_sources(self) -> List[str]:
        """
        获取所有可用的数据源名称
        
        返回:
            数据源名称列表
        """
        return list(self.data_sources.keys())
    
    def set_default_data_source(self, source_name: str) -> bool:
        """
        设置默认数据源
        
        参数:
            source_name: 数据源名称
            
        返回:
            是否设置成功
        """
        if source_name in self.data_sources:
            self.default_source = source_name
            self.logger.info(f"默认数据源已设置为: {source_name}")
            return True
        else:
            self.logger.warning(f"数据源 {source_name} 不存在，默认数据源未更改")
            return False 