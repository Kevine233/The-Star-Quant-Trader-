"""
加密货币数据管理器

负责管理和协调多个数据源，提供统一的数据访问接口。

日期：2025-05-17
"""

import logging
from typing import Dict, List, Union, Optional, Tuple, Any
import pandas as pd

from .base import CryptoDataSource
from .binance import BinanceDataSource
from .coingecko import CoinGeckoDataSource
from .glassnode import GlassnodeDataSource
from .lunarcrush import LunarCrushDataSource

# 配置日志
logger = logging.getLogger(__name__)

class CryptoDataManager:
    """加密货币数据管理器，统一管理多个数据源"""
    
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
        
        logger.info(f"加密货币数据管理器初始化成功，可用数据源: {list(self.data_sources.keys())}")
    
    def _init_data_sources(self):
        """初始化所有配置的数据源"""
        # 初始化Binance数据源
        if 'binance' in self.config:
            try:
                self.data_sources['binance'] = BinanceDataSource(self.config['binance'])
                # 设置为默认数据源（如果未设置）
                if self.default_source is None:
                    self.default_source = 'binance'
                logger.info("Binance数据源初始化成功")
            except Exception as e:
                logger.error(f"Binance数据源初始化失败: {e}")
        
        # 初始化CoinGecko数据源
        if 'coingecko' in self.config:
            try:
                self.data_sources['coingecko'] = CoinGeckoDataSource(self.config['coingecko'])
                # 如果没有Binance，设置为默认数据源
                if self.default_source is None:
                    self.default_source = 'coingecko'
                logger.info("CoinGecko数据源初始化成功")
            except Exception as e:
                logger.error(f"CoinGecko数据源初始化失败: {e}")
        
        # 初始化Glassnode数据源
        if 'glassnode' in self.config and self.config['glassnode'].get('api_key'):
            try:
                self.data_sources['glassnode'] = GlassnodeDataSource(self.config['glassnode'])
                logger.info("Glassnode数据源初始化成功")
            except Exception as e:
                logger.error(f"Glassnode数据源初始化失败: {e}")
        
        # 初始化LunarCrush数据源
        if 'lunarcrush' in self.config and self.config['lunarcrush'].get('api_key'):
            try:
                self.data_sources['lunarcrush'] = LunarCrushDataSource(self.config['lunarcrush'])
                logger.info("LunarCrush数据源初始化成功")
            except Exception as e:
                logger.error(f"LunarCrush数据源初始化失败: {e}")
        
        # 如果没有配置任何数据源，添加默认的CoinGecko数据源（不需要API密钥）
        if not self.data_sources:
            try:
                self.data_sources['coingecko'] = CoinGeckoDataSource({})
                self.default_source = 'coingecko'
                logger.info("添加默认CoinGecko数据源（无API密钥）")
            except Exception as e:
                logger.error(f"默认CoinGecko数据源初始化失败: {e}")
    
    def get_data_source(self, source_name: str = None) -> CryptoDataSource:
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
    
    def get_coin_list(self, source_name: str = None) -> pd.DataFrame:
        """
        获取加密货币列表
        
        参数:
            source_name: 数据源名称，如果为None则使用默认数据源
            
        返回:
            币种列表DataFrame
        """
        data_source = self.get_data_source(source_name)
        return data_source.get_coin_list()
    
    def get_kline_data(self, symbol: str, interval: str, start_time: int = None, end_time: int = None, limit: int = 500, source_name: str = None) -> pd.DataFrame:
        """
        获取K线数据
        
        参数:
            symbol: 交易对，如'BTC/USDT'
            interval: K线间隔，如'1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M'
            start_time: 开始时间戳（毫秒）
            end_time: 结束时间戳（毫秒）
            limit: 返回的K线数量限制
            source_name: 数据源名称，如果为None则使用默认数据源
            
        返回:
            K线数据DataFrame
        """
        # K线数据优先使用交易所数据源
        if source_name is None:
            if 'binance' in self.data_sources:
                source_name = 'binance'
            elif 'coingecko' in self.data_sources:
                source_name = 'coingecko'
        
        data_source = self.get_data_source(source_name)
        return data_source.get_kline_data(symbol, interval, start_time, end_time, limit)
    
    def get_ticker(self, symbol: str = None, source_name: str = None) -> pd.DataFrame:
        """
        获取最新行情数据
        
        参数:
            symbol: 交易对，如'BTC/USDT'，如果为None则返回所有交易对
            source_name: 数据源名称，如果为None则使用默认数据源
            
        返回:
            最新行情数据DataFrame
        """
        # 行情数据优先使用交易所数据源
        if source_name is None:
            if 'binance' in self.data_sources:
                source_name = 'binance'
            elif 'coingecko' in self.data_sources:
                source_name = 'coingecko'
        
        data_source = self.get_data_source(source_name)
        return data_source.get_ticker(symbol)
    
    def get_order_book(self, symbol: str, limit: int = 100, source_name: str = None) -> Dict:
        """
        获取订单簿数据
        
        参数:
            symbol: 交易对，如'BTC/USDT'
            limit: 返回的订单数量
            source_name: 数据源名称，如果为None则使用默认数据源
            
        返回:
            订单簿数据字典，包含bids和asks
        """
        # 订单簿数据只能使用交易所数据源
        if source_name is None:
            if 'binance' in self.data_sources:
                source_name = 'binance'
        
        data_source = self.get_data_source(source_name)
        return data_source.get_order_book(symbol, limit)
    
    def get_whale_transactions(self, coin: str, min_value_usd: float = 1000000, start_time: int = None, end_time: int = None, source_name: str = None) -> pd.DataFrame:
        """
        获取大额交易数据
        
        参数:
            coin: 币种，如'BTC'
            min_value_usd: 最小交易金额（美元）
            start_time: 开始时间戳（秒）
            end_time: 结束时间戳（秒）
            source_name: 数据源名称，如果为None则使用默认数据源
            
        返回:
            大额交易数据DataFrame
        """
        # 大额交易数据优先使用链上数据源
        if source_name is None:
            if 'glassnode' in self.data_sources:
                source_name = 'glassnode'
        
        data_source = self.get_data_source(source_name)
        return data_source.get_whale_transactions(coin, min_value_usd, start_time, end_time)
    
    def get_exchange_flow(self, coin: str, start_time: int = None, end_time: int = None, source_name: str = None) -> pd.DataFrame:
        """
        获取交易所资金流向数据
        
        参数:
            coin: 币种，如'BTC'
            start_time: 开始时间戳（秒）
            end_time: 结束时间戳（秒）
            source_name: 数据源名称，如果为None则使用默认数据源
            
        返回:
            交易所资金流向数据DataFrame
        """
        # 交易所资金流向数据优先使用链上数据源
        if source_name is None:
            if 'glassnode' in self.data_sources:
                source_name = 'glassnode'
        
        data_source = self.get_data_source(source_name)
        return data_source.get_exchange_flow(coin, start_time, end_time)
    
    def get_social_sentiment(self, coin: str, start_time: int = None, end_time: int = None, source_name: str = None) -> pd.DataFrame:
        """
        获取社交媒体情绪数据
        
        参数:
            coin: 币种，如'BTC'
            start_time: 开始时间戳（秒）
            end_time: 结束时间戳（秒）
            source_name: 数据源名称，如果为None则使用默认数据源
            
        返回:
            社交媒体情绪数据DataFrame
        """
        # 社交媒体情绪数据优先使用社交媒体数据源
        if source_name is None:
            if 'lunarcrush' in self.data_sources:
                source_name = 'lunarcrush'
            elif 'coingecko' in self.data_sources:
                source_name = 'coingecko'
        
        data_source = self.get_data_source(source_name)
        return data_source.get_social_sentiment(coin, start_time, end_time)
    
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