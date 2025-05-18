"""
Binance加密货币数据源

实现基于Binance API的数据获取功能。

日期：2025-05-17
"""

import time
import logging
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime, timedelta

from .base import CryptoDataSource

# 配置日志
logger = logging.getLogger(__name__)

class BinanceDataSource(CryptoDataSource):
    """基于Binance API的数据源实现"""
    
    def __init__(self, config: Dict = None):
        """
        初始化Binance数据源
        
        参数:
            config: 配置信息，包含API密钥等信息
        """
        super().__init__(config)
        self.name = "Binance"
        self.api_key = self.config.get('api_key', '')
        self.api_secret = self.config.get('api_secret', '')
        self.base_url = "https://api.binance.com"
        
        logger.info(f"Binance数据源初始化成功")
    
    def _make_request(self, endpoint: str, params: Dict = None, method: str = 'GET') -> Dict:
        """
        发送请求到Binance API
        
        参数:
            endpoint: API端点
            params: 请求参数
            method: 请求方法
            
        返回:
            API响应
        """
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        if self.api_key:
            headers['X-MBX-APIKEY'] = self.api_key
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers)
            else:
                response = requests.post(url, json=params, headers=headers)
            
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"Binance API请求失败: {e}")
            return {}
    
    def get_coin_list(self) -> pd.DataFrame:
        """
        获取币种列表
        
        返回:
            币种列表DataFrame
        """
        # 检查缓存
        cache_params = {'source': 'binance'}
        cached_data = self._load_from_cache('coin_list', cache_params)
        if cached_data is not None:
            return cached_data
        
        # 获取交易对信息
        exchange_info = self._make_request("/api/v3/exchangeInfo")
        
        if not exchange_info or 'symbols' not in exchange_info:
            logger.error("无法获取Binance交易对信息")
            return pd.DataFrame()
        
        # 提取USDT交易对
        usdt_symbols = []
        
        for symbol_info in exchange_info['symbols']:
            if symbol_info['quoteAsset'] == 'USDT' and symbol_info['status'] == 'TRADING':
                base_asset = symbol_info['baseAsset']
                symbol = symbol_info['symbol']
                
                usdt_symbols.append({
                    'symbol': symbol,
                    'base_asset': base_asset,
                    'quote_asset': 'USDT',
                    'is_trading': True
                })
        
        # 创建DataFrame
        coins_df = pd.DataFrame(usdt_symbols)
        
        # 缓存数据
        self._save_to_cache(coins_df, 'coin_list', cache_params)
        
        return coins_df
    
    def get_kline_data(self, symbol: str, interval: str, start_time: int = None, end_time: int = None, limit: int = 500) -> pd.DataFrame:
        """
        获取K线数据
        
        参数:
            symbol: 交易对，如'BTC/USDT'
            interval: K线间隔，如'1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M'
            start_time: 开始时间戳（毫秒）
            end_time: 结束时间戳（毫秒）
            limit: 返回的K线数量限制
            
        返回:
            K线数据DataFrame
        """
        # 标准化交易对格式
        symbol = symbol.replace('/', '')
        
        # 检查缓存
        cache_params = {
            'symbol': symbol,
            'interval': interval,
            'start': start_time or 0,
            'end': end_time or int(time.time() * 1000),
            'limit': limit
        }
        
        cached_data = self._load_from_cache('klines', cache_params)
        if cached_data is not None:
            return cached_data
        
        # 构建请求参数
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
            
        if end_time:
            params['endTime'] = end_time
        
        # 发送请求
        klines = self._make_request("/api/v3/klines", params)
        
        if not klines:
            logger.error(f"无法获取K线数据 - 交易对: {symbol}, 间隔: {interval}")
            return pd.DataFrame()
        
        # 处理返回数据
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                  'quote_volume', 'trades_count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
        
        df = pd.DataFrame(klines, columns=columns)
        
        # 转换数据类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                          'taker_buy_volume', 'taker_buy_quote_volume']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        # 转换时间戳
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # 设置索引
        df.set_index('timestamp', inplace=True)
        
        # 缓存数据
        self._save_to_cache(df.reset_index(), 'klines', cache_params)
        
        return df
    
    def get_ticker(self, symbol: str = None) -> pd.DataFrame:
        """
        获取最新行情数据
        
        参数:
            symbol: 交易对，如'BTC/USDT'，如果为None则返回所有交易对
            
        返回:
            最新行情数据DataFrame
        """
        endpoint = "/api/v3/ticker/24hr"
        params = {}
        
        if symbol:
            # 标准化交易对格式
            symbol = symbol.replace('/', '')
            params['symbol'] = symbol
        
        # 发送请求
        ticker_data = self._make_request(endpoint, params)
        
        if not ticker_data:
            logger.error(f"无法获取行情数据 - 交易对: {symbol or 'all'}")
            return pd.DataFrame()
        
        # 如果是单个交易对，转换为列表
        if not isinstance(ticker_data, list):
            ticker_data = [ticker_data]
        
        # 提取相关字段
        result = []
        for ticker in ticker_data:
            result.append({
                'symbol': ticker['symbol'],
                'price': float(ticker['lastPrice']),
                'price_change': float(ticker['priceChange']),
                'price_change_percent': float(ticker['priceChangePercent']),
                'high': float(ticker['highPrice']),
                'low': float(ticker['lowPrice']),
                'volume': float(ticker['volume']),
                'quote_volume': float(ticker['quoteVolume']),
                'trades': int(ticker['count']),
                'timestamp': datetime.fromtimestamp(int(ticker['closeTime']) / 1000)
            })
        
        return pd.DataFrame(result)
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """
        获取订单簿数据
        
        参数:
            symbol: 交易对，如'BTC/USDT'
            limit: 返回的订单数量
            
        返回:
            订单簿数据字典，包含bids和asks
        """
        # 标准化交易对格式
        symbol = symbol.replace('/', '')
        
        # 构建请求参数
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        # 发送请求
        orderbook = self._make_request("/api/v3/depth", params)
        
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            logger.error(f"无法获取订单簿数据 - 交易对: {symbol}")
            return {'bids': [], 'asks': []}
        
        # 处理返回数据
        result = {
            'bids': [[float(bid[0]), float(bid[1])] for bid in orderbook['bids']],
            'asks': [[float(ask[0]), float(ask[1])] for ask in orderbook['asks']],
            'timestamp': datetime.now()
        }
        
        return result
    
    def get_whale_transactions(self, coin: str, min_value_usd: float = 1000000, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        获取大额交易数据 (Binance API本身不提供此功能，返回空DataFrame)
        """
        logger.warning("Binance数据源不支持直接获取大额交易数据，请使用链上数据源")
        return pd.DataFrame()
    
    def get_exchange_flow(self, coin: str, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        获取交易所资金流向数据 (Binance API本身不提供此功能，返回空DataFrame)
        """
        logger.warning("Binance数据源不支持直接获取交易所资金流向数据，请使用链上数据源")
        return pd.DataFrame()
    
    def get_social_sentiment(self, coin: str, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        获取社交媒体情绪数据 (Binance API本身不提供此功能，返回空DataFrame)
        """
        logger.warning("Binance数据源不支持获取社交媒体情绪数据，请使用社交媒体数据源")
        return pd.DataFrame() 