"""
加密货币市场数据获取模块

本模块负责从多个数据源获取加密货币市场数据，包括：
1. 基础行情数据（K线、深度、成交量）
2. 链上数据（大户地址监控、资金流向）
3. 交易所资金流向
4. 社交媒体情绪数据

支持的数据源：
- Binance API (主要行情数据)
- CoinGecko API (市场概览数据)
- Glassnode API (链上数据，需要API密钥)
- LunarCrush API (社交媒体情绪数据，需要API密钥)

日期：2025-05-16
"""

import os
import sys
import time
import json
import logging
import datetime
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any

# 配置日志
logger = logging.getLogger(__name__)

class CryptoDataSource:
    """加密货币数据源基类，定义通用接口"""
    
    def __init__(self, config: Dict = None):
        """
        初始化数据源
        
        参数:
            config: 数据源配置，包含API密钥等信息
        """
        self.config = config or {}
        self.name = "基础数据源"
        self.cache_dir = os.path.join("data", "crypto", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"初始化数据源: {self.name}")
    
    def get_coin_list(self) -> pd.DataFrame:
        """获取加密货币列表"""
        raise NotImplementedError("子类必须实现此方法")
    
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
        raise NotImplementedError("子类必须实现此方法")
    
    def get_ticker(self, symbol: str = None) -> pd.DataFrame:
        """
        获取最新行情数据
        
        参数:
            symbol: 交易对，如'BTC/USDT'，如果为None则返回所有交易对
            
        返回:
            最新行情数据DataFrame
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """
        获取订单簿数据
        
        参数:
            symbol: 交易对，如'BTC/USDT'
            limit: 返回的订单数量
            
        返回:
            订单簿数据字典，包含bids和asks
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_whale_transactions(self, coin: str, min_value_usd: float = 1000000, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        获取大额交易数据
        
        参数:
            coin: 币种，如'BTC'
            min_value_usd: 最小交易金额（美元）
            start_time: 开始时间戳（毫秒）
            end_time: 结束时间戳（毫秒）
            
        返回:
            大额交易数据DataFrame
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_exchange_flow(self, coin: str, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        获取交易所资金流向数据
        
        参数:
            coin: 币种，如'BTC'
            start_time: 开始时间戳（毫秒）
            end_time: 结束时间戳（毫秒）
            
        返回:
            交易所资金流向数据DataFrame
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_social_sentiment(self, coin: str, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        获取社交媒体情绪数据
        
        参数:
            coin: 币种，如'BTC'
            start_time: 开始时间戳（毫秒）
            end_time: 结束时间戳（毫秒）
            
        返回:
            社交媒体情绪数据DataFrame
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
        param_str = "_".join([f"{k}_{v}" for k, v in params.items() if k != 'api_key'])
        return os.path.join(self.cache_dir, f"{data_type}_{param_str}.csv")
    
    def _load_from_cache(self, data_type: str, params: Dict, cache_hours: int = 24) -> Optional[pd.DataFrame]:
        """
        从缓存加载数据
        
        参数:
            data_type: 数据类型
            params: 请求参数
            cache_hours: 缓存有效小时数
            
        返回:
            缓存的DataFrame或None（如果缓存不存在或已过期）
        """
        cache_file = self._cache_file_path(data_type, params)
        if os.path.exists(cache_file):
            # 检查文件修改时间
            file_time = os.path.getmtime(cache_file)
            if (time.time() - file_time) / 3600 < cache_hours:  # 转换为小时数
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


class BinanceDataSource(CryptoDataSource):
    """基于Binance API的数据源实现"""
    
    def __init__(self, config: Dict = None):
        """
        初始化Binance数据源
        
        参数:
            config: 配置信息，可选包含api_key和api_secret
        """
        super().__init__(config)
        self.name = "Binance数据源"
        self.base_url = "https://api.binance.com"
        
        # 设置API密钥（如果有）
        self.api_key = self.config.get('api_key')
        self.api_secret = self.config.get('api_secret')
        
        # 设置请求头
        self.headers = {}
        if self.api_key:
            self.headers['X-MBX-APIKEY'] = self.api_key
        
        logger.info("Binance数据源初始化成功")
    
    def _make_request(self, endpoint: str, params: Dict = None, method: str = 'GET') -> Dict:
        """
        发送请求到Binance API
        
        参数:
            endpoint: API端点
            params: 请求参数
            method: 请求方法，'GET'或'POST'
            
        返回:
            响应数据字典
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=self.headers)
            else:
                response = requests.post(url, data=params, headers=self.headers)
            
            response.raise_for_status()  # 如果响应状态码不是200，抛出异常
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"请求Binance API失败: {e}")
            return {}
    
    def get_coin_list(self) -> pd.DataFrame:
        """获取Binance支持的所有交易对"""
        cache_params = {"type": "coin_list"}
        cached_data = self._load_from_cache("coin_list", cache_params, cache_hours=24)  # 交易对列表缓存24小时
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 获取交易对信息
            exchange_info = self._make_request("/api/v3/exchangeInfo")
            
            if not exchange_info or 'symbols' not in exchange_info:
                logger.error("获取Binance交易对信息失败")
                return pd.DataFrame(columns=['symbol', 'baseAsset', 'quoteAsset', 'status'])
            
            # 提取交易对信息
            symbols_data = []
            for symbol_info in exchange_info['symbols']:
                if symbol_info['status'] == 'TRADING':  # 只获取正在交易的交易对
                    symbols_data.append({
                        'symbol': symbol_info['symbol'],
                        'baseAsset': symbol_info['baseAsset'],
                        'quoteAsset': symbol_info['quoteAsset'],
                        'status': symbol_info['status']
                    })
            
            coin_list = pd.DataFrame(symbols_data)
            
            # 缓存数据
            self._save_to_cache(coin_list, "coin_list", cache_params)
            
            return coin_list
        except Exception as e:
            logger.error(f"获取Binance交易对列表失败: {e}")
            return pd.DataFrame(columns=['symbol', 'baseAsset', 'quoteAsset', 'status'])
    
    def get_kline_data(self, symbol: str, interval: str, start_time: int = None, end_time: int = None, limit: int = 500) -> pd.DataFrame:
        """获取K线数据"""
        # 标准化交易对格式
        symbol = symbol.replace('/', '')
        
        cache_params = {
            "symbol": symbol,
            "interval": interval,
            "start_time": start_time or 0,
            "end_time": end_time or int(time.time() * 1000),
            "limit": limit
        }
        
        # 实时数据不使用缓存
        if interval in ['1m', '3m', '5m', '15m', '30m', '1h'] and (time.time() * 1000 - cache_params["end_time"]) < 3600000:
            cached_data = None
        else:
            cached_data = self._load_from_cache("kline", cache_params, cache_hours=1)  # K线数据缓存1小时
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 准备请求参数
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            if start_time:
                params['startTime'] = start_time
            
            if end_time:
                params['endTime'] = end_time
            
            # 获取K线数据
            kline_data = self._make_request("/api/v3/klines", params)
            
            if not kline_data:
                logger.error(f"获取Binance K线数据失败 - 交易对: {symbol}, 间隔: {interval}")
                return pd.DataFrame(columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote'])
            
            # 转换为DataFrame
            df = pd.DataFrame(kline_data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base', 'taker_buy_quote']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 转换时间戳为日期时间
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # 删除无用列
            if 'ignore' in df.columns:
                df = df.drop('ignore', axis=1)
            
            # 缓存数据
            self._save_to_cache(df, "kline", cache_params)
            
            return df
        except Exception as e:
            logger.error(f"获取Binance K线数据失败 - 交易对: {symbol}, 间隔: {interval}, 错误: {e}")
            return pd.DataFrame(columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote'])
    
    def get_ticker(self, symbol: str = None) -> pd.DataFrame:
        """获取最新行情数据"""
        cache_params = {"symbol": symbol or "ALL"}
        
        # 实时数据不使用缓存
        cached_data = None
        
        try:
            # 准备请求参数
            params = {}
            if symbol:
                symbol = symbol.replace('/', '')
                params['symbol'] = symbol
                endpoint = "/api/v3/ticker/24hr"
            else:
                endpoint = "/api/v3/ticker/24hr"
            
            # 获取行情数据
            ticker_data = self._make_request(endpoint, params)
            
            if not ticker_data:
                logger.error(f"获取Binance行情数据失败 - 交易对: {symbol or 'ALL'}")
                return pd.DataFrame()
            
            # 转换为DataFrame
            if isinstance(ticker_data, list):
                df = pd.DataFrame(ticker_data)
            else:
                df = pd.DataFrame([ticker_data])
            
            # 转换数据类型
            numeric_cols = [
                'priceChange', 'priceChangePercent', 'weightedAvgPrice', 'prevClosePrice',
                'lastPrice', 'lastQty', 'bidPrice', 'bidQty', 'askPrice', 'askQty',
                'openPrice', 'highPrice', 'lowPrice', 'volume', 'quoteVolume'
            ]
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
        except Exception as e:
            logger.error(f"获取Binance行情数据失败 - 交易对: {symbol or 'ALL'}, 错误: {e}")
            return pd.DataFrame()
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """获取订单簿数据"""
        # 标准化交易对格式
        symbol = symbol.replace('/', '')
        
        try:
            # 准备请求参数
            params = {
                'symbol': symbol,
                'limit': limit
            }
            
            # 获取订单簿数据
            order_book = self._make_request("/api/v3/depth", params)
            
            if not order_book or 'bids' not in order_book or 'asks' not in order_book:
                logger.error(f"获取Binance订单簿数据失败 - 交易对: {symbol}")
                return {'bids': [], 'asks': []}
            
            # 转换数据类型
            bids = [[float(price), float(qty)] for price, qty in order_book['bids']]
            asks = [[float(price), float(qty)] for price, qty in order_book['asks']]
            
            return {
                'bids': bids,
                'asks': asks,
                'lastUpdateId': order_book.get('lastUpdateId')
            }
        except Exception as e:
            logger.error(f"获取Binance订单簿数据失败 - 交易对: {symbol}, 错误: {e}")
            return {'bids': [], 'asks': []}
    
    def get_whale_transactions(self, coin: str, min_value_usd: float = 1000000, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        获取大额交易数据
        
        注意：Binance API不直接提供链上大额交易数据，此方法仅作为示例，实际项目中可能需要使用其他数据源
        """
        logger.warning("Binance API不直接提供链上大额交易数据，请使用专门的链上数据源")
        return pd.DataFrame()
    
    def get_exchange_flow(self, coin: str, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        获取交易所资金流向数据
        
        注意：Binance API不直接提供交易所资金流向数据，此方法仅作为示例，实际项目中可能需要使用其他数据源
        """
        logger.warning("Binance API不直接提供交易所资金流向数据，请使用专门的链上数据源")
        return pd.DataFrame()
    
    def get_social_sentiment(self, coin: str, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        获取社交媒体情绪数据
        
        注意：Binance API不提供社交媒体情绪数据，此方法仅作为示例，实际项目中可能需要使用其他数据源
        """
        logger.warning("Binance API不提供社交媒体情绪数据，请使用专门的社交媒体数据源")
        return pd.DataFrame()


class CoinGeckoDataSource(CryptoDataSource):
    """基于CoinGecko API的数据源实现"""
    
    def __init__(self, config: Dict = None):
        """
        初始化CoinGecko数据源
        
        参数:
            config: 配置信息，可选包含api_key
        """
        super().__init__(config)
        self.name = "CoinGecko数据源"
        self.base_url = "https://api.coingecko.com/api/v3"
        
        # 设置API密钥（如果有）
        self.api_key = self.config.get('api_key')
        
        # 设置请求头
        self.headers = {}
        if self.api_key:
            self.headers['x-cg-pro-api-key'] = self.api_key
        
        logger.info("CoinGecko数据源初始化成功")
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        发送请求到CoinGecko API
        
        参数:
            endpoint: API端点
            params: 请求参数
            
        返回:
            响应数据字典
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            
            # 检查是否达到API速率限制
            if response.status_code == 429:
                logger.warning("CoinGecko API速率限制，等待60秒后重试")
                time.sleep(60)
                response = requests.get(url, params=params, headers=self.headers)
            
            response.raise_for_status()  # 如果响应状态码不是200，抛出异常
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"请求CoinGecko API失败: {e}")
            return {}
    
    def get_coin_list(self) -> pd.DataFrame:
        """获取CoinGecko支持的所有加密货币"""
        cache_params = {"type": "coin_list"}
        cached_data = self._load_from_cache("coin_list", cache_params, cache_hours=24)  # 币种列表缓存24小时
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 获取币种列表
            coin_list_data = self._make_request("/coins/list")
            
            if not coin_list_data:
                logger.error("获取CoinGecko币种列表失败")
                return pd.DataFrame(columns=['id', 'symbol', 'name'])
            
            # 转换为DataFrame
            coin_list = pd.DataFrame(coin_list_data)
            
            # 缓存数据
            self._save_to_cache(coin_list, "coin_list", cache_params)
            
            return coin_list
        except Exception as e:
            logger.error(f"获取CoinGecko币种列表失败: {e}")
            return pd.DataFrame(columns=['id', 'symbol', 'name'])
    
    def get_coin_market_data(self, vs_currency: str = 'usd', ids: List[str] = None, category: str = None, order: str = 'market_cap_desc', per_page: int = 100, page: int = 1) -> pd.DataFrame:
        """
        获取加密货币市场数据
        
        参数:
            vs_currency: 计价货币，如'usd', 'eur', 'jpy'等
            ids: 币种ID列表，如['bitcoin', 'ethereum']
            category: 币种类别，如'defi', 'stablecoins'等
            order: 排序方式，如'market_cap_desc', 'volume_desc'等
            per_page: 每页数量
            page: 页码
            
        返回:
            市场数据DataFrame
        """
        cache_params = {
            "vs_currency": vs_currency,
            "ids": ','.join(ids) if ids else 'all',
            "category": category or 'all',
            "order": order,
            "per_page": per_page,
            "page": page
        }
        
        # 市场数据变化快，缓存时间短
        cached_data = self._load_from_cache("market", cache_params, cache_hours=1)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 准备请求参数
            params = {
                'vs_currency': vs_currency,
                'order': order,
                'per_page': per_page,
                'page': page,
                'sparkline': 'false'
            }
            
            if ids:
                params['ids'] = ','.join(ids)
            
            if category:
                params['category'] = category
            
            # 获取市场数据
            market_data = self._make_request("/coins/markets", params)
            
            if not market_data:
                logger.error("获取CoinGecko市场数据失败")
                return pd.DataFrame()
            
            # 转换为DataFrame
            market_df = pd.DataFrame(market_data)
            
            # 缓存数据
            self._save_to_cache(market_df, "market", cache_params)
            
            return market_df
        except Exception as e:
            logger.error(f"获取CoinGecko市场数据失败: {e}")
            return pd.DataFrame()
    
    def get_kline_data(self, symbol: str, interval: str, start_time: int = None, end_time: int = None, limit: int = 500) -> pd.DataFrame:
        """
        获取K线数据
        
        注意：CoinGecko API的K线数据接口与Binance不同，需要先获取币种ID
        
        参数:
            symbol: 币种符号，如'BTC'
            interval: K线间隔，CoinGecko支持的间隔有'daily', 'hourly'
            start_time: 开始时间戳（秒）
            end_time: 结束时间戳（秒）
            limit: 返回的K线数量限制（CoinGecko API不支持此参数）
            
        返回:
            K线数据DataFrame
        """
        # 转换间隔格式
        if interval in ['1d', 'D', 'day', 'daily']:
            cg_interval = 'daily'
        else:
            cg_interval = 'hourly'  # CoinGecko只支持daily和hourly两种间隔
        
        # 获取币种ID
        coin_id = self._get_coin_id(symbol)
        if not coin_id:
            logger.error(f"无法获取币种ID - 币种: {symbol}")
            return pd.DataFrame(columns=['timestamp', 'price'])
        
        # 计算天数/小时数
        days = 1
        if start_time and end_time:
            if cg_interval == 'daily':
                days = max(1, int((end_time - start_time) / 86400))  # 秒转换为天
            else:
                days = max(1, int((end_time - start_time) / 3600))  # 秒转换为小时
        
        cache_params = {
            "coin_id": coin_id,
            "interval": cg_interval,
            "days": days
        }
        
        # 实时数据不使用缓存
        if days <= 1:
            cached_data = None
        else:
            cached_data = self._load_from_cache("kline", cache_params, cache_hours=1)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 准备请求参数
            params = {'days': days}
            
            # 获取市场图表数据
            endpoint = f"/coins/{coin_id}/market_chart"
            chart_data = self._make_request(endpoint, params)
            
            if not chart_data or 'prices' not in chart_data:
                logger.error(f"获取CoinGecko K线数据失败 - 币种: {symbol}, 间隔: {cg_interval}")
                return pd.DataFrame(columns=['timestamp', 'price'])
            
            # 提取价格数据
            prices = chart_data['prices']
            
            # 转换为DataFrame
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            
            # 转换时间戳为日期时间
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 如果有成交量数据，添加到DataFrame
            if 'total_volumes' in chart_data:
                volumes = chart_data['total_volumes']
                volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
                
                # 合并价格和成交量数据
                df = pd.merge(df, volume_df, on='timestamp', how='left')
            
            # 如果有市值数据，添加到DataFrame
            if 'market_caps' in chart_data:
                market_caps = chart_data['market_caps']
                market_cap_df = pd.DataFrame(market_caps, columns=['timestamp', 'market_cap'])
                market_cap_df['timestamp'] = pd.to_datetime(market_cap_df['timestamp'], unit='ms')
                
                # 合并市值数据
                df = pd.merge(df, market_cap_df, on='timestamp', how='left')
            
            # 缓存数据
            self._save_to_cache(df, "kline", cache_params)
            
            return df
        except Exception as e:
            logger.error(f"获取CoinGecko K线数据失败 - 币种: {symbol}, 间隔: {cg_interval}, 错误: {e}")
            return pd.DataFrame(columns=['timestamp', 'price'])
    
    def _get_coin_id(self, symbol: str) -> str:
        """
        根据币种符号获取CoinGecko的币种ID
        
        参数:
            symbol: 币种符号，如'BTC'
            
        返回:
            币种ID，如'bitcoin'
        """
        # 标准化符号
        symbol = symbol.lower()
        if '/' in symbol:
            symbol = symbol.split('/')[0]
        
        # 获取币种列表
        coin_list = self.get_coin_list()
        
        if coin_list.empty:
            return ""
        
        # 查找匹配的币种
        matches = coin_list[coin_list['symbol'].str.lower() == symbol]
        
        if matches.empty:
            # 尝试模糊匹配
            matches = coin_list[coin_list['symbol'].str.lower().str.contains(symbol)]
        
        if not matches.empty:
            # 返回第一个匹配的币种ID
            return matches.iloc[0]['id']
        
        return ""
    
    def get_ticker(self, symbol: str = None) -> pd.DataFrame:
        """
        获取最新行情数据
        
        参数:
            symbol: 币种符号，如'BTC'
            
        返回:
            最新行情数据DataFrame
        """
        if symbol:
            # 获取币种ID
            coin_id = self._get_coin_id(symbol)
            if not coin_id:
                logger.error(f"无法获取币种ID - 币种: {symbol}")
                return pd.DataFrame()
            
            try:
                # 获取币种详情
                endpoint = f"/coins/{coin_id}"
                coin_data = self._make_request(endpoint, {'localization': 'false', 'tickers': 'true', 'market_data': 'true'})
                
                if not coin_data:
                    logger.error(f"获取CoinGecko币种详情失败 - 币种: {symbol}")
                    return pd.DataFrame()
                
                # 提取行情数据
                tickers = coin_data.get('tickers', [])
                
                # 转换为DataFrame
                if tickers:
                    ticker_df = pd.DataFrame(tickers)
                    return ticker_df
                else:
                    # 如果没有tickers数据，使用market_data
                    market_data = coin_data.get('market_data', {})
                    if market_data:
                        market_df = pd.DataFrame([{
                            'symbol': symbol,
                            'current_price': market_data.get('current_price', {}).get('usd'),
                            'market_cap': market_data.get('market_cap', {}).get('usd'),
                            'total_volume': market_data.get('total_volume', {}).get('usd'),
                            'high_24h': market_data.get('high_24h', {}).get('usd'),
                            'low_24h': market_data.get('low_24h', {}).get('usd'),
                            'price_change_24h': market_data.get('price_change_24h'),
                            'price_change_percentage_24h': market_data.get('price_change_percentage_24h')
                        }])
                        return market_df
                    else:
                        return pd.DataFrame()
            except Exception as e:
                logger.error(f"获取CoinGecko行情数据失败 - 币种: {symbol}, 错误: {e}")
                return pd.DataFrame()
        else:
            # 获取所有币种的市场数据
            return self.get_coin_market_data()
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """
        获取订单簿数据
        
        注意：CoinGecko API不提供订单簿数据，此方法仅作为示例
        """
        logger.warning("CoinGecko API不提供订单簿数据")
        return {'bids': [], 'asks': []}
    
    def get_whale_transactions(self, coin: str, min_value_usd: float = 1000000, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        获取大额交易数据
        
        注意：CoinGecko API不提供链上大额交易数据，此方法仅作为示例
        """
        logger.warning("CoinGecko API不提供链上大额交易数据")
        return pd.DataFrame()
    
    def get_exchange_flow(self, coin: str, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        获取交易所资金流向数据
        
        注意：CoinGecko API不提供交易所资金流向数据，此方法仅作为示例
        """
        logger.warning("CoinGecko API不提供交易所资金流向数据")
        return pd.DataFrame()
    
    def get_social_sentiment(self, coin: str, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        获取社交媒体情绪数据
        
        注意：CoinGecko API不提供详细的社交媒体情绪数据，此方法仅作为示例
        """
        # 获取币种ID
        coin_id = self._get_coin_id(coin)
        if not coin_id:
            logger.error(f"无法获取币种ID - 币种: {coin}")
            return pd.DataFrame()
        
        try:
            # 获取币种详情
            endpoint = f"/coins/{coin_id}"
            coin_data = self._make_request(endpoint, {'localization': 'false', 'tickers': 'false', 'community_data': 'true'})
            
            if not coin_data or 'community_data' not in coin_data:
                logger.error(f"获取CoinGecko社区数据失败 - 币种: {coin}")
                return pd.DataFrame()
            
            # 提取社区数据
            community_data = coin_data['community_data']
            
            # 转换为DataFrame
            community_df = pd.DataFrame([{
                'twitter_followers': community_data.get('twitter_followers'),
                'reddit_subscribers': community_data.get('reddit_subscribers'),
                'reddit_average_posts_48h': community_data.get('reddit_average_posts_48h'),
                'reddit_average_comments_48h': community_data.get('reddit_average_comments_48h'),
                'telegram_channel_user_count': community_data.get('telegram_channel_user_count')
            }])
            
            return community_df
        except Exception as e:
            logger.error(f"获取CoinGecko社区数据失败 - 币种: {coin}, 错误: {e}")
            return pd.DataFrame()


class GlassnodeDataSource(CryptoDataSource):
    """基于Glassnode API的数据源实现，用于获取链上数据"""
    
    def __init__(self, config: Dict = None):
        """
        初始化Glassnode数据源
        
        参数:
            config: 配置信息，必须包含api_key
        """
        super().__init__(config)
        self.name = "Glassnode数据源"
        self.base_url = "https://api.glassnode.com/v1"
        
        # 检查配置中是否包含API密钥
        if not config or 'api_key' not in config:
            logger.error("未提供Glassnode API密钥，无法使用此数据源")
            raise ValueError("Glassnode数据源需要API密钥")
        
        self.api_key = config['api_key']
        logger.info("Glassnode数据源初始化成功")
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        发送请求到Glassnode API
        
        参数:
            endpoint: API端点
            params: 请求参数
            
        返回:
            响应数据字典
        """
        url = f"{self.base_url}{endpoint}"
        
        # 添加API密钥
        if params is None:
            params = {}
        params['api_key'] = self.api_key
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # 如果响应状态码不是200，抛出异常
            
            # Glassnode API返回的是JSON数组
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"请求Glassnode API失败: {e}")
            return []
    
    def get_coin_list(self) -> pd.DataFrame:
        """获取Glassnode支持的所有加密货币"""
        # Glassnode API不提供币种列表接口，返回常用币种
        common_coins = [
            {'symbol': 'BTC', 'name': 'Bitcoin'},
            {'symbol': 'ETH', 'name': 'Ethereum'},
            {'symbol': 'LTC', 'name': 'Litecoin'},
            {'symbol': 'AAVE', 'name': 'Aave'},
            {'symbol': 'ABT', 'name': 'Arcblock'},
            {'symbol': 'AMPL', 'name': 'Ampleforth'},
            {'symbol': 'ANT', 'name': 'Aragon'},
            {'symbol': 'ARMOR', 'name': 'Armor'},
            {'symbol': 'BADGER', 'name': 'Badger DAO'},
            {'symbol': 'BAL', 'name': 'Balancer'},
            {'symbol': 'BAND', 'name': 'Band Protocol'},
            {'symbol': 'BAT', 'name': 'Basic Attention Token'},
            {'symbol': 'BNT', 'name': 'Bancor'},
            {'symbol': 'BOND', 'name': 'BarnBridge'},
            {'symbol': 'BTC', 'name': 'Bitcoin'},
            {'symbol': 'COMP', 'name': 'Compound'},
            {'symbol': 'CRV', 'name': 'Curve DAO Token'},
            {'symbol': 'DAI', 'name': 'Dai'},
            {'symbol': 'ETH', 'name': 'Ethereum'},
            {'symbol': 'GNO', 'name': 'Gnosis'},
            {'symbol': 'LRC', 'name': 'Loopring'},
            {'symbol': 'MKR', 'name': 'Maker'},
            {'symbol': 'REN', 'name': 'Ren'},
            {'symbol': 'SNX', 'name': 'Synthetix'},
            {'symbol': 'UNI', 'name': 'Uniswap'},
            {'symbol': 'USDC', 'name': 'USD Coin'},
            {'symbol': 'USDT', 'name': 'Tether'},
            {'symbol': 'WBTC', 'name': 'Wrapped Bitcoin'},
            {'symbol': 'YFI', 'name': 'yearn.finance'},
            {'symbol': 'ZRX', 'name': '0x'}
        ]
        
        return pd.DataFrame(common_coins)
    
    def get_kline_data(self, symbol: str, interval: str, start_time: int = None, end_time: int = None, limit: int = 500) -> pd.DataFrame:
        """
        获取K线数据
        
        注意：Glassnode API不直接提供K线数据，此方法仅作为示例
        """
        logger.warning("Glassnode API不直接提供K线数据，请使用交易所数据源")
        return pd.DataFrame()
    
    def get_ticker(self, symbol: str = None) -> pd.DataFrame:
        """
        获取最新行情数据
        
        注意：Glassnode API不直接提供行情数据，此方法仅作为示例
        """
        logger.warning("Glassnode API不直接提供行情数据，请使用交易所数据源")
        return pd.DataFrame()
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """
        获取订单簿数据
        
        注意：Glassnode API不提供订单簿数据，此方法仅作为示例
        """
        logger.warning("Glassnode API不提供订单簿数据")
        return {'bids': [], 'asks': []}
    
    def get_whale_transactions(self, coin: str, min_value_usd: float = 1000000, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        获取大额交易数据
        
        参数:
            coin: 币种，如'BTC'
            min_value_usd: 最小交易金额（美元）
            start_time: 开始时间戳（秒）
            end_time: 结束时间戳（秒）
            
        返回:
            大额交易数据DataFrame
        """
        # 标准化币种符号
        coin = coin.upper()
        
        cache_params = {
            "coin": coin,
            "min_value_usd": min_value_usd,
            "start_time": start_time or 0,
            "end_time": end_time or int(time.time())
        }
        
        cached_data = self._load_from_cache("whale_transactions", cache_params, cache_hours=1)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 准备请求参数
            params = {
                'a': coin,
                'i': '24h'  # 时间间隔
            }
            
            if start_time:
                params['s'] = start_time
            
            if end_time:
                params['u'] = end_time
            
            # 获取大额转账数据
            # 注意：Glassnode API的具体端点可能与此不同，请参考官方文档
            endpoint = "/metrics/transactions/transfers_volume_large"
            whale_data = self._make_request(endpoint, params)
            
            if not whale_data:
                logger.error(f"获取Glassnode大额交易数据失败 - 币种: {coin}")
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame(whale_data)
            
            # 转换时间戳为日期时间
            if 't' in df.columns:
                df['timestamp'] = pd.to_datetime(df['t'], unit='s')
                df = df.drop('t', axis=1)
            
            # 转换值为数值类型
            if 'v' in df.columns:
                df['value'] = pd.to_numeric(df['v'], errors='coerce')
                df = df.drop('v', axis=1)
            
            # 缓存数据
            self._save_to_cache(df, "whale_transactions", cache_params)
            
            return df
        except Exception as e:
            logger.error(f"获取Glassnode大额交易数据失败 - 币种: {coin}, 错误: {e}")
            return pd.DataFrame()
    
    def get_exchange_flow(self, coin: str, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        获取交易所资金流向数据
        
        参数:
            coin: 币种，如'BTC'
            start_time: 开始时间戳（秒）
            end_time: 结束时间戳（秒）
            
        返回:
            交易所资金流向数据DataFrame
        """
        # 标准化币种符号
        coin = coin.upper()
        
        cache_params = {
            "coin": coin,
            "start_time": start_time or 0,
            "end_time": end_time or int(time.time())
        }
        
        cached_data = self._load_from_cache("exchange_flow", cache_params, cache_hours=1)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 准备请求参数
            params = {
                'a': coin,
                'i': '24h'  # 时间间隔
            }
            
            if start_time:
                params['s'] = start_time
            
            if end_time:
                params['u'] = end_time
            
            # 获取交易所流入数据
            endpoint_inflow = "/metrics/transactions/transfers_volume_to_exchanges_sum"
            inflow_data = self._make_request(endpoint_inflow, params)
            
            # 获取交易所流出数据
            endpoint_outflow = "/metrics/transactions/transfers_volume_from_exchanges_sum"
            outflow_data = self._make_request(endpoint_outflow, params)
            
            if not inflow_data or not outflow_data:
                logger.error(f"获取Glassnode交易所资金流向数据失败 - 币种: {coin}")
                return pd.DataFrame()
            
            # 转换为DataFrame
            inflow_df = pd.DataFrame(inflow_data)
            outflow_df = pd.DataFrame(outflow_data)
            
            # 转换时间戳为日期时间
            if 't' in inflow_df.columns:
                inflow_df['timestamp'] = pd.to_datetime(inflow_df['t'], unit='s')
                inflow_df = inflow_df.drop('t', axis=1)
            
            if 't' in outflow_df.columns:
                outflow_df['timestamp'] = pd.to_datetime(outflow_df['t'], unit='s')
                outflow_df = outflow_df.drop('t', axis=1)
            
            # 转换值为数值类型
            if 'v' in inflow_df.columns:
                inflow_df['inflow'] = pd.to_numeric(inflow_df['v'], errors='coerce')
                inflow_df = inflow_df.drop('v', axis=1)
            
            if 'v' in outflow_df.columns:
                outflow_df['outflow'] = pd.to_numeric(outflow_df['v'], errors='coerce')
                outflow_df = outflow_df.drop('v', axis=1)
            
            # 合并流入和流出数据
            if not inflow_df.empty and not outflow_df.empty:
                flow_df = pd.merge(inflow_df, outflow_df, on='timestamp', how='outer')
                
                # 计算净流入
                flow_df['net_flow'] = flow_df['inflow'] - flow_df['outflow']
                
                # 缓存数据
                self._save_to_cache(flow_df, "exchange_flow", cache_params)
                
                return flow_df
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取Glassnode交易所资金流向数据失败 - 币种: {coin}, 错误: {e}")
            return pd.DataFrame()
    
    def get_social_sentiment(self, coin: str, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        获取社交媒体情绪数据
        
        注意：Glassnode API不直接提供社交媒体情绪数据，此方法仅作为示例
        """
        logger.warning("Glassnode API不直接提供社交媒体情绪数据，请使用专门的社交媒体数据源")
        return pd.DataFrame()


class LunarCrushDataSource(CryptoDataSource):
    """基于LunarCrush API的数据源实现，用于获取社交媒体情绪数据"""
    
    def __init__(self, config: Dict = None):
        """
        初始化LunarCrush数据源
        
        参数:
            config: 配置信息，必须包含api_key
        """
        super().__init__(config)
        self.name = "LunarCrush数据源"
        self.base_url = "https://api.lunarcrush.com/v2"
        
        # 检查配置中是否包含API密钥
        if not config or 'api_key' not in config:
            logger.error("未提供LunarCrush API密钥，无法使用此数据源")
            raise ValueError("LunarCrush数据源需要API密钥")
        
        self.api_key = config['api_key']
        logger.info("LunarCrush数据源初始化成功")
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        发送请求到LunarCrush API
        
        参数:
            endpoint: API端点
            params: 请求参数
            
        返回:
            响应数据字典
        """
        url = f"{self.base_url}{endpoint}"
        
        # 添加API密钥
        if params is None:
            params = {}
        params['key'] = self.api_key
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # 如果响应状态码不是200，抛出异常
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"请求LunarCrush API失败: {e}")
            return {}
    
    def get_coin_list(self) -> pd.DataFrame:
        """获取LunarCrush支持的所有加密货币"""
        cache_params = {"type": "coin_list"}
        cached_data = self._load_from_cache("coin_list", cache_params, cache_hours=24)  # 币种列表缓存24小时
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 获取币种列表
            endpoint = "/assets"
            params = {'data': 'market'}
            coin_list_data = self._make_request(endpoint, params)
            
            if not coin_list_data or 'data' not in coin_list_data:
                logger.error("获取LunarCrush币种列表失败")
                return pd.DataFrame(columns=['id', 'symbol', 'name'])
            
            # 提取币种数据
            coins = coin_list_data['data']
            
            # 转换为DataFrame
            coin_list = pd.DataFrame(coins)
            
            # 缓存数据
            self._save_to_cache(coin_list, "coin_list", cache_params)
            
            return coin_list
        except Exception as e:
            logger.error(f"获取LunarCrush币种列表失败: {e}")
            return pd.DataFrame(columns=['id', 'symbol', 'name'])
    
    def get_kline_data(self, symbol: str, interval: str, start_time: int = None, end_time: int = None, limit: int = 500) -> pd.DataFrame:
        """
        获取K线数据
        
        注意：LunarCrush API不直接提供K线数据，此方法仅作为示例
        """
        logger.warning("LunarCrush API不直接提供K线数据，请使用交易所数据源")
        return pd.DataFrame()
    
    def get_ticker(self, symbol: str = None) -> pd.DataFrame:
        """
        获取最新行情数据
        
        注意：LunarCrush API不直接提供行情数据，此方法仅作为示例
        """
        logger.warning("LunarCrush API不直接提供行情数据，请使用交易所数据源")
        return pd.DataFrame()
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """
        获取订单簿数据
        
        注意：LunarCrush API不提供订单簿数据，此方法仅作为示例
        """
        logger.warning("LunarCrush API不提供订单簿数据")
        return {'bids': [], 'asks': []}
    
    def get_whale_transactions(self, coin: str, min_value_usd: float = 1000000, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        获取大额交易数据
        
        注意：LunarCrush API不提供链上大额交易数据，此方法仅作为示例
        """
        logger.warning("LunarCrush API不提供链上大额交易数据")
        return pd.DataFrame()
    
    def get_exchange_flow(self, coin: str, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        获取交易所资金流向数据
        
        注意：LunarCrush API不提供交易所资金流向数据，此方法仅作为示例
        """
        logger.warning("LunarCrush API不提供交易所资金流向数据")
        return pd.DataFrame()
    
    def get_social_sentiment(self, coin: str, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        获取社交媒体情绪数据
        
        参数:
            coin: 币种，如'BTC'
            start_time: 开始时间戳（秒）
            end_time: 结束时间戳（秒）
            
        返回:
            社交媒体情绪数据DataFrame
        """
        # 标准化币种符号
        coin = coin.upper()
        
        cache_params = {
            "coin": coin,
            "start_time": start_time or 0,
            "end_time": end_time or int(time.time())
        }
        
        cached_data = self._load_from_cache("social_sentiment", cache_params, cache_hours=1)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 准备请求参数
            params = {
                'symbol': coin,
                'data': 'social'
            }
            
            if start_time:
                params['start'] = start_time
            
            if end_time:
                params['end'] = end_time
            
            # 获取社交媒体情绪数据
            endpoint = "/assets"
            sentiment_data = self._make_request(endpoint, params)
            
            if not sentiment_data or 'data' not in sentiment_data or not sentiment_data['data']:
                logger.error(f"获取LunarCrush社交媒体情绪数据失败 - 币种: {coin}")
                return pd.DataFrame()
            
            # 提取社交媒体数据
            social_data = sentiment_data['data'][0]
            
            # 提取时间序列数据
            timeseries = social_data.get('timeSeries', [])
            
            if not timeseries:
                logger.error(f"LunarCrush社交媒体时间序列数据为空 - 币种: {coin}")
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame(timeseries)
            
            # 转换时间戳为日期时间
            if 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            
            # 缓存数据
            self._save_to_cache(df, "social_sentiment", cache_params)
            
            return df
        except Exception as e:
            logger.error(f"获取LunarCrush社交媒体情绪数据失败 - 币种: {coin}, 错误: {e}")
            return pd.DataFrame()


class CryptoDataManager:
    """加密货币数据管理器，整合多个数据源"""
    
    def __init__(self, config: Dict = None):
        """
        初始化数据管理器
        
        参数:
            config: 配置信息，包含各数据源的配置
        """
        self.config = config or {}
        self.data_sources = {}
        self.default_source = None
        
        # 初始化日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化数据源
        self._init_data_sources()
    
    def _init_data_sources(self):
        """初始化所有配置的数据源"""
        # 尝试初始化Binance数据源
        try:
            binance_config = self.config.get('binance', {})
            self.data_sources['binance'] = BinanceDataSource(binance_config)
            if self.default_source is None:
                self.default_source = 'binance'
            self.logger.info("Binance数据源初始化成功")
        except Exception as e:
            self.logger.warning(f"Binance数据源初始化失败: {e}")
        
        # 尝试初始化CoinGecko数据源
        try:
            coingecko_config = self.config.get('coingecko', {})
            self.data_sources['coingecko'] = CoinGeckoDataSource(coingecko_config)
            if self.default_source is None:
                self.default_source = 'coingecko'
            self.logger.info("CoinGecko数据源初始化成功")
        except Exception as e:
            self.logger.warning(f"CoinGecko数据源初始化失败: {e}")
        
        # 尝试初始化Glassnode数据源
        try:
            glassnode_config = self.config.get('glassnode', {})
            if glassnode_config and 'api_key' in glassnode_config:
                self.data_sources['glassnode'] = GlassnodeDataSource(glassnode_config)
                self.logger.info("Glassnode数据源初始化成功")
        except Exception as e:
            self.logger.warning(f"Glassnode数据源初始化失败: {e}")
        
        # 尝试初始化LunarCrush数据源
        try:
            lunarcrush_config = self.config.get('lunarcrush', {})
            if lunarcrush_config and 'api_key' in lunarcrush_config:
                self.data_sources['lunarcrush'] = LunarCrushDataSource(lunarcrush_config)
                self.logger.info("LunarCrush数据源初始化成功")
        except Exception as e:
            self.logger.warning(f"LunarCrush数据源初始化失败: {e}")
        
        if not self.data_sources:
            self.logger.error("没有可用的数据源，请检查配置和依赖安装")
            raise ValueError("没有可用的数据源")
    
    def get_data_source(self, source_name: str = None) -> CryptoDataSource:
        """
        获取指定的数据源
        
        参数:
            source_name: 数据源名称，如果为None则返回默认数据源
            
        返回:
            数据源对象
        """
        if source_name is None:
            source_name = self.default_source
        
        if source_name not in self.data_sources:
            self.logger.warning(f"数据源 {source_name} 不存在，使用默认数据源 {self.default_source}")
            source_name = self.default_source
        
        return self.data_sources[source_name]
    
    def get_coin_list(self, source_name: str = None) -> pd.DataFrame:
        """
        获取加密货币列表
        
        参数:
            source_name: 数据源名称，如果为None则使用默认数据源
            
        返回:
            加密货币列表DataFrame
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


# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 测试数据管理器
    config = {
        'binance': {},
        'coingecko': {},
        'glassnode': {'api_key': '你的Glassnode API密钥'},  # 替换为实际的API密钥
        'lunarcrush': {'api_key': '你的LunarCrush API密钥'}  # 替换为实际的API密钥
    }
    
    try:
        # 初始化数据管理器
        data_manager = CryptoDataManager(config)
        
        # 获取币种列表
        coin_list = data_manager.get_coin_list()
        print(f"币种列表前5行:\n{coin_list.head()}")
        
        # 获取K线数据
        kline_data = data_manager.get_kline_data('BTC/USDT', '1d', limit=10)
        print(f"\nK线数据前5行:\n{kline_data.head()}")
        
        # 获取行情数据
        ticker_data = data_manager.get_ticker('BTC/USDT')
        print(f"\n行情数据:\n{ticker_data.head()}")
        
        # 获取订单簿数据
        order_book = data_manager.get_order_book('BTC/USDT', limit=5)
        print(f"\n订单簿数据:\n{order_book}")
        
    except Exception as e:
        logging.error(f"测试过程中发生错误: {e}")
