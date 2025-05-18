"""
加密货币数据提供商

支持多种数据源访问的加密货币数据提供商实现，专为国内网络环境优化。
提供对多种交易所API的统一访问接口，包括Gate.io、火币、OKEx和币安。

日期：2025-05-17
"""

import requests
import pandas as pd
import time
import logging
from typing import Optional, Dict, Any, List
import numpy as np

from .base import CryptoDataSource

class CryptoDataProvider(CryptoDataSource):
    """加密货币数据提供商类，支持多种数据源访问"""
    
    def __init__(self, config: Dict = None):
        """
        初始化数据提供商
        
        参数:
            config: 配置字典，包含API设置和数据源选择
        """
        super().__init__(config)
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.api_source = self.config.get('api_source', 'gateio')  # 默认使用Gate.io
        self.name = f"多源数据提供商({self.api_source})"
        
        # 配置请求超时和重试参数
        self.timeout = self.config.get('request_timeout', 30)
        self.retry_count = self.config.get('retry_count', 3)
        self.retry_delay = self.config.get('retry_delay', 1)
        
        # 设置代理
        self.use_proxy = self.config.get('use_proxy', False)
        self.proxies = self.config.get('proxy', {})
        
        # 本地模式 - 当无法连接外部API时使用模拟数据
        self.local_mode = self.config.get('local_mode', False)
        self.local_data_cache = {}
    
    def _log_error(self, message: str):
        """记录错误日志"""
        self.logger.error(message)
    
    def get_kline_data(self, symbol: str, interval: str = '1h', 
                      start_time: int = None, end_time: int = None, 
                      limit: int = 100) -> Optional[pd.DataFrame]:
        """
        获取K线数据，重写父类方法
        
        参数:
            symbol: 交易对符号，如 'BTC_USDT'
            interval: K线间隔，如 '1m', '5m', '1h', '1d'等
            start_time: 开始时间戳（毫秒），可选
            end_time: 结束时间戳（毫秒），可选
            limit: 获取的K线数量
            
        返回:
            K线数据DataFrame
        """
        return self.get_market_data(symbol, interval, limit)
    
    def get_market_data(self, symbol: str, interval: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
        """
        通过选定的API获取市场数据
        
        参数:
            symbol: 交易对符号，如 'BTC_USDT'
            interval: K线间隔，如 '1m', '5m', '1h', '1d'等
            limit: 获取的K线数量
        
        返回:
            市场数据DataFrame或None（如果获取失败）
        """
        # 根据配置选择数据源
        if self.api_source == 'binance':
            return self._get_binance_data(symbol, interval, limit)
        elif self.api_source == 'huobi':
            return self._get_huobi_data(symbol, interval, limit)
        elif self.api_source == 'gateio':
            return self._get_gateio_data(symbol, interval, limit)
        elif self.api_source == 'okex':
            return self._get_okex_data(symbol, interval, limit)
        else:
            self._log_error(f"不支持的数据源: {self.api_source}")
            return None
    
    def _standardize_symbol(self, symbol: str, exchange: str) -> str:
        """
        将统一格式的交易对转换为各交易所支持的格式
        
        参数:
            symbol: 统一格式的交易对，如"BTC_USDT"
            exchange: 交易所名称
            
        返回:
            交易所特定格式的交易对符号
        """
        # 移除可能的下划线
        parts = symbol.replace('_', '').upper()
        
        if exchange == 'binance':
            return parts  # 币安使用无下划线格式，如 BTCUSDT
        elif exchange == 'huobi':
            return parts.lower()  # 火币使用小写无下划线，如 btcusdt
        elif exchange == 'gateio':
            # Gate.io使用下划线分隔，如 BTC_USDT
            if '_' not in symbol:
                # 假设最后4个或3个字符是计价货币（USDT、BTC等）
                if symbol.endswith(('USDT', 'BUSD')):
                    quote_len = 4
                elif symbol.endswith(('BTC', 'ETH', 'USD')):
                    quote_len = 3
                else:
                    quote_len = 4  # 默认假设是4位
                
                base = symbol[:-quote_len]
                quote = symbol[-quote_len:]
                return f"{base}_{quote}"
            return symbol.upper()
        elif exchange == 'okex':
            # OKEx使用'-'分隔，如 BTC-USDT
            if '_' in symbol:
                return symbol.replace('_', '-').upper()
            elif '-' not in symbol:
                # 类似Gate.io的处理
                if symbol.endswith(('USDT', 'BUSD')):
                    quote_len = 4
                elif symbol.endswith(('BTC', 'ETH', 'USD')):
                    quote_len = 3
                else:
                    quote_len = 4
                
                base = symbol[:-quote_len]
                quote = symbol[-quote_len:]
                return f"{base}-{quote}"
            return symbol.upper()
        
        # 默认返回原格式
        return symbol
    
    def _standardize_interval(self, interval: str, exchange: str) -> str:
        """
        将统一的时间间隔格式转换为交易所特定格式
        
        参数:
            interval: 统一格式的时间间隔
            exchange: 交易所名称
            
        返回:
            交易所特定格式的时间间隔
        """
        # 通用转换字典 - 为不同交易所的间隔格式
        exchange_intervals = {
            'binance': {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '12h': '12h', '1d': '1d', '1w': '1w'
            },
            'huobi': {
                '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
                '1h': '60min', '4h': '4hour', '1d': '1day', '1w': '1week'
            },
            'gateio': {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '12h': '12h', '1d': '1d', '1w': '7d'
            },
            'okex': {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1H', '4h': '4H', '12h': '12H', '1d': '1D', '1w': '1W'
            }
        }
        
        # 获取当前交易所的时间间隔映射
        intervals = exchange_intervals.get(exchange, {})
        # 返回映射后的时间间隔，如果没有对应的映射则返回原值
        return intervals.get(interval, interval)
    
    def _get_with_retry(self, url, params=None, headers=None):
        """
        带重试机制的GET请求
        
        参数:
            url: 请求URL
            params: 请求参数
            headers: 请求头
            
        返回:
            响应对象
        """
        for i in range(self.retry_count):
            try:
                # 配置代理
                request_kwargs = {
                    'timeout': self.timeout,
                    'params': params,
                    'headers': headers
                }
                
                if self.use_proxy and self.proxies:
                    request_kwargs['proxies'] = self.proxies
                
                response = requests.get(url, **request_kwargs)
                return response
            except Exception as e:
                self._log_error(f"请求失败(尝试 {i+1}/{self.retry_count}): {str(e)}")
                if i < self.retry_count - 1:  # 如果不是最后一次尝试，则等待后重试
                    time.sleep(self.retry_delay)
        
        # 如果所有尝试都失败
        raise Exception(f"在 {self.retry_count} 次尝试后请求仍然失败")
    
    def _get_binance_data(self, symbol: str, interval: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
        """
        使用币安公共API获取市场数据
        
        参数:
            symbol: 交易对符号，如 'BTC_USDT'
            interval: K线间隔，如 '1m', '5m', '1h', '1d'等
            limit: 获取的K线数量，最大1000
        
        返回:
            市场数据DataFrame
        """
        try:
            # 转换为币安格式
            binance_symbol = self._standardize_symbol(symbol, 'binance')
            binance_interval = self._standardize_interval(interval, 'binance')
            
            # 构建公共API请求URL
            base_url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': binance_symbol,
                'interval': binance_interval,
                'limit': limit
            }
            
            # 发送请求
            response = self._get_with_retry(base_url, params=params)
            
            if response.status_code == 200:
                # 处理返回数据
                data = response.json()
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                               'close_time', 'quote_asset_volume', 'number_of_trades', 
                                               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
                
                # 格式化时间戳
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # 转换数值列
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                    
                return df
            else:
                self._log_error(f"获取币安API数据失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self._log_error(f"获取币安API数据异常: {str(e)}")
            return None
    
    def _get_huobi_data(self, symbol: str, interval: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
        """
        使用火币公共API获取市场数据
        
        参数:
            symbol: 交易对符号，如 'BTC_USDT'
            interval: K线间隔，如 '1m', '5m', '1h', '1d'等
            limit: 获取的K线数量
        
        返回:
            市场数据DataFrame
        """
        try:
            # 转换为火币格式
            huobi_symbol = self._standardize_symbol(symbol, 'huobi')
            huobi_interval = self._standardize_interval(interval, 'huobi')
            
            # 构建公共API请求URL
            base_url = "https://api.huobi.pro/market/history/kline"
            params = {
                'symbol': huobi_symbol,
                'period': huobi_interval,
                'size': limit
            }
            
            # 发送请求
            response = self._get_with_retry(base_url, params=params)
            
            if response.status_code == 200:
                # 处理返回数据
                data = response.json()
                
                if data['status'] == 'ok':
                    klines = data['data']
                    df = pd.DataFrame(klines)
                    
                    # 重命名列以符合标准格式
                    df.rename(columns={
                        'id': 'timestamp',
                        'amount': 'volume',
                        'vol': 'quote_volume',
                        'count': 'number_of_trades'
                    }, inplace=True)
                    
                    # 格式化时间戳
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    
                    return df
                else:
                    self._log_error(f"获取火币API数据失败: {data['err-msg']}")
                    return None
            else:
                self._log_error(f"获取火币API数据失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self._log_error(f"获取火币API数据异常: {str(e)}")
            return None
    
    def _get_gateio_data(self, symbol: str, interval: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
        """
        使用Gate.io公共API获取市场数据
        
        参数:
            symbol: 交易对符号，如 'BTC_USDT'
            interval: K线间隔，如 '1m', '5m', '1h', '1d'等
            limit: 获取的K线数量
        
        返回:
            市场数据DataFrame
        """
        try:
            # 转换为Gate.io格式
            gateio_symbol = self._standardize_symbol(symbol, 'gateio')
            gateio_interval = self._standardize_interval(interval, 'gateio')
            
            # 构建公共API请求URL - Gate.io V4 API
            base_url = "https://api.gateio.ws/api/v4/spot/candlesticks"
            params = {
                'currency_pair': gateio_symbol,
                'interval': gateio_interval,
                'limit': limit
            }
            
            # 发送请求
            response = self._get_with_retry(base_url, params=params)
            
            if response.status_code == 200:
                # 处理返回数据
                data = response.json()
                
                # 检查API文档，确认返回的列顺序
                # Gate.io API 文档: https://www.gate.io/docs/developers/apiv4/
                # 返回字段: [timestamp, volume, close, high, low, open, amount, quote_volume]
                # 动态创建DataFrame并根据API返回的所有列创建
                if data and len(data) > 0:
                    column_names = ['timestamp', 'volume', 'close', 'high', 'low', 'open']
                    
                    # 检查返回的数据有多少列，并相应地调整列名
                    if len(data[0]) > 6:
                        # 如果有更多列，添加额外的列名
                        extra_columns = [f'extra_{i}' for i in range(len(data[0]) - 6)]
                        column_names.extend(extra_columns)
                    
                    df = pd.DataFrame(data, columns=column_names)
                    
                    # 格式化时间戳
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    
                    # 转换数值列
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in df.columns:
                            df[col] = df[col].astype(float)
                    
                    # 重新排序列以符合标准格式
                    result_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    available_columns = [col for col in result_columns if col in df.columns]
                    
                    return df[available_columns]
                else:
                    self._log_error(f"Gate.io API返回空数据")
                    return None
            else:
                self._log_error(f"获取Gate.io API数据失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self._log_error(f"获取Gate.io API数据异常: {str(e)}")
            return None
    
    def _get_okex_data(self, symbol: str, interval: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
        """
        使用OKEx公共API获取市场数据
        
        参数:
            symbol: 交易对符号，如 'BTC_USDT'
            interval: K线间隔，如 '1m', '5m', '1h', '1d'等
            limit: 获取的K线数量
        
        返回:
            市场数据DataFrame
        """
        try:
            # 转换为OKEx格式
            okex_symbol = self._standardize_symbol(symbol, 'okex')
            okex_interval = self._standardize_interval(interval, 'okex')
            
            # 构建公共API请求URL - OKEx V5 API
            base_url = "https://www.okx.com/api/v5/market/candles"
            params = {
                'instId': okex_symbol,
                'bar': okex_interval,
                'limit': limit
            }
            
            # 发送请求
            response = self._get_with_retry(base_url, params=params)
            
            if response.status_code == 200:
                # 处理返回数据
                data = response.json()
                
                if data['code'] == '0':
                    # OKEx返回的数据:
                    # [时间戳, 开盘价, 最高价, 最低价, 收盘价, 交易量, 交易额]
                    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']
                    df = pd.DataFrame(data['data'], columns=columns)
                    
                    # 格式化时间戳 (okex使用ISO格式时间戳)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # 转换数值列
                    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
                        df[col] = df[col].astype(float)
                    
                    return df
                else:
                    self._log_error(f"获取OKEx API数据失败: {data['msg']}")
                    return None
            else:
                self._log_error(f"获取OKEx API数据失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self._log_error(f"获取OKEx API数据异常: {str(e)}")
            return None

    def get_public_market_data(self, symbol, interval='1h', limit=100):
        """
        兼容旧接口的封装方法
        """
        return self.get_market_data(symbol, interval, limit)

    # 添加兼容web_controller的方法
    def get_klines(self, symbol, interval='1h', start_time=None, end_time=None):
        """
        兼容旧接口的get_klines方法
        
        参数:
            symbol: 交易对符号
            interval: 时间间隔
            start_time: 开始时间(不使用)
            end_time: 结束时间(不使用)
            
        返回:
            市场数据DataFrame，或者出错时返回特殊错误对象
        """
        try:
            # 忽略start_time和end_time，因为我们的API不支持时间范围查询
            limit = 200  # 使用较大的默认值
            df = self.get_market_data(symbol, interval, limit)
            
            # 检查数据框是否为空
            if df is None or df.empty:
                # 返回包含错误信息的结构，让API层能够识别这是一个错误
                self.logger.error(f"获取{symbol}的K线数据失败，API返回为空")
                raise Exception(f"获取{symbol}数据失败: API返回空数据")
            
            return df
        except Exception as e:
            self.logger.error(f"获取K线数据出错: {e}")
            # 异常时不返回DataFrame，而是抛出异常让上层处理
            raise Exception(f"获取{symbol}数据失败: {str(e)}")
    
    def search_cryptos(self, keyword):
        """
        搜索加密货币
        
        参数:
            keyword: 搜索关键词
            
        返回:
            符合条件的加密货币列表
        """
        # 常见加密货币的图标URL，使用可靠的CDN
        icons = {
            "BTC": "https://s2.coinmarketcap.com/static/img/coins/64x64/1.png",  # 比特币
            "ETH": "https://s2.coinmarketcap.com/static/img/coins/64x64/1027.png",  # 以太坊
            "BNB": "https://s2.coinmarketcap.com/static/img/coins/64x64/1839.png",  # 币安币
            "XRP": "https://s2.coinmarketcap.com/static/img/coins/64x64/52.png",   # 瑞波币
            "ADA": "https://s2.coinmarketcap.com/static/img/coins/64x64/2010.png", # 艾达币
            "SOL": "https://s2.coinmarketcap.com/static/img/coins/64x64/5426.png", # 索拉纳
            "DOT": "https://s2.coinmarketcap.com/static/img/coins/64x64/6636.png", # 波卡
            "DOGE": "https://s2.coinmarketcap.com/static/img/coins/64x64/74.png",  # 狗狗币
            "USDT": "https://s2.coinmarketcap.com/static/img/coins/64x64/825.png", # 泰达币
            "SHIB": "https://s2.coinmarketcap.com/static/img/coins/64x64/5994.png", # 柴犬币
            # 添加其他主要币种
            "AVAX": "https://s2.coinmarketcap.com/static/img/coins/64x64/5805.png", # 雪崩
            "LINK": "https://s2.coinmarketcap.com/static/img/coins/64x64/1975.png", # 链接
            "MATIC": "https://s2.coinmarketcap.com/static/img/coins/64x64/3890.png", # Polygon
            "LTC": "https://s2.coinmarketcap.com/static/img/coins/64x64/2.png",    # 莱特币
            "UNI": "https://s2.coinmarketcap.com/static/img/coins/64x64/7083.png"  # Uniswap
        }
        
        # 主流币种列表
        popular_cryptos = [
            {"symbol": "BTC_USDT", "name": "Bitcoin", "id": 1},
            {"symbol": "ETH_USDT", "name": "Ethereum", "id": 1027},
            {"symbol": "BNB_USDT", "name": "Binance Coin", "id": 1839},
            {"symbol": "SOL_USDT", "name": "Solana", "id": 5426},
            {"symbol": "XRP_USDT", "name": "XRP", "id": 52},
            {"symbol": "ADA_USDT", "name": "Cardano", "id": 2010},
            {"symbol": "DOGE_USDT", "name": "Dogecoin", "id": 74},
            {"symbol": "DOT_USDT", "name": "Polkadot", "id": 6636},
            {"symbol": "AVAX_USDT", "name": "Avalanche", "id": 5805},
            {"symbol": "MATIC_USDT", "name": "Polygon", "id": 3890}
        ]
        
        # 添加更多币种
        more_cryptos = [
            {"symbol": "LINK_USDT", "name": "Chainlink", "id": 1975},
            {"symbol": "UNI_USDT", "name": "Uniswap", "id": 7083},
            {"symbol": "ATOM_USDT", "name": "Cosmos", "id": 3794},
            {"symbol": "LTC_USDT", "name": "Litecoin", "id": 2},
            {"symbol": "ALGO_USDT", "name": "Algorand", "id": 4030},
            {"symbol": "FIL_USDT", "name": "Filecoin", "id": 2280},
            {"symbol": "ETC_USDT", "name": "Ethereum Classic", "id": 1321},
            {"symbol": "NEAR_USDT", "name": "NEAR Protocol", "id": 6535},
            {"symbol": "LUNA_USDT", "name": "Terra Luna", "id": 4172},
            {"symbol": "FTM_USDT", "name": "Fantom", "id": 3513}
        ]
        
        all_cryptos = popular_cryptos + more_cryptos
        
        # 为每个币种添加图标URL
        for crypto in all_cryptos:
            # CoinMarketCap图标URL
            crypto["image"] = f"https://s2.coinmarketcap.com/static/img/coins/64x64/{crypto['id']}.png"
            # 备用图标URL - 从github获取
            crypto["fallback_image"] = f"https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1.0.0/128/color/{crypto['symbol'].split('_')[0].lower()}.png"
            # 中国可访问的图标URL - 使用国内CDN
            crypto["cn_image"] = f"https://crypto-icons.oss-cn-hongkong.aliyuncs.com/{crypto['symbol'].split('_')[0].lower()}.png"
        
        if not keyword:
            return popular_cryptos
            
        # 根据关键词过滤
        keyword = keyword.lower()
        return [
            crypto for crypto in all_cryptos 
            if keyword in crypto["symbol"].lower() or keyword in crypto["name"].lower()
        ]
    
    def get_technical_indicators(self, symbol, interval='1h', start_time=None, end_time=None):
        """
        获取技术指标
        
        参数:
            symbol: 交易对符号
            interval: 时间间隔
            start_time: 开始时间(不使用)
            end_time: 结束时间(不使用)
            
        返回:
            技术指标DataFrame
        """
        try:
            # 获取市场数据
            market_data = self.get_market_data(symbol, interval, 100)
            if market_data is None or market_data.empty:
                return pd.DataFrame()
                
            # 计算基本技术指标
            import talib
            import numpy as np
            
            close = market_data['close'].values
            high = market_data['high'].values
            low = market_data['low'].values
            volume = market_data['volume'].values
            
            # 创建结果DataFrame
            result = pd.DataFrame(index=market_data.index)
            
            # 计算移动平均线
            result['MA5'] = talib.MA(close, timeperiod=5)
            result['MA10'] = talib.MA(close, timeperiod=10)
            result['MA20'] = talib.MA(close, timeperiod=20)
            result['MA60'] = talib.MA(close, timeperiod=60)
            
            # 计算MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            result['MACD'] = macd
            result['MACD_SIGNAL'] = macd_signal
            result['MACD_HIST'] = macd_hist
            
            # 计算RSI
            result['RSI'] = talib.RSI(close, timeperiod=14)
            
            # 计算布林带
            upper, middle, lower = talib.BBANDS(close, timeperiod=20)
            result['BOLL_UPPER'] = upper
            result['BOLL_MIDDLE'] = middle
            result['BOLL_LOWER'] = lower
            
            # 计算KDJ
            result['K'], result['D'] = talib.STOCH(high, low, close)
            result['J'] = 3 * result['K'] - 2 * result['D']
            
            return result
        except Exception as e:
            self._log_error(f"计算技术指标异常: {str(e)}")
            return pd.DataFrame()
    
    def get_order_book(self, symbol, limit=100):
        """
        获取订单簿数据
        
        参数:
            symbol: 交易对符号
            limit: 数量限制
            
        返回:
            订单簿数据字典
        """
        try:
            # 根据数据源选择不同的实现
            if self.api_source == 'binance':
                return self._get_binance_order_book(symbol, limit)
            elif self.api_source == 'huobi':
                return self._get_huobi_order_book(symbol, limit)
            elif self.api_source == 'gateio':
                return self._get_gateio_order_book(symbol, limit)
            elif self.api_source == 'okex':
                return self._get_okex_order_book(symbol, limit)
            else:
                self._log_error(f"不支持的数据源: {self.api_source}")
                return {"bids": [], "asks": []}
        except Exception as e:
            self._log_error(f"获取订单簿异常: {str(e)}")
            return {"bids": [], "asks": []}
    
    def _get_binance_order_book(self, symbol, limit=100):
        """获取币安订单簿"""
        try:
            # 转换为币安格式
            binance_symbol = self._standardize_symbol(symbol, 'binance')
            
            # 构建API请求URL
            base_url = "https://api.binance.com/api/v3/depth"
            params = {
                'symbol': binance_symbol,
                'limit': min(limit, 1000)  # 币安最大限制为1000
            }
            
            # 发送请求
            response = self._get_with_retry(base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # 处理返回数据
                bids = [[float(price), float(qty)] for price, qty in data['bids']]
                asks = [[float(price), float(qty)] for price, qty in data['asks']]
                
                return {
                    "bids": bids,
                    "asks": asks
                }
            else:
                self._log_error(f"获取币安订单簿失败: {response.status_code} - {response.text}")
                return {"bids": [], "asks": []}
                
        except Exception as e:
            self._log_error(f"获取币安订单簿异常: {str(e)}")
            return {"bids": [], "asks": []}
    
    def _get_huobi_order_book(self, symbol, limit=100):
        """获取火币订单簿"""
        try:
            # 转换为火币格式
            huobi_symbol = self._standardize_symbol(symbol, 'huobi')
            
            # 构建API请求URL
            base_url = "https://api.huobi.pro/market/depth"
            params = {
                'symbol': huobi_symbol,
                'type': 'step0',  # 获取完整深度
                'depth': min(limit, 150)  # 火币最大限制不同
            }
            
            # 发送请求
            response = self._get_with_retry(base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] == 'ok':
                    # 处理返回数据
                    bids = [[float(item[0]), float(item[1])] for item in data['tick']['bids']]
                    asks = [[float(item[0]), float(item[1])] for item in data['tick']['asks']]
                    
                    return {
                        "bids": bids[:limit],
                        "asks": asks[:limit]
                    }
                else:
                    self._log_error(f"获取火币订单簿失败: {data['err-msg']}")
                    return {"bids": [], "asks": []}
            else:
                self._log_error(f"获取火币订单簿失败: {response.status_code} - {response.text}")
                return {"bids": [], "asks": []}
                
        except Exception as e:
            self._log_error(f"获取火币订单簿异常: {str(e)}")
            return {"bids": [], "asks": []}
    
    def _get_gateio_order_book(self, symbol, limit=100):
        """获取Gate.io订单簿"""
        try:
            # 转换为Gate.io格式
            gateio_symbol = self._standardize_symbol(symbol, 'gateio')
            
            # 构建API请求URL
            base_url = "https://api.gateio.ws/api/v4/spot/order_book"
            params = {
                'currency_pair': gateio_symbol,
                'limit': min(limit, 100),  # Gate.io默认限制为100
                'with_id': 'true'
            }
            
            # 发送请求
            response = self._get_with_retry(base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # 处理返回数据
                bids = [[float(item[0]), float(item[1])] for item in data['bids']]
                asks = [[float(item[0]), float(item[1])] for item in data['asks']]
                
                return {
                    "bids": bids,
                    "asks": asks
                }
            else:
                self._log_error(f"获取Gate.io订单簿失败: {response.status_code} - {response.text}")
                return {"bids": [], "asks": []}
                
        except Exception as e:
            self._log_error(f"获取Gate.io订单簿异常: {str(e)}")
            return {"bids": [], "asks": []}
    
    def _get_okex_order_book(self, symbol, limit=100):
        """获取OKEx订单簿"""
        try:
            # 转换为OKEx格式
            okex_symbol = self._standardize_symbol(symbol, 'okex')
            
            # 构建API请求URL
            base_url = "https://www.okx.com/api/v5/market/books"
            params = {
                'instId': okex_symbol,
                'sz': min(limit, 400)  # OKEx最大限制为400
            }
            
            # 发送请求
            response = self._get_with_retry(base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['code'] == '0':
                    # 处理返回数据
                    book_data = data['data'][0]
                    bids = [[float(item[0]), float(item[1])] for item in book_data['bids']]
                    asks = [[float(item[0]), float(item[1])] for item in book_data['asks']]
                    
                    return {
                        "bids": bids,
                        "asks": asks
                    }
                else:
                    self._log_error(f"获取OKEx订单簿失败: {data['msg']}")
                    return {"bids": [], "asks": []}
            else:
                self._log_error(f"获取OKEx订单簿失败: {response.status_code} - {response.text}")
                return {"bids": [], "asks": []}
                
        except Exception as e:
            self._log_error(f"获取OKEx订单簿异常: {str(e)}")
            return {"bids": [], "asks": []} 