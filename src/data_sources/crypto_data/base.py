"""
加密货币数据源基类

定义所有加密货币数据源需要实现的通用接口。

日期：2025-05-17
"""

import os
import logging
import time
import pandas as pd
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