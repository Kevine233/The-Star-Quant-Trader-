"""
股票市场数据获取模块

本模块负责从多个数据源获取股票市场数据，提供统一的接口。

日期：2025-05-17
"""

from .base import StockDataSource
from .akshare import AKShareDataSource
from .tushare import TushareDataSource
from .baostock import BaostockDataSource
from .manager import StockDataManager

__all__ = [
    'StockDataSource',
    'AKShareDataSource',
    'TushareDataSource',
    'BaostockDataSource',
    'StockDataManager'
] 