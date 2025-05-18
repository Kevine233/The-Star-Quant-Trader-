"""
数据源基类模块，定义了所有数据源的通用接口。
本模块提供了数据源的抽象基类，所有具体的数据源实现都应该继承这个类。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import datetime
import logging
import time
import json
import os
import requests
from abc import ABC, abstractmethod

# 配置日志
logger = logging.getLogger(__name__)

class 数据源基类(ABC):
    """
    数据源抽象基类，定义了所有数据源的通用方法。
    所有具体的数据源实现都应该继承这个类。
    """
    
    def __init__(self, 配置: Dict[str, Any] = None):
        """
        初始化数据源。
        
        参数:
            配置: 数据源配置字典
        """
        self.配置 = 配置 if 配置 is not None else {}
        self.已连接 = False
        self.缓存 = {}
    
    @abstractmethod
    def 连接(self) -> bool:
        """
        连接到数据源。
        
        返回:
            连接是否成功
        """
        pass
    
    @abstractmethod
    def 断开连接(self) -> bool:
        """
        断开与数据源的连接。
        
        返回:
            断开连接是否成功
        """
        pass
    
    @abstractmethod
    def 获取数据(self, 参数: Dict[str, Any]) -> pd.DataFrame:
        """
        获取数据。
        
        参数:
            参数: 获取数据的参数字典
            
        返回:
            包含数据的DataFrame
        """
        pass
    
    def 是否已连接(self) -> bool:
        """
        检查是否已连接到数据源。
        
        返回:
            是否已连接
        """
        return self.已连接
    
    def 清除缓存(self):
        """清除数据缓存。"""
        self.缓存 = {}
        logger.info("数据缓存已清除")
    
    def _缓存数据(self, 键: str, 数据: pd.DataFrame):
        """
        缓存数据。
        
        参数:
            键: 缓存键
            数据: 要缓存的数据
        """
        self.缓存[键] = {
            '数据': 数据,
            '时间戳': datetime.datetime.now()
        }
    
    def _获取缓存数据(self, 键: str, 过期时间: Optional[datetime.timedelta] = None) -> Optional[pd.DataFrame]:
        """
        获取缓存数据。
        
        参数:
            键: 缓存键
            过期时间: 缓存过期时间，如果为None则不检查过期
            
        返回:
            缓存的数据，如果没有缓存或缓存已过期则返回None
        """
        if 键 not in self.缓存:
            return None
        
        缓存项 = self.缓存[键]
        
        if 过期时间 is not None:
            当前时间 = datetime.datetime.now()
            if 当前时间 - 缓存项['时间戳'] > 过期时间:
                return None
        
        return 缓存项['数据']


class 数据管理器:
    """
    数据管理器类，负责管理多个数据源并提供统一的数据访问接口。
    """
    
    def __init__(self):
        """初始化数据管理器。"""
        self.数据源 = {}
        self.数据清洗器 = None
    
    def 注册数据源(self, 名称: str, 数据源: 数据源基类):
        """
        注册数据源。
        
        参数:
            名称: 数据源名称
            数据源: 数据源实例
        """
        self.数据源[名称] = 数据源
        logger.info(f"已注册数据源: {名称}")
    
    def 注销数据源(self, 名称: str) -> bool:
        """
        注销数据源。
        
        参数:
            名称: 数据源名称
            
        返回:
            注销是否成功
        """
        if 名称 in self.数据源:
            数据源 = self.数据源[名称]
            if 数据源.是否已连接():
                数据源.断开连接()
            del self.数据源[名称]
            logger.info(f"已注销数据源: {名称}")
            return True
        else:
            logger.warning(f"数据源不存在: {名称}")
            return False
    
    def 获取数据源(self, 名称: str) -> Optional[数据源基类]:
        """
        获取数据源实例。
        
        参数:
            名称: 数据源名称
            
        返回:
            数据源实例，如果不存在则返回None
        """
        return self.数据源.get(名称)
    
    def 获取所有数据源(self) -> Dict[str, 数据源基类]:
        """
        获取所有数据源。
        
        返回:
            数据源字典，键为数据源名称，值为数据源实例
        """
        return self.数据源
    
    def 设置数据清洗器(self, 数据清洗器):
        """
        设置数据清洗器。
        
        参数:
            数据清洗器: 数据清洗器实例
        """
        self.数据清洗器 = 数据清洗器
        logger.info("已设置数据清洗器")
    
    def 获取数据(self, 数据源名称: str, 参数: Dict[str, Any], 清洗: bool = True) -> pd.DataFrame:
        """
        从指定数据源获取数据，并可选择是否进行清洗。
        
        参数:
            数据源名称: 数据源名称
            参数: 获取数据的参数字典
            清洗: 是否对数据进行清洗
            
        返回:
            包含数据的DataFrame
            
        异常:
            ValueError: 如果数据源不存在
        """
        if 数据源名称 not in self.数据源:
            raise ValueError(f"数据源不存在: {数据源名称}")
        
        数据源 = self.数据源[数据源名称]
        
        # 确保数据源已连接
        if not 数据源.是否已连接():
            连接成功 = 数据源.连接()
            if not 连接成功:
                raise ConnectionError(f"无法连接到数据源: {数据源名称}")
        
        # 获取数据
        数据 = 数据源.获取数据(参数)
        
        # 如果需要清洗且设置了数据清洗器，则进行数据清洗
        if 清洗 and self.数据清洗器 is not None:
            数据 = self.数据清洗器.清洗数据(数据)
        
        return 数据
    
    def 一键获取数据(self, 
                 市场类型: str, 
                 数据类型: str, 
                 开始日期: str, 
                 结束日期: str, 
                 标的列表: List[str] = None, 
                 清洗: bool = True) -> pd.DataFrame:
        """
        一键获取指定市场和类型的数据，自动选择合适的数据源。
        
        参数:
            市场类型: 市场类型，'A股'或'虚拟货币'
            数据类型: 数据类型，如'日线'、'分钟线'、'龙虎榜'等
            开始日期: 开始日期，格式为'YYYY-MM-DD'
            结束日期: 结束日期，格式为'YYYY-MM-DD'
            标的列表: 标的代码列表，如果为None则获取所有可用标的
            清洗: 是否对数据进行清洗
            
        返回:
            包含数据的DataFrame
            
        异常:
            ValueError: 如果没有合适的数据源
        """
        # 根据市场类型和数据类型选择合适的数据源
        合适数据源 = self._选择数据源(市场类型, 数据类型)
        
        if not 合适数据源:
            raise ValueError(f"没有合适的数据源用于获取 {市场类型} 的 {数据类型} 数据")
        
        # 构建参数
        参数 = {
            '开始日期': 开始日期,
            '结束日期': 结束日期,
            '数据类型': 数据类型
        }
        
        if 标的列表:
            参数['标的列表'] = 标的列表
        
        # 获取数据
        return self.获取数据(合适数据源, 参数, 清洗)
    
    def _选择数据源(self, 市场类型: str, 数据类型: str) -> Optional[str]:
        """
        根据市场类型和数据类型选择合适的数据源。
        
        参数:
            市场类型: 市场类型，'A股'或'虚拟货币'
            数据类型: 数据类型，如'日线'、'分钟线'、'龙虎榜'等
            
        返回:
            合适的数据源名称，如果没有合适的数据源则返回None
        """
        # 这里实现数据源选择逻辑
        # 可以根据数据源的优先级、可用性等因素进行选择
        
        for 名称, 数据源 in self.数据源.items():
            # 检查数据源是否支持指定的市场类型和数据类型
            # 这里假设数据源配置中包含了支持的市场类型和数据类型信息
            支持市场 = 数据源.配置.get('支持市场', [])
            支持数据类型 = 数据源.配置.get('支持数据类型', [])
            
            if 市场类型 in 支持市场 and 数据类型 in 支持数据类型:
                return 名称
        
        return None
