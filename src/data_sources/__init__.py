"""
"跟随庄家"自动交易系统的数据源模块。
本模块负责从各种来源获取A股和虚拟货币市场的数据。
"""

from abc import ABC, abstractmethod
import os
import json
import datetime
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import logging

# 配置日志
logger = logging.getLogger(__name__)

# 导入数据管理器
from .data_manager import 数据管理器, 数据源基类

# 导入加密货币数据源
from .crypto_data.provider import CryptoDataProvider

# 单例数据管理器实例
global_data_manager = None

def get_data_manager():
    """
    获取全局数据管理器实例。
    
    返回:
        数据管理器实例
    """
    global global_data_manager
    
    if global_data_manager is None:
        global_data_manager = 数据管理器()
        _初始化数据源(global_data_manager)
        
    return global_data_manager

def _初始化数据源(dm: 数据管理器):
    """
    初始化所有数据源并注册到数据管理器。
    
    参数:
        dm: 数据管理器实例
    """
    # 读取配置文件
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'config.json')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"读取配置文件失败: {str(e)}")
        config = {}
    
    # 获取加密货币数据源配置
    crypto_config = config.get('data_source', {}).get('crypto', {})
    
    # 初始化加密货币数据源
    try:
        # 使用新的CryptoDataProvider类
        crypto_data_provider = CryptoDataProvider(crypto_config)
        dm.注册数据源('crypto', crypto_data_provider)
        logger.info("已初始化加密货币数据源")
    except Exception as e:
        logger.error(f"初始化加密货币数据源失败: {str(e)}")
    
    # 初始化其他数据源
    # TODO: 实现其他数据源的初始化

class 数据源(ABC):
    """所有数据源的抽象基类。"""
    
    @abstractmethod
    def 获取市场数据(self, 股票代码: str, 开始日期: str, 结束日期: str, 
                 时间间隔: str = '1d') -> pd.DataFrame:
        """
        获取指定股票的市场数据(OHLCV)。
        
        参数:
            股票代码: 交易标的代码
            开始日期: 开始日期，格式为YYYY-MM-DD
            结束日期: 结束日期，格式为YYYY-MM-DD
            时间间隔: 数据间隔('1m', '5m', '15m', '30m', '1h', '4h', '1d'等)
            
        返回:
            包含以下列的DataFrame: [timestamp, open, high, low, close, volume]
        """
        pass
    
    @abstractmethod
    def 获取庄家指标(self, 股票代码: str, 开始日期: str, 结束日期: str) -> pd.DataFrame:
        """
        获取与"庄家"或机构活动相关的指标。
        
        参数:
            股票代码: 交易标的代码
            开始日期: 开始日期，格式为YYYY-MM-DD
            结束日期: 结束日期，格式为YYYY-MM-DD
            
        返回:
            包含庄家指标的DataFrame(格式取决于市场和数据源)
        """
        pass
    
    @abstractmethod
    def 获取基本面数据(self, 股票代码: str) -> Dict[str, Any]:
        """
        获取指定股票的基本面数据。
        
        参数:
            股票代码: 交易标的代码
            
        返回:
            包含基本面数据的字典
        """
        pass
    
    @abstractmethod
    def 搜索股票(self, 关键词: str) -> List[Dict[str, str]]:
        """
        根据关键词搜索股票。
        
        参数:
            关键词: 搜索关键词
            
        返回:
            包含股票信息的字典列表
        """
        pass
    
    def 保存到本地(self, 数据: Union[pd.DataFrame, Dict], 
                数据类型: str, 股票代码: str, 
                开始日期: Optional[str] = None, 
                结束日期: Optional[str] = None) -> str:
        """
        将数据保存到本地存储。
        
        参数:
            数据: 要保存的数据(DataFrame或字典)
            数据类型: 数据类型('market', 'smart_money', 'fundamental'等)
            股票代码: 交易标的代码
            开始日期: 开始日期，格式为YYYY-MM-DD(如适用)
            结束日期: 结束日期，格式为YYYY-MM-DD(如适用)
            
        返回:
            保存文件的路径
        """
        # 如果目录不存在则创建
        基础目录 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        数据目录 = os.path.join(基础目录, 数据类型)
        os.makedirs(数据目录, exist_ok=True)
        
        # 生成文件名
        if 开始日期 and 结束日期:
            文件名 = f"{股票代码}_{开始日期}_{结束日期}.csv" if isinstance(数据, pd.DataFrame) else f"{股票代码}_{开始日期}_{结束日期}.json"
        else:
            时间戳 = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            文件名 = f"{股票代码}_{时间戳}.csv" if isinstance(数据, pd.DataFrame) else f"{股票代码}_{时间戳}.json"
        
        文件路径 = os.path.join(数据目录, 文件名)
        
        # 保存数据
        if isinstance(数据, pd.DataFrame):
            数据.to_csv(文件路径, index=False)
        else:
            with open(文件路径, 'w') as f:
                json.dump(数据, f, indent=4)
        
        return 文件路径
    
    def 从本地加载(self, 数据类型: str, 股票代码: str, 
                开始日期: Optional[str] = None, 
                结束日期: Optional[str] = None) -> Union[pd.DataFrame, Dict, None]:
        """
        从本地存储加载数据。
        
        参数:
            数据类型: 数据类型('market', 'smart_money', 'fundamental'等)
            股票代码: 交易标的代码
            开始日期: 开始日期，格式为YYYY-MM-DD(如适用)
            结束日期: 结束日期，格式为YYYY-MM-DD(如适用)
            
        返回:
            DataFrame或字典形式的数据，如果未找到则返回None
        """
        基础目录 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        数据目录 = os.path.join(基础目录, 数据类型)
        
        if not os.path.exists(数据目录):
            return None
        
        # 查找匹配的文件
        if 开始日期 and 结束日期:
            csv模式 = f"{股票代码}_{开始日期}_{结束日期}.csv"
            json模式 = f"{股票代码}_{开始日期}_{结束日期}.json"
            
            for 文件名 in os.listdir(数据目录):
                if 文件名 == csv模式:
                    return pd.read_csv(os.path.join(数据目录, 文件名))
                elif 文件名 == json模式:
                    with open(os.path.join(数据目录, 文件名), 'r') as f:
                        return json.load(f)
        else:
            # 获取该股票最新的文件
            匹配文件 = [f for f in os.listdir(数据目录) if f.startswith(股票代码 + '_')]
            if not 匹配文件:
                return None
                
            最新文件 = max(匹配文件)
            文件路径 = os.path.join(数据目录, 最新文件)
            
            if 最新文件.endswith('.csv'):
                return pd.read_csv(文件路径)
            elif 最新文件.endswith('.json'):
                with open(文件路径, 'r') as f:
                    return json.load(f)
        
        return None
