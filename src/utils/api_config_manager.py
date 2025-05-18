"""
API配置管理器

用于管理和验证各种数据源API密钥。
提供保存、加载和测试API连接的功能。

日期：2025-05-21
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional, List, Tuple
import datetime

from src.utils.config_manager import ConfigManager

# 配置日志
logger = logging.getLogger(__name__)

class APIConfigManager:
    """API配置管理器类，处理市场数据源API密钥配置"""
    
    def __init__(self, config_path: str = None):
        """
        初始化API配置管理器
        
        参数:
            config_path: 配置文件路径
        """
        # 使用全局配置管理器
        self.config_manager = ConfigManager(config_path)
        logger.info("API配置管理器初始化成功")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        从配置管理器加载配置
        
        返回:
            配置信息字典
        """
        # 委托给ConfigManager的config属性
        return self.config_manager.config
    
    def update_crypto_api(self, provider, api_key='', api_secret='', use_public_api=False):
        """
        更新加密货币API配置
        
        参数:
            provider: 数据提供商
            api_key: API密钥
            api_secret: API密钥秘钥 
            use_public_api: 是否使用公共API(不需要密钥)
        
        返回:
            是否更新成功
        """
        try:
            # 获取当前加密货币配置
            crypto_config = self.config_manager.get_section('data_source.crypto')
            
            # 更新配置
            crypto_config['default_provider'] = provider
            crypto_config['use_public_api'] = use_public_api
            
            # 仅在不使用公共API且提供了密钥时保存密钥
            if not use_public_api:
                if api_key:
                    crypto_config['api_key'] = api_key
                if api_secret:
                    crypto_config['api_secret'] = api_secret
            
            # 保存配置
            self.config_manager.update_section('data_source.crypto', crypto_config)
            logger.info(f"加密货币API配置已更新，提供商: {provider}")
            
            return True
        except Exception as e:
            logger.error(f"更新加密货币API配置出错: {str(e)}")
            return False
            
    def update_stock_api(self, provider: str, api_key: str, api_params: Dict[str, Any] = None) -> bool:
        """
        更新股票API配置
        
        参数:
            provider: 数据提供商名称（例如'tushare'）
            api_key: API密钥
            api_params: 其他API参数（如果需要）
            
        返回:
            更新是否成功
        """
        try:
            # 获取当前股票配置
            stock_config = self.config_manager.get_section('data_source.stock')
            
            # 更新配置
            stock_config['default_provider'] = provider
            stock_config['api_key'] = api_key
            
            if api_params:
                for key, value in api_params.items():
                    stock_config[key] = value
            
            # 保存配置
            result = self.config_manager.update_section('data_source.stock', stock_config)
            logger.info(f"股票API配置已更新，提供商: {provider}")
            
            return result
        except Exception as e:
            logger.error(f"更新股票API配置失败: {e}")
            return False
    
    def test_binance_connection(self, api_key: str, api_secret: str) -> Tuple[bool, str]:
        """
        测试Binance API连接
        
        参数:
            api_key: Binance API密钥
            api_secret: Binance API密钥Secret
            
        返回:
            (连接是否成功, 状态信息)
        """
        try:
            url = "https://api.binance.com/api/v3/ping"
            headers = {"X-MBX-APIKEY": api_key}
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                return True, "Binance API连接成功"
            else:
                return False, f"Binance API连接失败，状态码: {response.status_code}"
        except Exception as e:
            return False, f"Binance API连接测试出错: {str(e)}"
    
    def test_tushare_connection(self, api_key: str) -> Tuple[bool, str]:
        """
        测试Tushare API连接
        
        参数:
            api_key: Tushare API密钥
            
        返回:
            (连接是否成功, 状态信息)
        """
        try:
            # 尝试导入Tushare
            import tushare as ts
            
            # 设置API密钥并获取数据
            ts.set_token(api_key)
            pro = ts.pro_api()
            
            # 测试获取股票列表
            df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name')
            
            if not df.empty:
                return True, f"Tushare API连接成功，获取到{len(df)}支股票信息"
            else:
                return False, "Tushare API连接失败，返回数据为空"
        except ImportError:
            return False, "Tushare未安装，请执行: pip install tushare"
        except Exception as e:
            return False, f"Tushare API连接测试出错: {str(e)}"
    
    def get_crypto_api_config(self):
        """
        获取加密货币API配置
        
        返回:
            加密货币API配置字典
        """
        try:
            # 获取配置
            crypto_config = self.config_manager.get_section('data_source.crypto')
            
            # 设置默认值（如果需要）
            if not crypto_config:
                crypto_config = {
                    'default_provider': 'binance',
                    'api_key': '',
                    'api_secret': '',
                    'use_public_api': False
                }
            
            # 确保 use_public_api 字段存在
            if 'use_public_api' not in crypto_config:
                crypto_config['use_public_api'] = False
                
            return crypto_config
        except Exception as e:
            logger.error(f"获取加密货币API配置出错: {str(e)}")
            # 返回默认配置
            return {
                'default_provider': 'binance',
                'api_key': '',
                'api_secret': '',
                'use_public_api': False
            }
    
    def get_stock_api_config(self) -> Dict[str, Any]:
        """
        获取股票API配置
        
        返回:
            股票API配置字典
        """
        stock_config = self.config_manager.get_section('data_source.stock')
        
        # 设置默认值（如果需要）
        if not stock_config:
            stock_config = {
                'default_provider': 'tushare',
                'api_key': ''
            }
            
        return stock_config 