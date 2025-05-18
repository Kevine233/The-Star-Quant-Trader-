"""
全局配置管理器

处理全系统配置，提供统一的配置访问、修改和保存接口。
使用单例模式确保全局只有一个配置实例。

日期：2025-05-23
"""

import os
import json
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple
import datetime

# 配置日志
logger = logging.getLogger(__name__)

class ConfigManager:
    """全局配置管理器，使用单例模式确保全局唯一"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config_path: str = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config_path: str = None):
        """
        初始化配置管理器
        
        参数:
            config_path: 配置文件路径
        """
        # 避免重复初始化
        if self._initialized:
            return
            
        if config_path is None:
            # 使用项目根目录下的config目录
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.config_path = os.path.join(base_dir, 'config', 'config.json')
        else:
            self.config_path = config_path
        
        # 初始化配置
        self.config = self._load_config()
        
        # 设置默认配置
        self._set_default_config()
        
        # 标记为已初始化
        self._initialized = True
        
        logger.info(f"配置管理器初始化完成，使用配置文件: {self.config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        从文件加载配置
        
        返回:
            配置信息字典
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"配置文件不存在: {self.config_path}，将使用默认配置")
                return {}
        except Exception as e:
            logger.error(f"加载配置文件出错: {e}")
            return {}
    
    def _save_config(self) -> bool:
        """
        保存配置到文件
        
        返回:
            保存是否成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            
            logger.info(f"配置已保存到: {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"保存配置文件出错: {e}")
            return False
    
    def _set_default_config(self):
        """设置默认配置，在没有现有配置的情况下使用"""
        default_config = {
            "host": "0.0.0.0",
            "port": 5000,
            "debug": False,
            "data_source": {
                "stock": {
                    "default_provider": "tushare",
                    "api_key": ""
                },
                "crypto": {
                    "default_provider": "binance",
                    "api_key": "",
                    "api_secret": "",
                    "use_public_api": False
                }
            },
            "backtest": {
                "default_initial_capital": 1000000,
                "default_commission_rate": 0.0003,
                "default_slippage": 0.0001
            },
            "trade": {
                "default_mode": "simulated",
                "broker_api": {
                    "name": "",
                    "api_key": "",
                    "api_secret": "",
                    "api_base_url": ""
                }
            },
            "risk_management": {
                "max_position_size": 0.2,
                "max_total_position": 0.8,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.1
            }
        }
        
        # 只添加缺失的配置项，不覆盖现有配置
        self._merge_config(default_config, self.config)
        self.config = default_config
    
    def _merge_config(self, target: Dict[str, Any], source: Dict[str, Any]):
        """
        合并配置，将source中的配置合并到target中
        
        参数:
            target: 目标配置字典
            source: 源配置字典
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_config(target[key], value)
            else:
                target[key] = value
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        获取配置项
        
        参数:
            path: 配置路径，用点分隔，如 'data_source.stock.api_key'
            default: 如果配置项不存在，返回的默认值
            
        返回:
            配置项的值，如果配置项不存在则返回默认值
        """
        parts = path.split('.')
        current = self.config
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def set(self, path: str, value: Any) -> bool:
        """
        设置配置项
        
        参数:
            path: 配置路径，用点分隔，如 'data_source.stock.api_key'
            value: 要设置的值
            
        返回:
            设置是否成功
        """
        parts = path.split('.')
        current = self.config
        
        # 遍历路径到倒数第二级
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        
        # 设置最后一级的值
        current[parts[-1]] = value
        
        # 保存配置
        return self._save_config()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        获取配置中的一个部分
        
        参数:
            section: 部分名称，如 'data_source.stock'
            
        返回:
            配置部分的字典副本，如果部分不存在则返回空字典
        """
        value = self.get(section, {})
        
        # 返回副本，避免修改原配置
        if isinstance(value, dict):
            return value.copy()
        else:
            return {}
    
    def update_section(self, section: str, config: Dict[str, Any]) -> bool:
        """
        更新配置中的一个部分
        
        参数:
            section: 部分名称，如 'data_source.stock'
            config: 新的配置字典
            
        返回:
            更新是否成功
        """
        parts = section.split('.')
        current = self.config
        
        # 遍历路径到倒数第二级
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # 设置或更新最后一级
        if parts[-1] not in current:
            current[parts[-1]] = {}
        
        # 更新配置
        current_section = current[parts[-1]]
        if isinstance(current_section, dict) and isinstance(config, dict):
            current_section.update(config)
        else:
            current[parts[-1]] = config
        
        # 保存配置
        return self._save_config()
    
    def save(self) -> bool:
        """
        保存当前配置到文件
        
        返回:
            保存是否成功
        """
        return self._save_config()
    
    def reload(self) -> bool:
        """
        重新从文件加载配置
        
        返回:
            重新加载是否成功
        """
        try:
            self.config = self._load_config()
            self._set_default_config()
            return True
        except Exception as e:
            logger.error(f"重新加载配置失败: {e}")
            return False 