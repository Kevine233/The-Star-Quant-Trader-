"""
通知模块基类

定义通知接口的基本结构
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class NotificationBase(ABC):
    """通知基类，定义所有通知方法必须实现的接口"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化通知方法
        
        参数:
            config: 配置信息
        """
        self.config = config
        
    @abstractmethod
    def send(self, message: str, subject: Optional[str] = None, **kwargs) -> bool:
        """
        发送通知
        
        参数:
            message: 通知内容
            subject: 通知主题
            **kwargs: 其他参数
            
        返回:
            发送是否成功
        """
        pass 