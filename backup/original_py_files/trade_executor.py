"""
交易执行模块

本模块实现了交易信号执行功能，支持模拟盘和实盘交易。
主要功能包括：
1. 交易信号生成与处理
2. 订单管理与执行
3. 持仓管理
4. 交易日志记录
5. 多券商API接口支持

日期：2025-05-16
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import json
import time
import threading
import queue
import requests
from abc import ABC, abstractmethod
import hmac
import hashlib
import base64
import uuid
import websocket
import ssl
from urllib.parse import urlencode

# 配置日志
logger = logging.getLogger(__name__)

class Order:
    """订单类，表示一个交易订单"""
    
    def __init__(self, symbol: str, order_type: str, direction: str, quantity: float, price: float = None, 
                 stop_price: float = None, order_id: str = None, status: str = "created", 
                 create_time: datetime = None, update_time: datetime = None, filled_quantity: float = 0,
                 filled_price: float = 0, commission: float = 0, message: str = ""):
        """
        初始化订单
        
        参数:
            symbol: 交易品种代码
            order_type: 订单类型，如'market', 'limit', 'stop', 'stop_limit'
            direction: 交易方向，'buy'或'sell'
            quantity: 交易数量
            price: 限价单价格
            stop_price: 止损/止盈价格
            order_id: 订单ID，如果为None则自动生成
            status: 订单状态，如'created', 'submitted', 'filled', 'canceled', 'rejected'
            create_time: 创建时间，如果为None则使用当前时间
            update_time: 更新时间，如果为None则使用当前时间
            filled_quantity: 已成交数量
            filled_price: 成交均价
            commission: 佣金
            message: 订单消息，如错误信息
        """
        self.symbol = symbol
        self.order_type = order_type
        self.direction = direction
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.order_id = order_id or str(uuid.uuid4())
        self.status = status
        self.create_time = create_time or datetime.now()
        self.update_time = update_time or datetime.now()
        self.filled_quantity = filled_quantity
        self.filled_price = filled_price
        self.commission = commission
        self.message = message
    
    def update(self, status: str = None, filled_quantity: float = None, filled_price: float = None, 
               commission: float = None, message: str = None):
        """
        更新订单状态
        
        参数:
            status: 新状态
            filled_quantity: 新的已成交数量
            filled_price: 新的成交均价
            commission: 新的佣金
            message: 新的订单消息
        """
        if status is not None:
            self.status = status
        
        if filled_quantity is not None:
            self.filled_quantity = filled_quantity
        
        if filled_price is not None:
            self.filled_price = filled_price
        
        if commission is not None:
            self.commission = commission
        
        if message is not None:
            self.message = message
        
        self.update_time = datetime.now()
    
    def is_filled(self) -> bool:
        """检查订单是否已完全成交"""
        return self.status == "filled" and self.filled_quantity >= self.quantity
    
    def is_active(self) -> bool:
        """检查订单是否仍然活跃"""
        return self.status in ["created", "submitted", "partially_filled"]
    
    def to_dict(self) -> Dict:
        """将订单转换为字典"""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "order_type": self.order_type,
            "direction": self.direction,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "status": self.status,
            "create_time": self.create_time.isoformat(),
            "update_time": self.update_time.isoformat(),
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "commission": self.commission,
            "message": self.message
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Order':
        """从字典创建订单"""
        create_time = datetime.fromisoformat(data["create_time"]) if isinstance(data["create_time"], str) else data["create_time"]
        update_time = datetime.fromisoformat(data["update_time"]) if isinstance(data["update_time"], str) else data["update_time"]
        
        return cls(
            symbol=data["symbol"],
            order_type=data["order_type"],
            direction=data["direction"],
            quantity=data["quantity"],
            price=data["price"],
            stop_price=data["stop_price"],
            order_id=data["order_id"],
            status=data["status"],
            create_time=create_time,
            update_time=update_time,
            filled_quantity=data["filled_quantity"],
            filled_price=data["filled_price"],
            commission=data["commission"],
            message=data["message"]
        )


class Position:
    """持仓类，表示一个交易品种的持仓"""
    
    def __init__(self, symbol: str, quantity: float = 0, average_price: float = 0, 
                 current_price: float = 0, open_time: datetime = None, 
                 realized_pnl: float = 0, unrealized_pnl: float = 0):
        """
        初始化持仓
        
        参数:
            symbol: 交易品种代码
            quantity: 持仓数量
            average_price: 持仓均价
            current_price: 当前价格
            open_time: 开仓时间，如果为None则使用当前时间
            realized_pnl: 已实现盈亏
            unrealized_pnl: 未实现盈亏
        """
        self.symbol = symbol
        self.quantity = quantity
        self.average_price = average_price
        self.current_price = current_price
        self.open_time = open_time or datetime.now()
        self.realized_pnl = realized_pnl
        self.unrealized_pnl = unrealized_pnl
    
    def update_price(self, price: float):
        """
        更新当前价格和未实现盈亏
        
        参数:
            price: 新的价格
        """
        self.current_price = price
        self.update_unrealized_pnl()
    
    def update_unrealized_pnl(self):
        """更新未实现盈亏"""
        if self.quantity != 0 and self.average_price != 0:
            self.unrealized_pnl = (self.current_price - self.average_price) * self.quantity
    
    def add(self, quantity: float, price: float):
        """
        增加持仓
        
        参数:
            quantity: 增加的数量
            price: 成交价格
        """
        if quantity <= 0:
            return
        
        # 计算新的持仓均价
        total_cost = self.average_price * self.quantity + price * quantity
        self.quantity += quantity
        self.average_price = total_cost / self.quantity if self.quantity > 0 else 0
        
        # 更新未实现盈亏
        self.update_unrealized_pnl()
    
    def reduce(self, quantity: float, price: float):
        """
        减少持仓
        
        参数:
            quantity: 减少的数量
            price: 成交价格
            
        返回:
            实现的盈亏
        """
        if quantity <= 0 or quantity > self.quantity:
            return 0
        
        # 计算实现盈亏
        realized_pnl = (price - self.average_price) * quantity
        self.realized_pnl += realized_pnl
        
        # 更新持仓
        self.quantity -= quantity
        
        # 如果持仓为0，重置均价
        if self.quantity == 0:
            self.average_price = 0
        
        # 更新未实现盈亏
        self.update_unrealized_pnl()
        
        return realized_pnl
    
    def to_dict(self) -> Dict:
        """将持仓转换为字典"""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_price": self.average_price,
            "current_price": self.current_price,
            "open_time": self.open_time.isoformat(),
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.realized_pnl + self.unrealized_pnl,
            "market_value": self.quantity * self.current_price
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Position':
        """从字典创建持仓"""
        open_time = datetime.fromisoformat(data["open_time"]) if isinstance(data["open_time"], str) else data["open_time"]
        
        return cls(
            symbol=data["symbol"],
            quantity=data["quantity"],
            average_price=data["average_price"],
            current_price=data["current_price"],
            open_time=open_time,
            realized_pnl=data["realized_pnl"],
            unrealized_pnl=data["unrealized_pnl"]
        )


class TradeExecutor(ABC):
    """交易执行器抽象基类，定义交易执行接口"""
    
    def __init__(self, config: Dict = None):
        """
        初始化交易执行器
        
        参数:
            config: 配置信息
        """
        self.config = config or {}
        self.name = "基础交易执行器"
        
        # 订单和持仓管理
        self.orders = {}  # 订单字典，键为订单ID
        self.positions = {}  # 持仓字典，键为交易品种代码
        
        # 账户信息
        self.account_info = {
            "cash": self.config.get("initial_cash", 1000000),
            "total_value": self.config.get("initial_cash", 1000000),
            "margin": 0,
            "realized_pnl": 0,
            "unrealized_pnl": 0
        }
        
        # 交易日志
        self.trade_log = []
        
        # 状态
        self.is_running = False
        self.last_update_time = None
        
        logger.info(f"{self.name} 初始化成功")
    
    @abstractmethod
    def connect(self) -> bool:
        """
        连接到交易系统
        
        返回:
            连接是否成功
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        断开与交易系统的连接
        
        返回:
            断开连接是否成功
        """
        pass
    
    @abstractmethod
    def place_order(self, order: Order) -> str:
        """
        下单
        
        参数:
            order: 订单对象
            
        返回:
            订单ID
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        取消订单
        
        参数:
            order_id: 订单ID
            
        返回:
            取消是否成功
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict:
        """
        获取订单状态
        
        参数:
            order_id: 订单ID
            
        返回:
            订单状态字典
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """
        获取当前持仓
        
        返回:
            持仓字典，键为交易品种代码
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict:
        """
        获取账户信息
        
        返回:
            账户信息字典
        """
        pass
    
    def update_account_info(self):
        """更新账户信息"""
        # 计算持仓市值和未实现盈亏
        position_value = 0
        unrealized_pnl = 0
        
        for position in self.positions.values():
            position_value += position.quantity * position.current_price
            unrealized_pnl += position.unrealized_pnl
        
        # 更新账户信息
        self.account_info["unrealized_pnl"] = unrealized_pnl
        self.account_info["total_value"] = self.account_info["cash"] + position_value
        
        self.last_update_time = datetime.now()
    
    def log_trade(self, trade_type: str, symbol: str, quantity: float, price: float, 
                  order_id: str = None, commission: float = 0, message: str = ""):
        """
        记录交易日志
        
        参数:
            trade_type: 交易类型，如'buy', 'sell'
            symbol: 交易品种代码
            quantity: 交易数量
            price: 交易价格
            order_id: 订单ID
            commission: 佣金
            message: 交易消息
        """
        log_entry = {
            "time": datetime.now().isoformat(),
            "trade_type": trade_type,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "order_id": order_id,
            "commission": commission,
            "message": message
        }
        
        self.trade_log.append(log_entry)
        logger.info(f"交易记录: {trade_type} {symbol} {quantity}@{price}, 订单ID: {order_id}, 佣金: {commission}")
    
    def save_trade_log(self, file_path: str):
        """
        保存交易日志
        
        参数:
            file_path: 保存路径
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.trade_log, f, ensure_ascii=False, indent=4)
            
            logger.info(f"交易日志已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存交易日志失败: {e}")
    
    def load_trade_log(self, file_path: str):
        """
        加载交易日志
        
        参数:
            file_path: 加载路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.trade_log = json.load(f)
            
            logger.info(f"交易日志已从 {file_path} 加载")
        except Exception as e:
            logger.error(f"加载交易日志失败: {e}")
    
    def get_trade_summary(self) -> Dict:
        """
        获取交易摘要
        
        返回:
            交易摘要字典
        """
        if not self.trade_log:
            return {
                "total_trades": 0,
                "buy_trades": 0,
                "sell_trades": 0,
                "total_commission": 0,
                "total_volume": 0,
                "symbols_traded": []
            }
        
        # 统计交易
        buy_trades = [log for log in self.trade_log if log["trade_type"] == "buy"]
        sell_trades = [log for log in self.trade_log if log["trade_type"] == "sell"]
        
        # 计算总佣金和交易量
        total_commission = sum(log["commission"] for log in self.trade_log)
        total_volume = sum(log["quantity"] * log["price"] for log in self.trade_log)
        
        # 获取交易的品种
        symbols_traded = list(set(log["symbol"] for log in self.trade_log))
        
        return {
            "total_trades": len(self.trade_log),
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "total_commission": total_commission,
            "total_volume": total_volume,
            "symbols_traded": symbols_traded
        }


class SimulatedTradeExecutor(TradeExecutor):
    """模拟交易执行器，用于模拟盘交易"""
    
    def __init__(self, config: Dict = None):
        """
        初始化模拟交易执行器
        
        参数:
            config: 配置信息
        """
        super().__init__(config)
        self.name = "模拟交易执行器"
        
        # 模拟交易特有配置
        self.commission_rate = self.config.get("commission_rate", 0.0003)  # 佣金率
        self.slippage = self.config.get("slippage", 0.0001)  # 滑点
        self.tax_rate = self.config.get("tax_rate", 0.001)  # 印花税率（仅卖出时收取）
        self.min_commission = self.config.get("min_commission", 5)  # 最低佣金
        
        # 市场数据
        self.market_data = {}  # 市场数据字典，键为交易品种代码
        
        # 订单处理队列
        self.order_queue = queue.Queue()
        self.order_processor_thread = None
        
        logger.info(f"{self.name} 初始化成功")
    
    def connect(self) -> bool:
        """
        连接到模拟交易系统
        
        返回:
            连接是否成功
        """
        if self.is_running:
            logger.warning("模拟交易系统已经在运行")
            return True
        
        # 启动订单处理线程
        self.is_running = True
        self.order_processor_thread = threading.Thread(target=self._process_orders)
        self.order_processor_thread.daemon = True
        self.order_processor_thread.start()
        
        logger.info("模拟交易系统已连接")
        return True
    
    def disconnect(self) -> bool:
        """
        断开与模拟交易系统的连接
        
        返回:
            断开连接是否成功
        """
        if not self.is_running:
            logger.warning("模拟交易系统未在运行")
            return True
        
        # 停止订单处理线程
        self.is_running = False
        if self.order_processor_thread and self.order_processor_thread.is_alive():
            self.order_processor_thread.join(timeout=5)
        
        logger.info("模拟交易系统已断开连接")
        return True
    
    def place_order(self, order: Order) -> str:
        """
        下单
        
        参数:
            order: 订单对象
            
        返回:
            订单ID
        """
        if not self.is_running:
            logger.error("模拟交易系统未在运行，无法下单")
            order.update(status="rejected", message="交易系统未在运行")
            return order.order_id
        
        # 检查市场数据
        if order.symbol not in self.market_data:
            logger.error(f"无法下单，缺少市场数据 - 品种: {order.symbol}")
            order.update(status="rejected", message="缺少市场数据")
            return order.order_id
        
        # 检查订单类型
        if order.order_type not in ["market", "limit", "stop", "stop_limit"]:
            logger.error(f"无法下单，不支持的订单类型 - 类型: {order.order_type}")
            order.update(status="rejected", message=f"不支持的订单类型: {order.order_type}")
            return order.order_id
        
        # 检查资金是否足够（仅买入订单）
        if order.direction == "buy":
            # 估算订单金额
            market_price = self.market_data[order.symbol].get("close", 0)
            estimated_price = order.price or market_price
            estimated_cost = estimated_price * order.quantity
            
            # 检查资金
            if estimated_cost > self.account_info["cash"]:
                logger.error(f"无法下单，资金不足 - 品种: {order.symbol}, 估算成本: {estimated_cost}, 可用资金: {self.account_info['cash']}")
                order.update(status="rejected", message="资金不足")
                return order.order_id
        
        # 检查持仓是否足够（仅卖出订单）
        if order.direction == "sell":
            position = self.positions.get(order.symbol)
            if not position or position.quantity < order.quantity:
                logger.error(f"无法下单，持仓不足 - 品种: {order.symbol}, 订单数量: {order.quantity}, 持仓数量: {position.quantity if position else 0}")
                order.update(status="rejected", message="持仓不足")
                return order.order_id
        
        # 更新订单状态
        order.update(status="submitted")
        
        # 添加到订单字典
        self.orders[order.order_id] = order
        
        # 添加到订单处理队列
        self.order_queue.put(order.order_id)
        
        logger.info(f"订单已提交 - ID: {order.order_id}, 品种: {order.symbol}, 类型: {order.order_type}, 方向: {order.direction}, 数量: {order.quantity}, 价格: {order.price}")
        return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """
        取消订单
        
        参数:
            order_id: 订单ID
            
        返回:
            取消是否成功
        """
        if not self.is_running:
            logger.error("模拟交易系统未在运行，无法取消订单")
            return False
        
        # 检查订单是否存在
        if order_id not in self.orders:
            logger.error(f"无法取消订单，订单不存在 - ID: {order_id}")
            return False
        
        # 获取订单
        order = self.orders[order_id]
        
        # 检查订单是否可以取消
        if not order.is_active():
            logger.error(f"无法取消订单，订单状态不允许取消 - ID: {order_id}, 状态: {order.status}")
            return False
        
        # 更新订单状态
        order.update(status="canceled", message="用户取消")
        
        logger.info(f"订单已取消 - ID: {order_id}")
        return True
    
    def get_order_status(self, order_id: str) -> Dict:
        """
        获取订单状态
        
        参数:
            order_id: 订单ID
            
        返回:
            订单状态字典
        """
        if order_id not in self.orders:
            logger.error(f"无法获取订单状态，订单不存在 - ID: {order_id}")
            return {}
        
        return self.orders[order_id].to_dict()
    
    def get_positions(self) -> Dict[str, Position]:
        """
        获取当前持仓
        
        返回:
            持仓字典，键为交易品种代码
        """
        return self.positions
    
    def get_account_info(self) -> Dict:
        """
        获取账户信息
        
        返回:
            账户信息字典
        """
        # 更新账户信息
        self.update_account_info()
        return self.account_info
    
    def update_market_data(self, symbol: str, data: Dict):
        """
        更新市场数据
        
        参数:
            symbol: 交易品种代码
            data: 市场数据字典
        """
        self.market_data[symbol] = data
        
        # 更新持仓价格
        if symbol in self.positions:
            self.positions[symbol].update_price(data.get("close", 0))
        
        # 更新账户信息
        self.update_account_info()
    
    def _process_orders(self):
        """订单处理线程"""
        while self.is_running:
            try:
                # 从队列获取订单ID
                try:
                    order_id = self.order_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # 检查订单是否存在
                if order_id not in self.orders:
                    logger.warning(f"订单处理失败，订单不存在 - ID: {order_id}")
                    continue
                
                # 获取订单
                order = self.orders[order_id]
                
                # 检查订单是否活跃
                if not order.is_active():
                    logger.warning(f"订单处理失败，订单状态不活跃 - ID: {order_id}, 状态: {order.status}")
                    continue
                
                # 检查市场数据
                if order.symbol not in self.market_data:
                    logger.warning(f"订单处理失败，缺少市场数据 - 品种: {order.symbol}")
                    continue
                
                # 获取市场数据
                market_data = self.market_data[order.symbol]
                
                # 处理不同类型的订单
                if order.order_type == "market":
                    self._process_market_order(order, market_data)
                elif order.order_type == "limit":
                    self._process_limit_order(order, market_data)
                elif order.order_type == "stop":
                    self._process_stop_order(order, market_data)
                elif order.order_type == "stop_limit":
                    self._process_stop_limit_order(order, market_data)
                
                # 标记任务完成
                self.order_queue.task_done()
            
            except Exception as e:
                logger.error(f"订单处理线程发生错误: {e}")
    
    def _process_market_order(self, order: Order, market_data: Dict):
        """
        处理市价单
        
        参数:
            order: 订单对象
            market_data: 市场数据字典
        """
        # 获取市场价格
        market_price = market_data.get("close", 0)
        
        if market_price <= 0:
            logger.warning(f"市价单处理失败，无效的市场价格 - 品种: {order.symbol}, 价格: {market_price}")
            return
        
        # 考虑滑点
        execution_price = market_price * (1 + self.slippage) if order.direction == "buy" else market_price * (1 - self.slippage)
        
        # 执行订单
        self._execute_order(order, execution_price)
    
    def _process_limit_order(self, order: Order, market_data: Dict):
        """
        处理限价单
        
        参数:
            order: 订单对象
            market_data: 市场数据字典
        """
        # 检查价格
        if order.price is None or order.price <= 0:
            logger.warning(f"限价单处理失败，无效的限价 - 品种: {order.symbol}, 价格: {order.price}")
            return
        
        # 获取市场价格
        market_price = market_data.get("close", 0)
        
        if market_price <= 0:
            logger.warning(f"限价单处理失败，无效的市场价格 - 品种: {order.symbol}, 价格: {market_price}")
            return
        
        # 检查是否可以成交
        can_execute = False
        
        if order.direction == "buy" and market_price <= order.price:
            can_execute = True
        elif order.direction == "sell" and market_price >= order.price:
            can_execute = True
        
        if can_execute:
            # 考虑滑点
            execution_price = min(market_price * (1 + self.slippage), order.price) if order.direction == "buy" else max(market_price * (1 - self.slippage), order.price)
            
            # 执行订单
            self._execute_order(order, execution_price)
        else:
            # 将订单放回队列，等待下次处理
            self.order_queue.put(order.order_id)
    
    def _process_stop_order(self, order: Order, market_data: Dict):
        """
        处理止损单
        
        参数:
            order: 订单对象
            market_data: 市场数据字典
        """
        # 检查止损价格
        if order.stop_price is None or order.stop_price <= 0:
            logger.warning(f"止损单处理失败，无效的止损价格 - 品种: {order.symbol}, 价格: {order.stop_price}")
            return
        
        # 获取市场价格
        market_price = market_data.get("close", 0)
        
        if market_price <= 0:
            logger.warning(f"止损单处理失败，无效的市场价格 - 品种: {order.symbol}, 价格: {market_price}")
            return
        
        # 检查是否触发止损
        triggered = False
        
        if order.direction == "buy" and market_price >= order.stop_price:
            triggered = True
        elif order.direction == "sell" and market_price <= order.stop_price:
            triggered = True
        
        if triggered:
            # 转换为市价单执行
            order.order_type = "market"
            self._process_market_order(order, market_data)
        else:
            # 将订单放回队列，等待下次处理
            self.order_queue.put(order.order_id)
    
    def _process_stop_limit_order(self, order: Order, market_data: Dict):
        """
        处理止损限价单
        
        参数:
            order: 订单对象
            market_data: 市场数据字典
        """
        # 检查止损价格和限价
        if order.stop_price is None or order.stop_price <= 0 or order.price is None or order.price <= 0:
            logger.warning(f"止损限价单处理失败，无效的价格 - 品种: {order.symbol}, 止损价格: {order.stop_price}, 限价: {order.price}")
            return
        
        # 获取市场价格
        market_price = market_data.get("close", 0)
        
        if market_price <= 0:
            logger.warning(f"止损限价单处理失败，无效的市场价格 - 品种: {order.symbol}, 价格: {market_price}")
            return
        
        # 检查是否触发止损
        triggered = False
        
        if order.direction == "buy" and market_price >= order.stop_price:
            triggered = True
        elif order.direction == "sell" and market_price <= order.stop_price:
            triggered = True
        
        if triggered:
            # 转换为限价单执行
            order.order_type = "limit"
            self._process_limit_order(order, market_data)
        else:
            # 将订单放回队列，等待下次处理
            self.order_queue.put(order.order_id)
    
    def _execute_order(self, order: Order, execution_price: float):
        """
        执行订单
        
        参数:
            order: 订单对象
            execution_price: 执行价格
        """
        # 计算成交金额
        execution_value = execution_price * order.quantity
        
        # 计算佣金
        commission = max(self.min_commission, execution_value * self.commission_rate)
        
        # 计算印花税（仅卖出时收取）
        tax = execution_value * self.tax_rate if order.direction == "sell" else 0
        
        # 更新账户资金
        if order.direction == "buy":
            # 检查资金是否足够
            total_cost = execution_value + commission
            
            if total_cost > self.account_info["cash"]:
                logger.warning(f"订单执行失败，资金不足 - 品种: {order.symbol}, 成本: {total_cost}, 可用资金: {self.account_info['cash']}")
                order.update(status="rejected", message="资金不足")
                return
            
            # 扣除资金
            self.account_info["cash"] -= total_cost
            
            # 更新持仓
            if order.symbol not in self.positions:
                self.positions[order.symbol] = Position(order.symbol)
            
            self.positions[order.symbol].add(order.quantity, execution_price)
        
        elif order.direction == "sell":
            # 检查持仓是否足够
            position = self.positions.get(order.symbol)
            
            if not position or position.quantity < order.quantity:
                logger.warning(f"订单执行失败，持仓不足 - 品种: {order.symbol}, 订单数量: {order.quantity}, 持仓数量: {position.quantity if position else 0}")
                order.update(status="rejected", message="持仓不足")
                return
            
            # 减少持仓
            realized_pnl = position.reduce(order.quantity, execution_price)
            
            # 增加资金
            self.account_info["cash"] += (execution_value - commission - tax)
            self.account_info["realized_pnl"] += realized_pnl
            
            # 如果持仓为0，删除持仓记录
            if position.quantity == 0:
                del self.positions[order.symbol]
        
        # 更新订单状态
        order.update(
            status="filled",
            filled_quantity=order.quantity,
            filled_price=execution_price,
            commission=commission
        )
        
        # 记录交易日志
        self.log_trade(
            trade_type=order.direction,
            symbol=order.symbol,
            quantity=order.quantity,
            price=execution_price,
            order_id=order.order_id,
            commission=commission,
            message=f"订单已成交，{'买入' if order.direction == 'buy' else '卖出'} {order.quantity} {order.symbol} @ {execution_price}"
        )
        
        # 更新账户信息
        self.update_account_info()
        
        logger.info(f"订单已成交 - ID: {order.order_id}, 品种: {order.symbol}, 方向: {order.direction}, 数量: {order.quantity}, 价格: {execution_price}, 佣金: {commission}")


class BrokerAPIExecutor(TradeExecutor):
    """券商API交易执行器，用于实盘交易"""
    
    def __init__(self, config: Dict = None):
        """
        初始化券商API交易执行器
        
        参数:
            config: 配置信息，必须包含API密钥等信息
        """
        super().__init__(config)
        self.name = "券商API交易执行器"
        
        # 检查配置
        required_config = ["api_key", "api_secret", "broker_name"]
        for key in required_config:
            if key not in self.config:
                raise ValueError(f"缺少必要的配置项: {key}")
        
        # 券商API特有配置
        self.api_key = self.config["api_key"]
        self.api_secret = self.config["api_secret"]
        self.broker_name = self.config["broker_name"]
        self.api_base_url = self.config.get("api_base_url", "")
        
        # API会话
        self.session = requests.Session()
        
        # WebSocket连接
        self.ws = None
        self.ws_thread = None
        self.ws_connected = False
        
        # 订单和持仓缓存
        self.orders_cache = {}
        self.positions_cache = {}
        self.account_info_cache = {}
        self.last_cache_update = None
        
        logger.info(f"{self.name} ({self.broker_name}) 初始化成功")
    
    def connect(self) -> bool:
        """
        连接到券商API
        
        返回:
            连接是否成功
        """
        if self.is_running:
            logger.warning(f"{self.broker_name} API已经连接")
            return True
        
        try:
            # 测试API连接
            response = self._api_request("GET", "/api/v1/ping")
            
            if not response or "success" not in response or not response["success"]:
                logger.error(f"连接{self.broker_name} API失败: {response.get('message', '未知错误')}")
                return False
            
            # 获取账户信息
            account_info = self._api_request("GET", "/api/v1/account")
            
            if not account_info or "success" not in account_info or not account_info["success"]:
                logger.error(f"获取{self.broker_name}账户信息失败: {account_info.get('message', '未知错误')}")
                return False
            
            # 更新账户信息缓存
            self.account_info_cache = account_info.get("data", {})
            
            # 获取持仓信息
            positions = self._api_request("GET", "/api/v1/positions")
            
            if positions and "success" in positions and positions["success"]:
                # 更新持仓缓存
                self.positions_cache = {
                    position["symbol"]: Position(
                        symbol=position["symbol"],
                        quantity=position["quantity"],
                        average_price=position["average_price"],
                        current_price=position["current_price"],
                        open_time=datetime.fromisoformat(position["open_time"]) if "open_time" in position else datetime.now(),
                        realized_pnl=position.get("realized_pnl", 0),
                        unrealized_pnl=position.get("unrealized_pnl", 0)
                    )
                    for position in positions.get("data", [])
                }
            
            # 获取未完成订单
            open_orders = self._api_request("GET", "/api/v1/orders", {"status": "open"})
            
            if open_orders and "success" in open_orders and open_orders["success"]:
                # 更新订单缓存
                self.orders_cache = {
                    order["order_id"]: Order.from_dict(order)
                    for order in open_orders.get("data", [])
                }
            
            # 启动WebSocket连接
            self._connect_websocket()
            
            # 设置状态
            self.is_running = True
            self.last_cache_update = datetime.now()
            
            logger.info(f"{self.broker_name} API连接成功")
            return True
        
        except Exception as e:
            logger.error(f"连接{self.broker_name} API失败: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        断开与券商API的连接
        
        返回:
            断开连接是否成功
        """
        if not self.is_running:
            logger.warning(f"{self.broker_name} API未连接")
            return True
        
        try:
            # 关闭WebSocket连接
            self._disconnect_websocket()
            
            # 关闭会话
            self.session.close()
            
            # 设置状态
            self.is_running = False
            
            logger.info(f"{self.broker_name} API已断开连接")
            return True
        
        except Exception as e:
            logger.error(f"断开{self.broker_name} API连接失败: {e}")
            return False
    
    def place_order(self, order: Order) -> str:
        """
        下单
        
        参数:
            order: 订单对象
            
        返回:
            订单ID
        """
        if not self.is_running:
            logger.error(f"{self.broker_name} API未连接，无法下单")
            order.update(status="rejected", message="交易系统未连接")
            return order.order_id
        
        try:
            # 准备请求参数
            params = {
                "symbol": order.symbol,
                "order_type": order.order_type,
                "direction": order.direction,
                "quantity": order.quantity
            }
            
            if order.price is not None:
                params["price"] = order.price
            
            if order.stop_price is not None:
                params["stop_price"] = order.stop_price
            
            # 发送API请求
            response = self._api_request("POST", "/api/v1/orders", params)
            
            if not response or "success" not in response or not response["success"]:
                error_message = response.get("message", "未知错误") if response else "API请求失败"
                logger.error(f"下单失败: {error_message}")
                order.update(status="rejected", message=error_message)
                return order.order_id
            
            # 获取返回的订单ID
            order_data = response.get("data", {})
            broker_order_id = order_data.get("order_id", order.order_id)
            
            # 更新订单信息
            order.order_id = broker_order_id
            order.update(status="submitted")
            
            # 添加到订单字典
            self.orders[order.order_id] = order
            self.orders_cache[order.order_id] = order
            
            logger.info(f"订单已提交 - ID: {order.order_id}, 品种: {order.symbol}, 类型: {order.order_type}, 方向: {order.direction}, 数量: {order.quantity}, 价格: {order.price}")
            return order.order_id
        
        except Exception as e:
            logger.error(f"下单失败: {e}")
            order.update(status="rejected", message=str(e))
            return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """
        取消订单
        
        参数:
            order_id: 订单ID
            
        返回:
            取消是否成功
        """
        if not self.is_running:
            logger.error(f"{self.broker_name} API未连接，无法取消订单")
            return False
        
        try:
            # 发送API请求
            response = self._api_request("DELETE", f"/api/v1/orders/{order_id}")
            
            if not response or "success" not in response or not response["success"]:
                error_message = response.get("message", "未知错误") if response else "API请求失败"
                logger.error(f"取消订单失败: {error_message}")
                return False
            
            # 更新订单状态
            if order_id in self.orders:
                self.orders[order_id].update(status="canceled", message="用户取消")
            
            if order_id in self.orders_cache:
                self.orders_cache[order_id].update(status="canceled", message="用户取消")
            
            logger.info(f"订单已取消 - ID: {order_id}")
            return True
        
        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Dict:
        """
        获取订单状态
        
        参数:
            order_id: 订单ID
            
        返回:
            订单状态字典
        """
        if not self.is_running:
            logger.error(f"{self.broker_name} API未连接，无法获取订单状态")
            return {}
        
        try:
            # 检查缓存
            if order_id in self.orders_cache:
                # 如果缓存时间不超过5秒，直接返回缓存
                if self.last_cache_update and (datetime.now() - self.last_cache_update).total_seconds() < 5:
                    return self.orders_cache[order_id].to_dict()
            
            # 发送API请求
            response = self._api_request("GET", f"/api/v1/orders/{order_id}")
            
            if not response or "success" not in response or not response["success"]:
                error_message = response.get("message", "未知错误") if response else "API请求失败"
                logger.error(f"获取订单状态失败: {error_message}")
                return {}
            
            # 获取订单数据
            order_data = response.get("data", {})
            
            # 更新订单缓存
            if order_data:
                order = Order.from_dict(order_data)
                self.orders_cache[order_id] = order
                
                # 同时更新订单字典
                self.orders[order_id] = order
            
            # 更新缓存时间
            self.last_cache_update = datetime.now()
            
            return order_data
        
        except Exception as e:
            logger.error(f"获取订单状态失败: {e}")
            return {}
    
    def get_positions(self) -> Dict[str, Position]:
        """
        获取当前持仓
        
        返回:
            持仓字典，键为交易品种代码
        """
        if not self.is_running:
            logger.error(f"{self.broker_name} API未连接，无法获取持仓")
            return {}
        
        try:
            # 检查缓存
            if self.positions_cache and self.last_cache_update and (datetime.now() - self.last_cache_update).total_seconds() < 5:
                return self.positions_cache
            
            # 发送API请求
            response = self._api_request("GET", "/api/v1/positions")
            
            if not response or "success" not in response or not response["success"]:
                error_message = response.get("message", "未知错误") if response else "API请求失败"
                logger.error(f"获取持仓失败: {error_message}")
                return self.positions_cache  # 返回缓存的持仓
            
            # 获取持仓数据
            positions_data = response.get("data", [])
            
            # 更新持仓缓存
            self.positions_cache = {
                position["symbol"]: Position(
                    symbol=position["symbol"],
                    quantity=position["quantity"],
                    average_price=position["average_price"],
                    current_price=position["current_price"],
                    open_time=datetime.fromisoformat(position["open_time"]) if "open_time" in position else datetime.now(),
                    realized_pnl=position.get("realized_pnl", 0),
                    unrealized_pnl=position.get("unrealized_pnl", 0)
                )
                for position in positions_data
            }
            
            # 同时更新持仓字典
            self.positions = self.positions_cache.copy()
            
            # 更新缓存时间
            self.last_cache_update = datetime.now()
            
            return self.positions_cache
        
        except Exception as e:
            logger.error(f"获取持仓失败: {e}")
            return self.positions_cache  # 返回缓存的持仓
    
    def get_account_info(self) -> Dict:
        """
        获取账户信息
        
        返回:
            账户信息字典
        """
        if not self.is_running:
            logger.error(f"{self.broker_name} API未连接，无法获取账户信息")
            return {}
        
        try:
            # 检查缓存
            if self.account_info_cache and self.last_cache_update and (datetime.now() - self.last_cache_update).total_seconds() < 5:
                return self.account_info_cache
            
            # 发送API请求
            response = self._api_request("GET", "/api/v1/account")
            
            if not response or "success" not in response or not response["success"]:
                error_message = response.get("message", "未知错误") if response else "API请求失败"
                logger.error(f"获取账户信息失败: {error_message}")
                return self.account_info_cache  # 返回缓存的账户信息
            
            # 获取账户数据
            account_data = response.get("data", {})
            
            # 更新账户信息缓存
            self.account_info_cache = account_data
            
            # 同时更新账户信息
            self.account_info = {
                "cash": account_data.get("cash", 0),
                "total_value": account_data.get("total_value", 0),
                "margin": account_data.get("margin", 0),
                "realized_pnl": account_data.get("realized_pnl", 0),
                "unrealized_pnl": account_data.get("unrealized_pnl", 0)
            }
            
            # 更新缓存时间
            self.last_cache_update = datetime.now()
            
            return self.account_info_cache
        
        except Exception as e:
            logger.error(f"获取账户信息失败: {e}")
            return self.account_info_cache  # 返回缓存的账户信息
    
    def _api_request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """
        发送API请求
        
        参数:
            method: 请求方法，如'GET', 'POST', 'DELETE'
            endpoint: API端点
            params: 请求参数
            
        返回:
            响应数据字典
        """
        if not self.api_base_url:
            logger.error("API基础URL未设置")
            return {}
        
        url = f"{self.api_base_url}{endpoint}"
        
        # 准备请求头
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": self.api_key
        }
        
        # 添加签名
        timestamp = str(int(time.time() * 1000))
        signature_payload = f"{timestamp}{method}{endpoint}"
        
        if params:
            if method == "GET":
                signature_payload += "?" + urlencode(params)
            else:
                signature_payload += json.dumps(params)
        
        signature = hmac.new(
            self.api_secret.encode(),
            signature_payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        headers["X-TIMESTAMP"] = timestamp
        headers["X-SIGNATURE"] = signature
        
        try:
            # 发送请求
            if method == "GET":
                response = self.session.get(url, params=params, headers=headers)
            elif method == "POST":
                response = self.session.post(url, json=params, headers=headers)
            elif method == "DELETE":
                response = self.session.delete(url, headers=headers)
            else:
                logger.error(f"不支持的请求方法: {method}")
                return {}
            
            # 检查响应状态
            response.raise_for_status()
            
            # 解析响应数据
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API请求失败: {e}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"解析API响应失败: {e}")
            return {}
    
    def _connect_websocket(self):
        """连接WebSocket"""
        if self.ws_connected:
            logger.warning("WebSocket已连接")
            return
        
        # 获取WebSocket URL
        ws_url = self.config.get("ws_url", "")
        
        if not ws_url:
            logger.warning("WebSocket URL未设置，跳过WebSocket连接")
            return
        
        try:
            # 生成认证信息
            timestamp = str(int(time.time() * 1000))
            signature = hmac.new(
                self.api_secret.encode(),
                f"{timestamp}websocket-connect".encode(),
                hashlib.sha256
            ).hexdigest()
            
            # 连接WebSocket
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_open=lambda ws: self._on_ws_open(ws, timestamp, signature),
                on_message=self._on_ws_message,
                on_error=self._on_ws_error,
                on_close=self._on_ws_close
            )
            
            # 启动WebSocket线程
            self.ws_thread = threading.Thread(target=self.ws.run_forever, kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}})
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            logger.info("WebSocket连接已启动")
        
        except Exception as e:
            logger.error(f"连接WebSocket失败: {e}")
    
    def _disconnect_websocket(self):
        """断开WebSocket连接"""
        if not self.ws_connected:
            return
        
        try:
            # 关闭WebSocket连接
            if self.ws:
                self.ws.close()
            
            # 等待WebSocket线程结束
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=5)
            
            self.ws_connected = False
            logger.info("WebSocket连接已关闭")
        
        except Exception as e:
            logger.error(f"断开WebSocket连接失败: {e}")
    
    def _on_ws_open(self, ws, timestamp, signature):
        """
        WebSocket连接打开回调
        
        参数:
            ws: WebSocket对象
            timestamp: 时间戳
            signature: 签名
        """
        logger.info("WebSocket连接已打开")
        
        # 发送认证消息
        auth_message = {
            "type": "auth",
            "api_key": self.api_key,
            "timestamp": timestamp,
            "signature": signature
        }
        
        ws.send(json.dumps(auth_message))
        
        # 订阅账户和订单更新
        subscribe_message = {
            "type": "subscribe",
            "channels": ["account", "orders", "positions"]
        }
        
        ws.send(json.dumps(subscribe_message))
        
        self.ws_connected = True
    
    def _on_ws_message(self, ws, message):
        """
        WebSocket消息回调
        
        参数:
            ws: WebSocket对象
            message: 消息内容
        """
        try:
            data = json.loads(message)
            
            # 处理不同类型的消息
            if "type" in data:
                if data["type"] == "auth":
                    # 认证响应
                    if data.get("status") == "success":
                        logger.info("WebSocket认证成功")
                    else:
                        logger.error(f"WebSocket认证失败: {data.get('message', '未知错误')}")
                
                elif data["type"] == "subscribe":
                    # 订阅响应
                    if data.get("status") == "success":
                        logger.info(f"WebSocket订阅成功: {data.get('channels', [])}")
                    else:
                        logger.error(f"WebSocket订阅失败: {data.get('message', '未知错误')}")
                
                elif data["type"] == "account":
                    # 账户更新
                    self.account_info_cache = data.get("data", {})
                    self.account_info = {
                        "cash": self.account_info_cache.get("cash", 0),
                        "total_value": self.account_info_cache.get("total_value", 0),
                        "margin": self.account_info_cache.get("margin", 0),
                        "realized_pnl": self.account_info_cache.get("realized_pnl", 0),
                        "unrealized_pnl": self.account_info_cache.get("unrealized_pnl", 0)
                    }
                    self.last_cache_update = datetime.now()
                
                elif data["type"] == "order":
                    # 订单更新
                    order_data = data.get("data", {})
                    if order_data and "order_id" in order_data:
                        order = Order.from_dict(order_data)
                        self.orders_cache[order.order_id] = order
                        self.orders[order.order_id] = order
                        self.last_cache_update = datetime.now()
                        
                        # 记录成交日志
                        if order.status == "filled" and order.filled_quantity > 0:
                            self.log_trade(
                                trade_type=order.direction,
                                symbol=order.symbol,
                                quantity=order.filled_quantity,
                                price=order.filled_price,
                                order_id=order.order_id,
                                commission=order.commission,
                                message=f"订单已成交，{'买入' if order.direction == 'buy' else '卖出'} {order.filled_quantity} {order.symbol} @ {order.filled_price}"
                            )
                
                elif data["type"] == "position":
                    # 持仓更新
                    position_data = data.get("data", {})
                    if position_data and "symbol" in position_data:
                        position = Position(
                            symbol=position_data["symbol"],
                            quantity=position_data["quantity"],
                            average_price=position_data["average_price"],
                            current_price=position_data["current_price"],
                            open_time=datetime.fromisoformat(position_data["open_time"]) if "open_time" in position_data else datetime.now(),
                            realized_pnl=position_data.get("realized_pnl", 0),
                            unrealized_pnl=position_data.get("unrealized_pnl", 0)
                        )
                        
                        if position.quantity > 0:
                            self.positions_cache[position.symbol] = position
                            self.positions[position.symbol] = position
                        else:
                            # 如果持仓为0，删除持仓记录
                            if position.symbol in self.positions_cache:
                                del self.positions_cache[position.symbol]
                            if position.symbol in self.positions:
                                del self.positions[position.symbol]
                        
                        self.last_cache_update = datetime.now()
        
        except json.JSONDecodeError as e:
            logger.error(f"解析WebSocket消息失败: {e}")
        except Exception as e:
            logger.error(f"处理WebSocket消息失败: {e}")
    
    def _on_ws_error(self, ws, error):
        """
        WebSocket错误回调
        
        参数:
            ws: WebSocket对象
            error: 错误信息
        """
        logger.error(f"WebSocket错误: {error}")
    
    def _on_ws_close(self, ws, close_status_code, close_msg):
        """
        WebSocket关闭回调
        
        参数:
            ws: WebSocket对象
            close_status_code: 关闭状态码
            close_msg: 关闭消息
        """
        logger.info(f"WebSocket连接已关闭: {close_status_code} {close_msg}")
        self.ws_connected = False


# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 测试模拟交易执行器
    try:
        # 初始化模拟交易执行器
        config = {
            "initial_cash": 1000000,
            "commission_rate": 0.0003,
            "slippage": 0.0001,
            "tax_rate": 0.001
        }
        
        executor = SimulatedTradeExecutor(config)
        
        # 连接交易系统
        executor.connect()
        
        # 更新市场数据
        market_data = {
            "open": 100,
            "high": 105,
            "low": 95,
            "close": 102,
            "volume": 1000000
        }
        
        executor.update_market_data("AAPL", market_data)
        
        # 创建买入订单
        buy_order = Order(
            symbol="AAPL",
            order_type="market",
            direction="buy",
            quantity=100
        )
        
        # 下单
        order_id = executor.place_order(buy_order)
        
        # 等待订单处理
        time.sleep(1)
        
        # 获取订单状态
        order_status = executor.get_order_status(order_id)
        print(f"订单状态: {order_status}")
        
        # 获取持仓
        positions = executor.get_positions()
        print(f"持仓: {positions}")
        
        # 获取账户信息
        account_info = executor.get_account_info()
        print(f"账户信息: {account_info}")
        
        # 更新市场数据（价格上涨）
        market_data["close"] = 110
        executor.update_market_data("AAPL", market_data)
        
        # 获取更新后的持仓和账户信息
        positions = executor.get_positions()
        account_info = executor.get_account_info()
        print(f"更新后持仓: {positions}")
        print(f"更新后账户信息: {account_info}")
        
        # 创建卖出订单
        sell_order = Order(
            symbol="AAPL",
            order_type="market",
            direction="sell",
            quantity=50
        )
        
        # 下单
        order_id = executor.place_order(sell_order)
        
        # 等待订单处理
        time.sleep(1)
        
        # 获取订单状态
        order_status = executor.get_order_status(order_id)
        print(f"卖出订单状态: {order_status}")
        
        # 获取更新后的持仓和账户信息
        positions = executor.get_positions()
        account_info = executor.get_account_info()
        print(f"卖出后持仓: {positions}")
        print(f"卖出后账户信息: {account_info}")
        
        # 获取交易摘要
        trade_summary = executor.get_trade_summary()
        print(f"交易摘要: {trade_summary}")
        
        # 断开连接
        executor.disconnect()
        
    except Exception as e:
        logging.error(f"测试过程中发生错误: {e}")
