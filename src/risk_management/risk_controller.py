"""
风险控制模块

本模块实现了量化交易系统的风险控制功能，包括：
1. 止盈止损策略
2. 仓位管理
3. 风险度量与监控
4. 交易限制与预警

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

# 配置日志
logger = logging.getLogger(__name__)

class RiskController:
    """风险控制器，负责管理交易风险"""
    
    def __init__(self, config: Dict = None):
        """
        初始化风险控制器
        
        参数:
            config: 配置信息
        """
        self.config = config or {}
        
        # 风险控制参数
        self.max_position_size = self.config.get("max_position_size", 0.2)  # 单个持仓最大比例
        self.max_total_position = self.config.get("max_total_position", 0.8)  # 总持仓最大比例
        self.stop_loss_pct = self.config.get("stop_loss_pct", 0.05)  # 止损百分比
        self.take_profit_pct = self.config.get("take_profit_pct", 0.1)  # 止盈百分比
        self.max_drawdown = self.config.get("max_drawdown", 0.2)  # 最大回撤限制
        self.volatility_threshold = self.config.get("volatility_threshold", 0.03)  # 波动率阈值
        self.max_trades_per_day = self.config.get("max_trades_per_day", 10)  # 每日最大交易次数
        self.min_holding_period = self.config.get("min_holding_period", 1)  # 最小持仓周期（天）
        
        # 风险监控状态
        self.risk_status = {
            "total_position_ratio": 0,  # 总持仓比例
            "max_single_position_ratio": 0,  # 最大单个持仓比例
            "current_drawdown": 0,  # 当前回撤
            "daily_trade_count": 0,  # 当日交易次数
            "portfolio_volatility": 0,  # 组合波动率
            "risk_level": "low",  # 风险等级：low, medium, high, extreme
            "warnings": []  # 风险警告列表
        }
        
        # 历史数据
        self.portfolio_values = []  # 组合价值历史
        self.position_history = {}  # 持仓历史
        self.trade_history = []  # 交易历史
        
        # 最高水位线
        self.high_water_mark = 0
        
        logger.info("风险控制器初始化成功")
    
    def update_portfolio_status(self, account_info: Dict, positions: Dict):
        """
        更新投资组合状态
        
        参数:
            account_info: 账户信息
            positions: 持仓信息
        """
        # 计算总资产
        total_value = account_info.get("total_value", 0)
        cash = account_info.get("cash", 0)
        
        # 更新最高水位线
        if total_value > self.high_water_mark:
            self.high_water_mark = total_value
        
        # 计算当前回撤
        if self.high_water_mark > 0:
            current_drawdown = (self.high_water_mark - total_value) / self.high_water_mark
            self.risk_status["current_drawdown"] = current_drawdown
        
        # 计算持仓比例
        position_value = total_value - cash
        if total_value > 0:
            total_position_ratio = position_value / total_value
            self.risk_status["total_position_ratio"] = total_position_ratio
        
        # 计算单个持仓最大比例
        max_single_position_ratio = 0
        for symbol, position in positions.items():
            position_ratio = (position.quantity * position.current_price) / total_value if total_value > 0 else 0
            if position_ratio > max_single_position_ratio:
                max_single_position_ratio = position_ratio
        
        self.risk_status["max_single_position_ratio"] = max_single_position_ratio
        
        # 记录组合价值历史
        self.portfolio_values.append({
            "timestamp": datetime.now().isoformat(),
            "total_value": total_value,
            "cash": cash,
            "position_value": position_value
        })
        
        # 计算组合波动率（如果有足够的历史数据）
        if len(self.portfolio_values) >= 10:
            values = [entry["total_value"] for entry in self.portfolio_values[-10:]]
            returns = np.diff(values) / values[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # 年化波动率
            self.risk_status["portfolio_volatility"] = volatility
        
        # 更新风险等级
        self._update_risk_level()
        
        # 检查风险警告
        self._check_risk_warnings()
        
        logger.info(f"投资组合状态已更新 - 总价值: {total_value}, 持仓比例: {total_position_ratio:.2f}, 当前回撤: {self.risk_status['current_drawdown']:.2f}")
    
    def check_trade_risk(self, symbol: str, direction: str, quantity: float, price: float, 
                         account_info: Dict, positions: Dict) -> Tuple[bool, str]:
        """
        检查交易风险
        
        参数:
            symbol: 交易品种代码
            direction: 交易方向，'buy'或'sell'
            quantity: 交易数量
            price: 交易价格
            account_info: 账户信息
            positions: 持仓信息
            
        返回:
            (是否允许交易, 拒绝原因)
        """
        # 计算总资产
        total_value = account_info.get("total_value", 0)
        cash = account_info.get("cash", 0)
        
        # 检查每日交易次数限制
        if self.risk_status["daily_trade_count"] >= self.max_trades_per_day:
            return False, f"超过每日最大交易次数限制 ({self.max_trades_per_day})"
        
        # 买入风险检查
        if direction == "buy":
            # 检查资金是否足够
            trade_value = quantity * price
            if trade_value > cash:
                return False, "可用资金不足"
            
            # 检查单个持仓限制
            current_position_value = 0
            if symbol in positions:
                current_position_value = positions[symbol].quantity * positions[symbol].current_price
            
            new_position_value = current_position_value + trade_value
            new_position_ratio = new_position_value / total_value if total_value > 0 else 0
            
            if new_position_ratio > self.max_position_size:
                return False, f"超过单个持仓最大比例限制 ({self.max_position_size:.2f})"
            
            # 检查总持仓限制
            position_value = total_value - cash
            new_total_position_value = position_value + trade_value
            new_total_position_ratio = new_total_position_value / total_value if total_value > 0 else 0
            
            if new_total_position_ratio > self.max_total_position:
                return False, f"超过总持仓最大比例限制 ({self.max_total_position:.2f})"
            
            # 检查风险等级
            if self.risk_status["risk_level"] == "extreme":
                return False, "当前风险等级过高 (extreme)"
        
        # 卖出风险检查
        elif direction == "sell":
            # 检查持仓是否足够
            if symbol not in positions or positions[symbol].quantity < quantity:
                return False, "持仓不足"
            
            # 检查最小持仓周期
            position = positions[symbol]
            holding_days = (datetime.now() - position.open_time).days
            
            if holding_days < self.min_holding_period:
                return False, f"未达到最小持仓周期 ({self.min_holding_period} 天)"
        
        # 增加交易计数
        self.risk_status["daily_trade_count"] += 1
        
        # 记录交易历史
        self.trade_history.append({
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "price": price,
            "value": quantity * price
        })
        
        return True, ""
    
    def generate_stop_orders(self, positions: Dict) -> List[Dict]:
        """
        生成止盈止损订单
        
        参数:
            positions: 持仓信息
            
        返回:
            止盈止损订单列表
        """
        stop_orders = []
        
        for symbol, position in positions.items():
            # 计算止损价格
            stop_loss_price = position.average_price * (1 - self.stop_loss_pct)
            
            # 计算止盈价格
            take_profit_price = position.average_price * (1 + self.take_profit_pct)
            
            # 创建止损订单
            stop_loss_order = {
                "symbol": symbol,
                "order_type": "stop",
                "direction": "sell",
                "quantity": position.quantity,
                "stop_price": stop_loss_price,
                "reason": "stop_loss"
            }
            
            # 创建止盈订单
            take_profit_order = {
                "symbol": symbol,
                "order_type": "limit",
                "direction": "sell",
                "quantity": position.quantity,
                "price": take_profit_price,
                "reason": "take_profit"
            }
            
            stop_orders.append(stop_loss_order)
            stop_orders.append(take_profit_order)
        
        return stop_orders
    
    def calculate_position_size(self, symbol: str, price: float, account_info: Dict, 
                                risk_per_trade: float = None) -> float:
        """
        计算建议的仓位大小
        
        参数:
            symbol: 交易品种代码
            price: 交易价格
            account_info: 账户信息
            risk_per_trade: 每笔交易风险比例，如果为None则使用默认值
            
        返回:
            建议的交易数量
        """
        # 获取账户总价值
        total_value = account_info.get("total_value", 0)
        
        # 获取每笔交易风险比例
        if risk_per_trade is None:
            risk_per_trade = self.config.get("risk_per_trade", 0.01)  # 默认每笔交易风险1%
        
        # 计算风险金额
        risk_amount = total_value * risk_per_trade
        
        # 计算止损点数
        stop_loss_points = price * self.stop_loss_pct
        
        # 计算建议的交易数量
        if stop_loss_points > 0:
            quantity = risk_amount / stop_loss_points
        else:
            # 如果无法计算止损点数，使用默认方法
            quantity = (total_value * self.max_position_size) / price
        
        # 确保数量为正
        quantity = max(0, quantity)
        
        # 确保不超过单个持仓限制
        max_quantity = (total_value * self.max_position_size) / price
        quantity = min(quantity, max_quantity)
        
        return quantity
    
    def reset_daily_counters(self):
        """重置每日计数器"""
        self.risk_status["daily_trade_count"] = 0
        logger.info("每日风险计数器已重置")
    
    def get_risk_report(self) -> Dict:
        """
        获取风险报告
        
        返回:
            风险报告字典
        """
        # 计算历史最大回撤
        max_drawdown = 0
        if len(self.portfolio_values) > 1:
            values = [entry["total_value"] for entry in self.portfolio_values]
            peak = values[0]
            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak if peak > 0 else 0
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        
        # 计算夏普比率（如果有足够的历史数据）
        sharpe_ratio = None
        if len(self.portfolio_values) >= 30:
            values = [entry["total_value"] for entry in self.portfolio_values]
            returns = np.diff(values) / values[:-1]
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            risk_free_rate = self.config.get("risk_free_rate", 0.02) / 252  # 日化无风险利率
            
            if std_return > 0:
                sharpe_ratio = (avg_return - risk_free_rate) / std_return * np.sqrt(252)  # 年化夏普比率
        
        # 计算交易统计
        win_trades = 0
        loss_trades = 0
        total_profit = 0
        total_loss = 0
        
        for trade in self.trade_history:
            if trade["direction"] == "sell":
                # 简化的盈亏计算，实际应该考虑持仓成本
                if "profit" in trade and trade["profit"] > 0:
                    win_trades += 1
                    total_profit += trade["profit"]
                else:
                    loss_trades += 1
                    total_loss += abs(trade.get("profit", 0))
        
        total_trades = win_trades + loss_trades
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        return {
            "risk_level": self.risk_status["risk_level"],
            "current_drawdown": self.risk_status["current_drawdown"],
            "max_drawdown": max_drawdown,
            "portfolio_volatility": self.risk_status["portfolio_volatility"],
            "sharpe_ratio": sharpe_ratio,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_position_ratio": self.risk_status["total_position_ratio"],
            "max_single_position_ratio": self.risk_status["max_single_position_ratio"],
            "warnings": self.risk_status["warnings"],
            "timestamp": datetime.now().isoformat()
        }
    
    def save_risk_data(self, file_path: str):
        """
        保存风险数据
        
        参数:
            file_path: 保存路径
        """
        data = {
            "risk_status": self.risk_status,
            "portfolio_values": self.portfolio_values,
            "trade_history": self.trade_history,
            "high_water_mark": self.high_water_mark,
            "config": self.config
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            logger.info(f"风险数据已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存风险数据失败: {e}")
    
    def load_risk_data(self, file_path: str):
        """
        加载风险数据
        
        参数:
            file_path: 加载路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.risk_status = data.get("risk_status", self.risk_status)
            self.portfolio_values = data.get("portfolio_values", [])
            self.trade_history = data.get("trade_history", [])
            self.high_water_mark = data.get("high_water_mark", 0)
            
            logger.info(f"风险数据已从 {file_path} 加载")
        except Exception as e:
            logger.error(f"加载风险数据失败: {e}")
    
    def _update_risk_level(self):
        """更新风险等级"""
        # 初始化风险分数
        risk_score = 0
        
        # 根据总持仓比例评估风险
        total_position_ratio = self.risk_status["total_position_ratio"]
        if total_position_ratio > self.max_total_position:
            risk_score += 3
        elif total_position_ratio > self.max_total_position * 0.8:
            risk_score += 2
        elif total_position_ratio > self.max_total_position * 0.6:
            risk_score += 1
        
        # 根据单个持仓最大比例评估风险
        max_single_position_ratio = self.risk_status["max_single_position_ratio"]
        if max_single_position_ratio > self.max_position_size:
            risk_score += 3
        elif max_single_position_ratio > self.max_position_size * 0.8:
            risk_score += 2
        elif max_single_position_ratio > self.max_position_size * 0.6:
            risk_score += 1
        
        # 根据当前回撤评估风险
        current_drawdown = self.risk_status["current_drawdown"]
        if current_drawdown > self.max_drawdown:
            risk_score += 3
        elif current_drawdown > self.max_drawdown * 0.8:
            risk_score += 2
        elif current_drawdown > self.max_drawdown * 0.6:
            risk_score += 1
        
        # 根据组合波动率评估风险
        portfolio_volatility = self.risk_status["portfolio_volatility"]
        if portfolio_volatility > self.volatility_threshold:
            risk_score += 2
        elif portfolio_volatility > self.volatility_threshold * 0.8:
            risk_score += 1
        
        # 根据风险分数确定风险等级
        if risk_score >= 8:
            self.risk_status["risk_level"] = "extreme"
        elif risk_score >= 5:
            self.risk_status["risk_level"] = "high"
        elif risk_score >= 3:
            self.risk_status["risk_level"] = "medium"
        else:
            self.risk_status["risk_level"] = "low"
    
    def _check_risk_warnings(self):
        """检查风险警告"""
        warnings = []
        
        # 检查总持仓比例
        total_position_ratio = self.risk_status["total_position_ratio"]
        if total_position_ratio > self.max_total_position:
            warnings.append(f"总持仓比例 ({total_position_ratio:.2f}) 超过限制 ({self.max_total_position:.2f})")
        
        # 检查单个持仓最大比例
        max_single_position_ratio = self.risk_status["max_single_position_ratio"]
        if max_single_position_ratio > self.max_position_size:
            warnings.append(f"单个持仓最大比例 ({max_single_position_ratio:.2f}) 超过限制 ({self.max_position_size:.2f})")
        
        # 检查当前回撤
        current_drawdown = self.risk_status["current_drawdown"]
        if current_drawdown > self.max_drawdown:
            warnings.append(f"当前回撤 ({current_drawdown:.2f}) 超过限制 ({self.max_drawdown:.2f})")
        
        # 检查组合波动率
        portfolio_volatility = self.risk_status["portfolio_volatility"]
        if portfolio_volatility > self.volatility_threshold:
            warnings.append(f"组合波动率 ({portfolio_volatility:.2f}) 超过阈值 ({self.volatility_threshold:.2f})")
        
        # 更新风险警告列表
        self.risk_status["warnings"] = warnings


class PositionSizer:
    """仓位管理器，负责计算最优仓位大小"""
    
    def __init__(self, config: Dict = None):
        """
        初始化仓位管理器
        
        参数:
            config: 配置信息
        """
        self.config = config or {}
        
        # 仓位管理参数
        self.default_risk_per_trade = self.config.get("risk_per_trade", 0.01)  # 默认每笔交易风险1%
        self.max_position_size = self.config.get("max_position_size", 0.2)  # 单个持仓最大比例
        self.position_sizing_method = self.config.get("position_sizing_method", "fixed_risk")  # 仓位管理方法
        self.kelly_fraction = self.config.get("kelly_fraction", 0.5)  # 凯利公式分数
        
        logger.info("仓位管理器初始化成功")
    
    def calculate_position_size(self, symbol: str, price: float, stop_loss_price: float, 
                                account_info: Dict, win_rate: float = None, 
                                reward_risk_ratio: float = None) -> float:
        """
        计算建议的仓位大小
        
        参数:
            symbol: 交易品种代码
            price: 交易价格
            stop_loss_price: 止损价格
            account_info: 账户信息
            win_rate: 胜率，如果为None则使用默认值
            reward_risk_ratio: 盈亏比，如果为None则使用默认值
            
        返回:
            建议的交易数量
        """
        # 获取账户总价值
        total_value = account_info.get("total_value", 0)
        
        # 计算止损点数
        stop_loss_points = abs(price - stop_loss_price)
        
        # 根据不同的仓位管理方法计算仓位大小
        if self.position_sizing_method == "fixed_risk":
            # 固定风险法
            risk_amount = total_value * self.default_risk_per_trade
            quantity = risk_amount / stop_loss_points if stop_loss_points > 0 else 0
        
        elif self.position_sizing_method == "fixed_ratio":
            # 固定比例法
            quantity = (total_value * self.max_position_size) / price
        
        elif self.position_sizing_method == "kelly":
            # 凯利公式法
            if win_rate is None or reward_risk_ratio is None:
                # 如果没有提供胜率和盈亏比，使用固定风险法
                risk_amount = total_value * self.default_risk_per_trade
                quantity = risk_amount / stop_loss_points if stop_loss_points > 0 else 0
            else:
                # 计算凯利比例
                kelly_pct = win_rate - (1 - win_rate) / reward_risk_ratio
                kelly_pct = max(0, kelly_pct)  # 确保不为负
                
                # 应用凯利分数（通常使用半凯利或四分之一凯利）
                kelly_pct *= self.kelly_fraction
                
                # 计算仓位大小
                risk_amount = total_value * kelly_pct
                quantity = risk_amount / stop_loss_points if stop_loss_points > 0 else 0
        
        elif self.position_sizing_method == "volatility":
            # 波动率调整法
            # 这里简化处理，实际应该考虑历史波动率
            volatility = self.config.get("volatility", 0.02)  # 默认波动率2%
            risk_amount = total_value * self.default_risk_per_trade
            adjusted_risk = risk_amount * (0.02 / volatility)  # 根据波动率调整风险
            quantity = adjusted_risk / stop_loss_points if stop_loss_points > 0 else 0
        
        else:
            # 默认使用固定风险法
            risk_amount = total_value * self.default_risk_per_trade
            quantity = risk_amount / stop_loss_points if stop_loss_points > 0 else 0
        
        # 确保数量为正
        quantity = max(0, quantity)
        
        # 确保不超过单个持仓限制
        max_quantity = (total_value * self.max_position_size) / price
        quantity = min(quantity, max_quantity)
        
        return quantity
    
    def calculate_position_adjustment(self, symbol: str, current_position: float, 
                                      price: float, account_info: Dict, 
                                      market_condition: str = "normal") -> float:
        """
        计算持仓调整量
        
        参数:
            symbol: 交易品种代码
            current_position: 当前持仓数量
            price: 当前价格
            account_info: 账户信息
            market_condition: 市场状况，可以是'normal', 'bullish', 'bearish'
            
        返回:
            建议的调整数量（正数表示增加，负数表示减少）
        """
        # 获取账户总价值
        total_value = account_info.get("total_value", 0)
        
        # 计算当前持仓价值
        current_position_value = current_position * price
        
        # 计算当前持仓比例
        current_position_ratio = current_position_value / total_value if total_value > 0 else 0
        
        # 根据市场状况调整目标持仓比例
        target_ratio = self.max_position_size  # 默认目标比例
        
        if market_condition == "bullish":
            # 牛市增加持仓
            target_ratio = min(self.max_position_size * 1.2, 0.25)
        elif market_condition == "bearish":
            # 熊市减少持仓
            target_ratio = self.max_position_size * 0.5
        
        # 计算目标持仓价值
        target_position_value = total_value * target_ratio
        
        # 计算目标持仓数量
        target_position = target_position_value / price if price > 0 else 0
        
        # 计算调整数量
        adjustment = target_position - current_position
        
        return adjustment
    
    def calculate_pyramid_position(self, symbol: str, entry_price: float, current_price: float, 
                                  current_position: float, account_info: Dict, 
                                  num_levels: int = 3) -> List[Dict]:
        """
        计算金字塔加仓策略的仓位
        
        参数:
            symbol: 交易品种代码
            entry_price: 初始入场价格
            current_price: 当前价格
            current_position: 当前持仓数量
            account_info: 账户信息
            num_levels: 金字塔层数
            
        返回:
            加仓计划列表，每个元素包含价格和数量
        """
        # 获取账户总价值
        total_value = account_info.get("total_value", 0)
        
        # 只有在价格上涨时才考虑金字塔加仓
        if current_price <= entry_price:
            return []
        
        # 计算价格增长百分比
        price_increase_pct = (current_price - entry_price) / entry_price
        
        # 如果价格增长不足5%，不考虑加仓
        if price_increase_pct < 0.05:
            return []
        
        # 计算每层价格增长阈值
        level_thresholds = [0.05 + 0.05 * i for i in range(num_levels)]
        
        # 计算当前应该处于哪一层
        current_level = 0
        for i, threshold in enumerate(level_thresholds):
            if price_increase_pct >= threshold:
                current_level = i + 1
        
        # 如果已经超过最高层，不再加仓
        if current_level == 0:
            return []
        
        # 计算初始仓位价值
        initial_position_value = entry_price * current_position
        
        # 计算每层加仓比例（递减）
        level_ratios = [1.0]
        for i in range(1, num_levels):
            level_ratios.append(level_ratios[i-1] * 0.7)  # 每层减少30%
        
        # 计算每层加仓数量
        pyramid_plan = []
        for i in range(current_level):
            level_price = entry_price * (1 + level_thresholds[i])
            level_value = initial_position_value * level_ratios[i]
            level_quantity = level_value / level_price
            
            pyramid_plan.append({
                "level": i + 1,
                "price": level_price,
                "quantity": level_quantity
            })
        
        return pyramid_plan


# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 测试风险控制器
    try:
        # 初始化风险控制器
        config = {
            "max_position_size": 0.2,
            "max_total_position": 0.8,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.1,
            "max_drawdown": 0.2,
            "volatility_threshold": 0.03,
            "max_trades_per_day": 10,
            "min_holding_period": 1,
            "risk_per_trade": 0.01,
            "risk_free_rate": 0.02
        }
        
        risk_controller = RiskController(config)
        
        # 模拟账户信息
        account_info = {
            "total_value": 1000000,
            "cash": 800000,
            "margin": 0,
            "realized_pnl": 0,
            "unrealized_pnl": 0
        }
        
        # 模拟持仓信息
        from collections import namedtuple
        Position = namedtuple('Position', ['symbol', 'quantity', 'average_price', 'current_price', 'open_time', 'realized_pnl', 'unrealized_pnl'])
        
        positions = {
            "AAPL": Position("AAPL", 100, 150, 160, datetime.now() - timedelta(days=5), 0, 1000),
            "MSFT": Position("MSFT", 50, 300, 310, datetime.now() - timedelta(days=3), 0, 500)
        }
        
        # 更新投资组合状态
        risk_controller.update_portfolio_status(account_info, positions)
        
        # 检查交易风险
        symbol = "GOOGL"
        direction = "buy"
        quantity = 10
        price = 2500
        
        allowed, reason = risk_controller.check_trade_risk(symbol, direction, quantity, price, account_info, positions)
        print(f"交易是否允许: {allowed}, 原因: {reason}")
        
        # 生成止盈止损订单
        stop_orders = risk_controller.generate_stop_orders(positions)
        print(f"止盈止损订单: {stop_orders}")
        
        # 计算建议的仓位大小
        suggested_quantity = risk_controller.calculate_position_size(symbol, price, account_info)
        print(f"建议的仓位大小: {suggested_quantity}")
        
        # 获取风险报告
        risk_report = risk_controller.get_risk_report()
        print(f"风险报告: {risk_report}")
        
        # 测试仓位管理器
        position_sizer = PositionSizer(config)
        
        # 计算建议的仓位大小
        stop_loss_price = 2400
        suggested_quantity = position_sizer.calculate_position_size(symbol, price, stop_loss_price, account_info, 0.6, 2.0)
        print(f"建议的仓位大小 (仓位管理器): {suggested_quantity}")
        
        # 计算持仓调整量
        current_position = 5
        adjustment = position_sizer.calculate_position_adjustment(symbol, current_position, price, account_info, "bullish")
        print(f"建议的调整数量: {adjustment}")
        
        # 计算金字塔加仓策略
        entry_price = 2300
        pyramid_plan = position_sizer.calculate_pyramid_position(symbol, entry_price, price, current_position, account_info)
        print(f"金字塔加仓计划: {pyramid_plan}")
        
    except Exception as e:
        logging.error(f"测试过程中发生错误: {e}")
