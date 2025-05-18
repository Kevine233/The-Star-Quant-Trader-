"""
回测引擎模块

提供基础的回测引擎实现，用于评估交易策略的历史表现。

日期：2025-05-17
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import datetime
import logging
import os

# 配置日志
logger = logging.getLogger(__name__)

class 回测引擎:
    """
    回测引擎类，用于评估交易策略在历史数据上的表现。
    支持一键回测和自动参数优化功能。
    """
    
    def __init__(self, 
               初始资金: float = 100000.0,
               手续费率: float = 0.0003,
               滑点: float = 0.0001):
        """
        初始化回测引擎。
        
        参数:
            初始资金: 回测初始资金
            手续费率: 交易手续费率（买入和卖出都适用）
            滑点: 交易滑点率
        """
        self.初始资金 = 初始资金
        self.手续费率 = 手续费率
        self.滑点 = 滑点
        self.重置()
    
    def 重置(self):
        """重置回测状态。"""
        self.资金 = self.初始资金
        self.持仓 = {}  # 股票代码 -> 持仓数量
        self.持仓成本 = {}  # 股票代码 -> 平均成本
        self.交易记录 = []
        self.资金曲线 = []
        self.当前日期 = None
    
    def 执行回测(self, 
              市场数据: pd.DataFrame, 
              信号数据: pd.DataFrame,
              股票代码: str) -> Dict[str, Any]:
        """
        执行回测过程。
        
        参数:
            市场数据: 包含OHLCV数据的DataFrame
            信号数据: 包含交易信号的DataFrame
            股票代码: 交易标的代码
            
        返回:
            包含回测结果的字典
        """
        logger.info(f"开始回测 {股票代码}")
        
        # 重置回测状态
        self.重置()
        
        # 确保数据索引对齐
        共同索引 = 市场数据.index.intersection(信号数据.index)
        市场数据 = 市场数据.loc[共同索引]
        信号数据 = 信号数据.loc[共同索引]
        
        # 初始化结果记录
        self.资金曲线 = [{'日期': 市场数据.index[0], '资产': self.初始资金, '现金': self.初始资金, '持仓': 0.0}]
        
        # 遍历每个交易日
        for i in range(len(市场数据)):
            当前日期 = 市场数据.index[i]
            self.当前日期 = 当前日期
            
            # 获取当日价格和信号
            开盘价 = 市场数据['open'].iloc[i]
            最高价 = 市场数据['high'].iloc[i]
            最低价 = 市场数据['low'].iloc[i]
            收盘价 = 市场数据['close'].iloc[i]
            信号 = 信号数据['信号'].iloc[i]
            
            # 处理交易信号
            if 信号 == 1:  # 买入信号
                self._执行买入(股票代码, 收盘价, 当前日期, 信号数据.iloc[i])
            elif 信号 == -1:  # 卖出信号
                self._执行卖出(股票代码, 收盘价, 当前日期, 信号数据.iloc[i])
            
            # 更新资金曲线
            总资产 = self._计算总资产(股票代码, 收盘价)
            持仓价值 = 总资产 - self.资金
            self.资金曲线.append({
                '日期': 当前日期,
                '资产': 总资产,
                '现金': self.资金,
                '持仓': 持仓价值
            })
        
        # 计算回测结果指标
        结果 = self._计算回测指标()
        
        logger.info(f"回测完成，最终资产: {结果['最终资产']:.2f}，收益率: {结果['收益率']*100:.2f}%")
        return 结果
    
    def _执行买入(self, 股票代码: str, 价格: float, 日期: datetime.datetime, 信号行: pd.Series):
        """执行买入操作。"""
        # 考虑滑点
        实际价格 = 价格 * (1 + self.滑点)
        
        # 计算可买入数量（使用可用资金的90%，避免资金不足）
        可用资金 = self.资金 * 0.9
        数量 = int(可用资金 / 实际价格)
        
        if 数量 <= 0:
            logger.warning(f"{日期}: 资金不足，无法买入 {股票代码}")
            return
        
        # 计算交易成本
        交易金额 = 数量 * 实际价格
        手续费 = 交易金额 * self.手续费率
        总成本 = 交易金额 + 手续费
        
        # 检查资金是否足够
        if 总成本 > self.资金:
            数量 = int((self.资金 / (实际价格 * (1 + self.手续费率))))
            交易金额 = 数量 * 实际价格
            手续费 = 交易金额 * self.手续费率
            总成本 = 交易金额 + 手续费
        
        if 数量 <= 0:
            logger.warning(f"{日期}: 资金不足，无法买入 {股票代码}")
            return
        
        # 更新持仓和资金
        if 股票代码 in self.持仓:
            # 计算新的平均成本
            原持仓 = self.持仓[股票代码]
            原成本 = self.持仓成本[股票代码]
            新持仓 = 原持仓 + 数量
            新成本 = (原持仓 * 原成本 + 交易金额) / 新持仓
            self.持仓[股票代码] = 新持仓
            self.持仓成本[股票代码] = 新成本
        else:
            self.持仓[股票代码] = 数量
            self.持仓成本[股票代码] = 实际价格
        
        self.资金 -= 总成本
        
        # 记录交易
        交易记录 = {
            '日期': 日期,
            '类型': '买入',
            '股票代码': 股票代码,
            '价格': 实际价格,
            '数量': 数量,
            '交易金额': 交易金额,
            '手续费': 手续费,
            '剩余资金': self.资金,
            '信号解释': 信号行.get('信号解释', '')
        }
        self.交易记录.append(交易记录)
        
        logger.debug(f"{日期}: 买入 {股票代码} {数量}股，价格 {实际价格:.2f}，总成本 {总成本:.2f}")
    
    def _执行卖出(self, 股票代码: str, 价格: float, 日期: datetime.datetime, 信号行: pd.Series):
        """执行卖出操作。"""
        if 股票代码 not in self.持仓 or self.持仓[股票代码] <= 0:
            logger.warning(f"{日期}: 没有持仓，无法卖出 {股票代码}")
            return
        
        # 考虑滑点
        实际价格 = 价格 * (1 - self.滑点)
        
        # 卖出全部持仓
        数量 = self.持仓[股票代码]
        交易金额 = 数量 * 实际价格
        手续费 = 交易金额 * self.手续费率
        净收入 = 交易金额 - 手续费
        
        # 计算盈亏
        成本 = self.持仓成本[股票代码] * 数量
        盈亏 = 净收入 - 成本
        
        # 更新持仓和资金
        self.持仓[股票代码] = 0
        self.资金 += 净收入
        
        # 记录交易
        交易记录 = {
            '日期': 日期,
            '类型': '卖出',
            '股票代码': 股票代码,
            '价格': 实际价格,
            '数量': 数量,
            '交易金额': 交易金额,
            '手续费': 手续费,
            '净收入': 净收入,
            '盈亏': 盈亏,
            '剩余资金': self.资金,
            '信号解释': 信号行.get('信号解释', '')
        }
        self.交易记录.append(交易记录)
        
        logger.debug(f"{日期}: 卖出 {股票代码} {数量}股，价格 {实际价格:.2f}，净收入 {净收入:.2f}，盈亏 {盈亏:.2f}")
    
    def _计算总资产(self, 股票代码: str, 当前价格: float) -> float:
        """计算当前总资产。"""
        持仓价值 = 0.0
        if 股票代码 in self.持仓:
            持仓价值 = self.持仓[股票代码] * 当前价格
        
        return self.资金 + 持仓价值
    
    def _计算回测指标(self) -> Dict[str, Any]:
        """计算回测结果指标。"""
        # 转换资金曲线为DataFrame
        资金曲线 = pd.DataFrame(self.资金曲线)
        资金曲线['日期'] = pd.to_datetime(资金曲线['日期'])
        资金曲线 = 资金曲线.set_index('日期')
        
        # 计算基本指标
        初始资产 = 资金曲线['资产'].iloc[0]
        最终资产 = 资金曲线['资产'].iloc[-1]
        收益率 = (最终资产 / 初始资产) - 1
        
        # 计算年化收益率
        开始日期 = 资金曲线.index[0]
        结束日期 = 资金曲线.index[-1]
        交易天数 = (结束日期 - 开始日期).days
        if 交易天数 > 0:
            年化收益率 = (1 + 收益率) ** (365 / 交易天数) - 1
        else:
            年化收益率 = 0
        
        # 计算最大回撤
        资金曲线['累计最大值'] = 资金曲线['资产'].cummax()
        资金曲线['回撤'] = (资金曲线['资产'] - 资金曲线['累计最大值']) / 资金曲线['累计最大值']
        最大回撤 = abs(资金曲线['回撤'].min())
        
        # 计算收益回撤比
        收益回撤比 = 年化收益率 / 最大回撤 if 最大回撤 > 0 else float('inf')
        
        # 计算夏普比率（假设无风险利率为0）
        日收益率 = 资金曲线['资产'].pct_change().dropna()
        if len(日收益率) > 1:
            夏普比率 = np.sqrt(252) * 日收益率.mean() / 日收益率.std() if 日收益率.std() > 0 else 0
        else:
            夏普比率 = 0
        
        # 计算交易统计
        交易次数 = len(self.交易记录)
        if 交易次数 > 0:
            盈利交易 = [交易 for 交易 in self.交易记录 if 交易.get('类型') == '卖出' and 交易.get('盈亏', 0) > 0]
            亏损交易 = [交易 for 交易 in self.交易记录 if 交易.get('类型') == '卖出' and 交易.get('盈亏', 0) <= 0]
            
            盈利次数 = len(盈利交易)
            亏损次数 = len(亏损交易)
            
            胜率 = 盈利次数 / (盈利次数 + 亏损次数) if (盈利次数 + 亏损次数) > 0 else 0
            
            平均盈利 = sum(交易['盈亏'] for 交易 in 盈利交易) / 盈利次数 if 盈利次数 > 0 else 0
            平均亏损 = sum(交易['盈亏'] for 交易 in 亏损交易) / 亏损次数 if 亏损次数 > 0 else 0
            
            盈亏比 = abs(平均盈利 / 平均亏损) if 平均亏损 != 0 else float('inf')
        else:
            胜率 = 0
            盈亏比 = 0
            盈利次数 = 0
            亏损次数 = 0
        
        # 汇总结果
        结果 = {
            '初始资产': 初始资产,
            '最终资产': 最终资产,
            '收益率': 收益率,
            '年化收益率': 年化收益率,
            '最大回撤': 最大回撤,
            '收益回撤比': 收益回撤比,
            '夏普比率': 夏普比率,
            '交易次数': 交易次数,
            '盈利次数': 盈利次数,
            '亏损次数': 亏损次数,
            '胜率': 胜率,
            '盈亏比': 盈亏比,
            '资金曲线': 资金曲线,
            '交易记录': self.交易记录
        }
        
        return 结果


# 英文命名版本，功能与中文版完全相同
class BacktestEngine:
    """
    Backtest engine class, used to evaluate the historical performance of trading strategies.
    Supports one-click backtesting and automatic parameter optimization.
    """
    
    def __init__(self, 
               initial_capital: float = 100000.0,
               commission_rate: float = 0.0003,
               slippage: float = 0.0001):
        """
        Initialize the backtest engine.
        
        Parameters:
            initial_capital: Initial capital for backtesting
            commission_rate: Trading commission rate (applied to both buy and sell)
            slippage: Trading slippage rate
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.reset()
    
    def reset(self):
        """Reset backtest state."""
        self.capital = self.initial_capital
        self.positions = {}  # stock_code -> position size
        self.position_costs = {}  # stock_code -> average cost
        self.trade_records = []
        self.equity_curve = []
        self.current_date = None
    
    def run_backtest(self, 
                  market_data: pd.DataFrame, 
                  signal_data: pd.DataFrame,
                  symbol: str) -> Dict[str, Any]:
        """
        Execute the backtest process.
        
        Parameters:
            market_data: DataFrame containing OHLCV data
            signal_data: DataFrame containing trading signals
            symbol: Trading symbol code
            
        Returns:
            Dictionary containing backtest results
        """
        logger.info(f"Starting backtest for {symbol}")
        
        # Reset backtest state
        self.reset()
        
        # Ensure data indices are aligned
        common_index = market_data.index.intersection(signal_data.index)
        market_data = market_data.loc[common_index]
        signal_data = signal_data.loc[common_index]
        
        # Initialize results recording
        self.equity_curve = [{'date': market_data.index[0], 'equity': self.initial_capital, 'cash': self.initial_capital, 'position_value': 0.0}]
        
        # Iterate through each trading day
        for i in range(len(market_data)):
            current_date = market_data.index[i]
            self.current_date = current_date
            
            # Get the price and signal for the day
            open_price = market_data['open'].iloc[i]
            high_price = market_data['high'].iloc[i]
            low_price = market_data['low'].iloc[i]
            close_price = market_data['close'].iloc[i]
            signal = signal_data['信号'].iloc[i]  # We keep the Chinese column name for compatibility
            
            # Process trading signal
            if signal == 1:  # Buy signal
                self._execute_buy(symbol, close_price, current_date, signal_data.iloc[i])
            elif signal == -1:  # Sell signal
                self._execute_sell(symbol, close_price, current_date, signal_data.iloc[i])
            
            # Update equity curve
            total_equity = self._calculate_total_equity(symbol, close_price)
            position_value = total_equity - self.capital
            self.equity_curve.append({
                'date': current_date,
                'equity': total_equity,
                'cash': self.capital,
                'position_value': position_value
            })
        
        # Calculate backtest metrics
        results = self._calculate_backtest_metrics()
        
        logger.info(f"Backtest completed, final equity: {results['final_equity']:.2f}, return: {results['return']*100:.2f}%")
        return results
    
    def _execute_buy(self, symbol: str, price: float, date: datetime.datetime, signal_row: pd.Series):
        """Execute buy operation."""
        # Consider slippage
        actual_price = price * (1 + self.slippage)
        
        # Calculate buy quantity (use 90% of available capital to avoid insufficient funds)
        available_capital = self.capital * 0.9
        quantity = int(available_capital / actual_price)
        
        if quantity <= 0:
            logger.warning(f"{date}: Insufficient funds, cannot buy {symbol}")
            return
        
        # Calculate transaction cost
        trade_amount = quantity * actual_price
        commission = trade_amount * self.commission_rate
        total_cost = trade_amount + commission
        
        # Check if capital is sufficient
        if total_cost > self.capital:
            quantity = int((self.capital / (actual_price * (1 + self.commission_rate))))
            trade_amount = quantity * actual_price
            commission = trade_amount * self.commission_rate
            total_cost = trade_amount + commission
        
        if quantity <= 0:
            logger.warning(f"{date}: Insufficient funds, cannot buy {symbol}")
            return
        
        # Update position and capital
        if symbol in self.positions:
            # Calculate new average cost
            original_position = self.positions[symbol]
            original_cost = self.position_costs[symbol]
            new_position = original_position + quantity
            new_cost = (original_position * original_cost + trade_amount) / new_position
            self.positions[symbol] = new_position
            self.position_costs[symbol] = new_cost
        else:
            self.positions[symbol] = quantity
            self.position_costs[symbol] = actual_price
        
        self.capital -= total_cost
        
        # Record trade
        trade_record = {
            'date': date,
            'type': 'buy',
            'symbol': symbol,
            'price': actual_price,
            'quantity': quantity,
            'amount': trade_amount,
            'commission': commission,
            'remaining_capital': self.capital,
            'signal_explanation': signal_row.get('信号解释', '')
        }
        self.trade_records.append(trade_record)
        
        logger.debug(f"{date}: Buy {symbol} {quantity} shares, price {actual_price:.2f}, total cost {total_cost:.2f}")
    
    def _execute_sell(self, symbol: str, price: float, date: datetime.datetime, signal_row: pd.Series):
        """Execute sell operation."""
        if symbol not in self.positions or self.positions[symbol] <= 0:
            logger.warning(f"{date}: No position, cannot sell {symbol}")
            return
        
        # Consider slippage
        actual_price = price * (1 - self.slippage)
        
        # Sell all positions
        quantity = self.positions[symbol]
        trade_amount = quantity * actual_price
        commission = trade_amount * self.commission_rate
        net_income = trade_amount - commission
        
        # Calculate profit/loss
        cost = self.position_costs[symbol] * quantity
        pnl = net_income - cost
        
        # Update position and capital
        self.positions[symbol] = 0
        self.capital += net_income
        
        # Record trade
        trade_record = {
            'date': date,
            'type': 'sell',
            'symbol': symbol,
            'price': actual_price,
            'quantity': quantity,
            'amount': trade_amount,
            'commission': commission,
            'net_income': net_income,
            'pnl': pnl,
            'remaining_capital': self.capital,
            'signal_explanation': signal_row.get('信号解释', '')
        }
        self.trade_records.append(trade_record)
        
        logger.debug(f"{date}: Sell {symbol} {quantity} shares, price {actual_price:.2f}, net income {net_income:.2f}, PnL {pnl:.2f}")
    
    def _calculate_total_equity(self, symbol: str, current_price: float) -> float:
        """Calculate current total equity."""
        position_value = 0.0
        if symbol in self.positions:
            position_value = self.positions[symbol] * current_price
        
        return self.capital + position_value
    
    def _calculate_backtest_metrics(self) -> Dict[str, Any]:
        """Calculate backtest result metrics."""
        # Convert equity curve to DataFrame
        equity_curve = pd.DataFrame(self.equity_curve)
        equity_curve['date'] = pd.to_datetime(equity_curve['date'])
        equity_curve = equity_curve.set_index('date')
        
        # Calculate basic metrics
        initial_equity = equity_curve['equity'].iloc[0]
        final_equity = equity_curve['equity'].iloc[-1]
        total_return = (final_equity / initial_equity) - 1
        
        # Calculate annualized return
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        trading_days = (end_date - start_date).days
        if trading_days > 0:
            annual_return = (1 + total_return) ** (365 / trading_days) - 1
        else:
            annual_return = 0
        
        # Calculate maximum drawdown
        equity_curve['cumulative_max'] = equity_curve['equity'].cummax()
        equity_curve['drawdown'] = (equity_curve['equity'] - equity_curve['cumulative_max']) / equity_curve['cumulative_max']
        max_drawdown = abs(equity_curve['drawdown'].min())
        
        # Calculate Calmar ratio
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else float('inf')
        
        # Calculate Sharpe ratio (assume risk-free rate is 0)
        daily_returns = equity_curve['equity'].pct_change().dropna()
        if len(daily_returns) > 1:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate trade statistics
        trade_count = len(self.trade_records)
        if trade_count > 0:
            profitable_trades = [trade for trade in self.trade_records if trade.get('type') == 'sell' and trade.get('pnl', 0) > 0]
            losing_trades = [trade for trade in self.trade_records if trade.get('type') == 'sell' and trade.get('pnl', 0) <= 0]
            
            profitable_count = len(profitable_trades)
            losing_count = len(losing_trades)
            
            win_rate = profitable_count / (profitable_count + losing_count) if (profitable_count + losing_count) > 0 else 0
            
            avg_profit = sum(trade['pnl'] for trade in profitable_trades) / profitable_count if profitable_count > 0 else 0
            avg_loss = sum(trade['pnl'] for trade in losing_trades) / losing_count if losing_count > 0 else 0
            
            profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
        else:
            win_rate = 0
            profit_loss_ratio = 0
            profitable_count = 0
            losing_count = 0
        
        # Summarize results
        results = {
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'sharpe_ratio': sharpe_ratio,
            'trade_count': trade_count,
            'profitable_count': profitable_count,
            'losing_count': losing_count,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'equity_curve': equity_curve,
            'trade_records': self.trade_records,
            # Add Chinese-named fields for compatibility
            '初始资产': initial_equity,
            '最终资产': final_equity,
            '收益率': total_return,
            '年化收益率': annual_return,
            '最大回撤': max_drawdown,
            '收益回撤比': calmar_ratio,
            '夏普比率': sharpe_ratio,
            '交易次数': trade_count,
            '盈利次数': profitable_count,
            '亏损次数': losing_count,
            '胜率': win_rate,
            '盈亏比': profit_loss_ratio,
            '资金曲线': equity_curve,
            '交易记录': self.trade_records
        }
        
        return results 