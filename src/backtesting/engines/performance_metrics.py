"""
回测性能指标计算模块

提供各种回测性能指标的计算函数，用于评估交易策略的表现。

日期：2025-05-17
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union

def calculate_metrics(
    equity_curve: pd.DataFrame,
    trade_records: List[Dict[str, Any]],
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252
) -> Dict[str, Any]:
    """
    计算回测性能指标。
    
    参数:
        equity_curve: 资金曲线DataFrame，必须包含'equity'列
        trade_records: 交易记录列表
        risk_free_rate: 无风险利率，年化，默认为0
        trading_days_per_year: 每年交易日数，默认为252
        
    返回:
        包含各种性能指标的字典
    """
    if len(equity_curve) < 2:
        return _empty_metrics()
    
    # 确保资金曲线正确索引
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        if '日期' in equity_curve.columns:
            equity_curve['日期'] = pd.to_datetime(equity_curve['日期'])
            equity_curve = equity_curve.set_index('日期')
        elif 'date' in equity_curve.columns:
            equity_curve['date'] = pd.to_datetime(equity_curve['date'])
            equity_curve = equity_curve.set_index('date')
    
    # 标准化列名
    if 'equity' not in equity_curve.columns and '资产' in equity_curve.columns:
        equity_curve['equity'] = equity_curve['资产']
    
    # 基本指标计算
    metrics = {}
    
    # 收益指标
    initial_equity = equity_curve['equity'].iloc[0]
    final_equity = equity_curve['equity'].iloc[-1]
    total_return = (final_equity / initial_equity) - 1
    
    metrics['initial_equity'] = initial_equity
    metrics['final_equity'] = final_equity
    metrics['total_return'] = total_return
    
    # 年化指标
    start_date = equity_curve.index[0]
    end_date = equity_curve.index[-1]
    days = (end_date - start_date).days
    years = days / 365.0 if days > 0 else 0
    
    if years > 0:
        annual_return = (1 + total_return) ** (1 / years) - 1
    else:
        annual_return = 0
    
    metrics['annual_return'] = annual_return
    
    # 波动率
    daily_returns = equity_curve['equity'].pct_change().dropna()
    if len(daily_returns) > 1:
        daily_volatility = daily_returns.std()
        annual_volatility = daily_volatility * np.sqrt(trading_days_per_year)
    else:
        daily_volatility = 0
        annual_volatility = 0
    
    metrics['daily_volatility'] = daily_volatility
    metrics['annual_volatility'] = annual_volatility
    
    # 回撤
    equity_curve['cummax'] = equity_curve['equity'].cummax()
    equity_curve['drawdown'] = (equity_curve['equity'] - equity_curve['cummax']) / equity_curve['cummax']
    max_drawdown = abs(equity_curve['drawdown'].min())
    
    metrics['max_drawdown'] = max_drawdown
    
    # 风险调整收益
    if annual_volatility > 0:
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    else:
        sharpe_ratio = 0
    
    if max_drawdown > 0:
        calmar_ratio = annual_return / max_drawdown
    else:
        calmar_ratio = float('inf')
    
    metrics['sharpe_ratio'] = sharpe_ratio
    metrics['calmar_ratio'] = calmar_ratio
    
    # 交易统计
    metrics['trade_count'] = len(trade_records)
    
    if trade_records:
        # 筛选出卖出交易
        sell_trades = [t for t in trade_records if t.get('type', t.get('类型', '')).lower() in ['sell', '卖出']]
        
        # 盈利/亏损交易
        pnl_key = next((k for k in ['pnl', '盈亏'] if k in trade_records[0]), None)
        
        if pnl_key and sell_trades:
            profit_trades = [t for t in sell_trades if t.get(pnl_key, 0) > 0]
            loss_trades = [t for t in sell_trades if t.get(pnl_key, 0) <= 0]
            
            profit_count = len(profit_trades)
            loss_count = len(loss_trades)
            
            metrics['profit_count'] = profit_count
            metrics['loss_count'] = loss_count
            
            # 胜率
            if profit_count + loss_count > 0:
                win_rate = profit_count / (profit_count + loss_count)
            else:
                win_rate = 0
            
            metrics['win_rate'] = win_rate
            
            # 平均盈亏
            if profit_count > 0:
                avg_profit = sum(t.get(pnl_key, 0) for t in profit_trades) / profit_count
            else:
                avg_profit = 0
                
            if loss_count > 0:
                avg_loss = sum(t.get(pnl_key, 0) for t in loss_trades) / loss_count
            else:
                avg_loss = 0
            
            metrics['avg_profit'] = avg_profit
            metrics['avg_loss'] = avg_loss
            
            # 盈亏比
            if avg_loss != 0:
                profit_loss_ratio = abs(avg_profit / avg_loss)
            else:
                profit_loss_ratio = float('inf')
            
            metrics['profit_loss_ratio'] = profit_loss_ratio
            
            # 期望收益
            expectancy = (win_rate * avg_profit) + ((1 - win_rate) * avg_loss)
            metrics['expectancy'] = expectancy
    else:
        metrics['profit_count'] = 0
        metrics['loss_count'] = 0
        metrics['win_rate'] = 0
        metrics['avg_profit'] = 0
        metrics['avg_loss'] = 0
        metrics['profit_loss_ratio'] = 0
        metrics['expectancy'] = 0
    
    # 添加中文键
    chinese_keys = {
        'initial_equity': '初始资产',
        'final_equity': '最终资产',
        'total_return': '收益率',
        'annual_return': '年化收益率',
        'daily_volatility': '日波动率',
        'annual_volatility': '年化波动率',
        'max_drawdown': '最大回撤',
        'sharpe_ratio': '夏普比率',
        'calmar_ratio': '收益回撤比',
        'trade_count': '交易次数',
        'profit_count': '盈利次数',
        'loss_count': '亏损次数',
        'win_rate': '胜率',
        'avg_profit': '平均盈利',
        'avg_loss': '平均亏损',
        'profit_loss_ratio': '盈亏比',
        'expectancy': '期望收益'
    }
    
    for en_key, zh_key in chinese_keys.items():
        if en_key in metrics:
            metrics[zh_key] = metrics[en_key]
    
    return metrics

def _empty_metrics() -> Dict[str, Any]:
    """返回空的指标集合"""
    return {
        'initial_equity': 0,
        'final_equity': 0,
        'total_return': 0,
        'annual_return': 0,
        'daily_volatility': 0,
        'annual_volatility': 0,
        'max_drawdown': 0,
        'sharpe_ratio': 0,
        'calmar_ratio': 0,
        'trade_count': 0,
        'profit_count': 0,
        'loss_count': 0,
        'win_rate': 0,
        'avg_profit': 0,
        'avg_loss': 0,
        'profit_loss_ratio': 0,
        'expectancy': 0,
        # 中文指标
        '初始资产': 0,
        '最终资产': 0,
        '收益率': 0,
        '年化收益率': 0,
        '日波动率': 0,
        '年化波动率': 0,
        '最大回撤': 0,
        '夏普比率': 0,
        '收益回撤比': 0,
        '交易次数': 0,
        '盈利次数': 0,
        '亏损次数': 0,
        '胜率': 0,
        '平均盈利': 0,
        '平均亏损': 0,
        '盈亏比': 0,
        '期望收益': 0
    } 