"""
回测结果可视化模块

提供绘制回测性能图表和参数分布图表的功能。

日期：2025-05-17
"""

import os
import datetime
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns

# 配置日志
logger = logging.getLogger(__name__)

# 配置Matplotlib样式
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def create_performance_chart(
    backtest_results: Dict[str, Any],
    market_data: pd.DataFrame,
    signal_data: pd.DataFrame,
    symbol: str,
    save_path: Optional[str] = None,
    show_chart: bool = False
) -> str:
    """
    创建回测性能图表。
    
    参数:
        backtest_results: 回测结果字典
        market_data: 市场数据DataFrame
        signal_data: 信号数据DataFrame
        symbol: 交易标的代码
        save_path: 保存路径，如果为None则自动生成
        show_chart: 是否显示图表
        
    返回:
        图表文件路径
    """
    logger.info(f"正在为 {symbol} 创建回测性能图表...")
    
    if save_path is None:
        # 创建图表目录
        chart_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'charts')
        os.makedirs(chart_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join(chart_dir, f"{symbol}_performance_{timestamp}.png")
    
    # 获取回测数据
    equity_curve = backtest_results.get('equity_curve', None)
    if equity_curve is None or market_data is None or signal_data is None:
        logger.error("缺少创建图表所需的数据")
        return ""
    
    # 确保关键列存在
    if 'equity' not in equity_curve.columns and '资产' in equity_curve.columns:
        equity_curve['equity'] = equity_curve['资产']
    if 'cash' not in equity_curve.columns and '现金' in equity_curve.columns:
        equity_curve['cash'] = equity_curve['现金']
    if 'position_value' not in equity_curve.columns and '持仓' in equity_curve.columns:
        equity_curve['position_value'] = equity_curve['持仓']
    
    # 获取交易记录
    trade_records = backtest_results.get('trade_records', [])
    
    # 创建多子图布局
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(4, 1, height_ratios=[2, 1.5, 1, 1])
    
    # 1. 资金曲线
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(equity_curve.index, equity_curve['equity'], label='Total Equity', linewidth=2)
    ax1.plot(equity_curve.index, equity_curve['cash'], label='Cash', alpha=0.7)
    ax1.plot(equity_curve.index, equity_curve['position_value'], label='Position Value', alpha=0.7)
    
    # 添加买入卖出标记
    buy_dates = []
    sell_dates = []
    buy_values = []
    sell_values = []
    
    for trade in trade_records:
        trade_date = trade.get('date', trade.get('日期'))
        trade_type = str(trade.get('type', trade.get('类型', ''))).lower()
        
        if trade_type in ['buy', '买入']:
            buy_dates.append(trade_date)
            buy_values.append(trade.get('remaining_capital', trade.get('剩余资金', 0)) + 
                             trade.get('amount', trade.get('交易金额', 0)))
        elif trade_type in ['sell', '卖出']:
            sell_dates.append(trade_date)
            sell_values.append(trade.get('remaining_capital', trade.get('剩余资金', 0)))
    
    ax1.scatter(buy_dates, buy_values, marker='^', color='green', s=100, label='Buy')
    ax1.scatter(sell_dates, sell_values, marker='v', color='red', s=100, label='Sell')
    
    ax1.set_title(f"{symbol} Equity Curve", fontsize=16)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # 添加收益率和回撤等关键指标标注
    initial_equity = backtest_results.get('initial_equity', backtest_results.get('初始资产', equity_curve['equity'].iloc[0]))
    final_equity = backtest_results.get('final_equity', backtest_results.get('最终资产', equity_curve['equity'].iloc[-1]))
    total_return = backtest_results.get('return', backtest_results.get('收益率', (final_equity / initial_equity) - 1))
    max_drawdown = backtest_results.get('max_drawdown', backtest_results.get('最大回撤', 0))
    
    text_info = f"Return: {total_return*100:.2f}%\nMax Drawdown: {max_drawdown*100:.2f}%"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.05, text_info, transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom', bbox=props)
    
    # 2. 价格和信号
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(market_data.index, market_data['close'], label='Close Price')
    
    # 添加买入卖出信号
    buy_signals = signal_data[signal_data['信号'] == 1].index
    sell_signals = signal_data[signal_data['信号'] == -1].index
    
    buy_prices = [market_data.loc[date]['close'] if date in market_data.index else np.nan for date in buy_signals]
    sell_prices = [market_data.loc[date]['close'] if date in market_data.index else np.nan for date in sell_signals]
    
    ax2.scatter(buy_signals, buy_prices, marker='^', color='green', s=100, label='Buy Signal')
    ax2.scatter(sell_signals, sell_prices, marker='v', color='red', s=100, label='Sell Signal')
    
    ax2.set_title(f"{symbol} Price and Trading Signals", fontsize=16)
    ax2.set_ylabel('Price', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    # 3. 成交量
    if 'volume' in market_data.columns:
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.bar(market_data.index, market_data['volume'], color='gray', alpha=0.7)
        ax3.set_title(f"{symbol} Volume", fontsize=16)
        ax3.set_ylabel('Volume', fontsize=12)
        ax3.grid(True)
    
    # 4. 智能资金指标
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    
    if '操纵分数' in signal_data.columns or 'smart_money_score' in signal_data.columns:
        score_key = '操纵分数' if '操纵分数' in signal_data.columns else 'smart_money_score'
        ax4.plot(signal_data.index, signal_data[score_key], label='Smart Money Score', linewidth=2)
        
        if '成交量异常分数' in signal_data.columns or 'volume_anomaly_score' in signal_data.columns:
            vol_key = '成交量异常分数' if '成交量异常分数' in signal_data.columns else 'volume_anomaly_score'
            ax4.plot(signal_data.index, signal_data[vol_key], label='Volume Anomaly', alpha=0.7)
        
        if '价格模式分数' in signal_data.columns or 'price_pattern_score' in signal_data.columns:
            pattern_key = '价格模式分数' if '价格模式分数' in signal_data.columns else 'price_pattern_score'
            ax4.plot(signal_data.index, signal_data[pattern_key], label='Price Pattern', alpha=0.7)
        
        ax4.set_title("Smart Money Indicators", fontsize=16)
        ax4.set_ylabel('Score', fontsize=12)
        ax4.legend(loc='upper left')
        ax4.grid(True)
    else:
        # 如果没有智能资金指标，显示日收益率
        daily_returns = equity_curve['equity'].pct_change()
        ax4.bar(equity_curve.index, daily_returns, color='gray', alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title("Daily Returns", fontsize=16)
        ax4.set_ylabel('Return (%)', fontsize=12)
        ax4.grid(True)
    
    # 设置X轴日期格式
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels() if 'volume' in market_data.columns else ax2.get_xticklabels(), visible=False)
    
    formatter = mdates.DateFormatter('%Y-%m-%d')
    ax4.xaxis.set_major_formatter(formatter)
    
    if len(market_data) > 60:
        ax4.xaxis.set_major_locator(mdates.MonthLocator())
    else:
        ax4.xaxis.set_major_locator(mdates.WeekdayLocator())
    
    plt.xlabel('Date', fontsize=12)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show_chart:
        plt.show()
    else:
        plt.close()
    
    logger.info(f"回测性能图表已保存至: {save_path}")
    return save_path

def create_parameter_distribution_chart(
    optimization_results: List[Dict[str, Any]],
    objective: str,
    parameters: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show_chart: bool = False
) -> str:
    """
    创建参数分布图表。
    
    参数:
        optimization_results: 优化结果列表
        objective: 优化目标指标
        parameters: 要显示的参数列表，如果为None则显示所有参数
        save_path: 保存路径，如果为None则自动生成
        show_chart: 是否显示图表
        
    返回:
        图表文件路径
    """
    if not optimization_results:
        logger.error("优化结果为空，无法创建参数分布图")
        return ""
    
    logger.info(f"正在创建参数分布图表，优化目标: {objective}...")
    
    if save_path is None:
        # 创建图表目录
        chart_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'charts')
        os.makedirs(chart_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join(chart_dir, f"parameter_distribution_{timestamp}.png")
    
    # 提取参数和目标指标值
    first_result = optimization_results[0]
    
    params_key = 'parameters' if 'parameters' in first_result else '参数'
    results_key = 'backtest_results' if 'backtest_results' in first_result else '回测结果'
    
    # 确定参数列表
    all_params = list(first_result[params_key].keys())
    if parameters is None:
        parameters = all_params
    else:
        # 确保所有指定的参数都存在
        parameters = [p for p in parameters if p in all_params]
    
    # 如果没有有效参数，返回
    if not parameters:
        logger.error("没有有效参数，无法创建参数分布图")
        return ""
    
    # 确保目标指标存在
    obj_key = objective
    for key in first_result[results_key].keys():
        if key.lower() == objective.lower():
            obj_key = key
            break
    
    # 提取参数和目标值
    data = []
    for result in optimization_results:
        row = {param: result[params_key][param] for param in parameters}
        row[objective] = result[results_key].get(obj_key, 0)
        data.append(row)
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 创建图表
    param_count = len(parameters)
    if param_count <= 2:
        fig_size = (10, 6)
    else:
        cols = min(3, param_count)
        rows = (param_count + cols - 1) // cols
        fig_size = (5 * cols, 4 * rows)
    
    plt.figure(figsize=fig_size)
    
    for i, param in enumerate(parameters):
        plt.subplot(rows if param_count > 2 else 1, cols if param_count > 2 else param_count, i + 1)
        
        # 散点图
        plt.scatter(df[param], df[objective], alpha=0.7)
        
        # 添加趋势线
        if len(df) > 1:
            try:
                z = np.polyfit(df[param], df[objective], 1)
                p = np.poly1d(z)
                plt.plot(df[param], p(df[param]), "r--", alpha=0.7)
            except:
                pass  # 如果拟合失败，不添加趋势线
        
        plt.title(f"{param} vs {objective}")
        plt.xlabel(param)
        plt.ylabel(objective)
        plt.grid(True)
    
    # 如果参数多于1个，添加相关性热图
    if param_count > 1:
        if param_count <= 3:
            plt.subplot(rows, cols, param_count + 1)
        else:
            plt.figure(figsize=(10, 8))
        
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Parameter Correlation")
        plt.tight_layout()
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show_chart:
        plt.show()
    else:
        plt.close()
    
    logger.info(f"参数分布图表已保存至: {save_path}")
    return save_path 