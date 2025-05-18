"""
回测引擎模块

本模块实现了高性能的回测引擎，用于测试和优化交易策略。
主要功能包括：
1. 历史数据回放
2. 交易信号生成与执行
3. 性能指标计算
4. 可视化结果展示
5. 参数优化

日期：2025-05-16
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from itertools import product
import json
import time
from tqdm import tqdm
import pickle
from scipy import stats
from sklearn.model_selection import ParameterGrid
from bayes_opt import BayesianOptimization
import warnings

# 配置日志
logger = logging.getLogger(__name__)

class BacktestEngine:
    """高性能回测引擎，用于测试和优化交易策略"""
    
    def __init__(self, config: Dict = None):
        """
        初始化回测引擎
        
        参数:
            config: 配置信息，包含回测参数和设置
        """
        self.config = config or {}
        
        # 设置默认参数
        self.initial_capital = self.config.get('initial_capital', 1000000)  # 初始资金
        self.commission_rate = self.config.get('commission_rate', 0.0003)  # 佣金率
        self.slippage = self.config.get('slippage', 0.0001)  # 滑点
        self.tax_rate = self.config.get('tax_rate', 0.001)  # 印花税率（仅卖出时收取）
        self.min_commission = self.config.get('min_commission', 5)  # 最低佣金
        self.position_sizing = self.config.get('position_sizing', 'equal')  # 仓位管理策略
        self.max_positions = self.config.get('max_positions', 5)  # 最大持仓数量
        self.risk_free_rate = self.config.get('risk_free_rate', 0.03)  # 无风险利率（年化）
        
        # 回测结果
        self.results = {}
        self.trades = []
        self.positions = {}
        self.equity_curve = None
        
        # 回测状态
        self.current_date = None
        self.current_capital = self.initial_capital
        self.current_positions = {}
        self.current_cash = self.initial_capital
        
        # 性能指标
        self.metrics = {}
        
        logger.info("回测引擎初始化成功")
    
    def run_backtest(self, data: Dict[str, pd.DataFrame], strategy_func: Callable, strategy_params: Dict = None) -> Dict:
        """
        运行回测
        
        参数:
            data: 回测数据，格式为 {symbol: DataFrame}
            strategy_func: 策略函数，接收数据和参数，返回交易信号
            strategy_params: 策略参数
            
        返回:
            回测结果字典
        """
        if not data:
            logger.error("回测数据为空")
            return {}
        
        # 重置回测状态
        self._reset_backtest()
        
        # 设置策略参数
        params = strategy_params or {}
        
        # 获取所有交易日
        all_dates = self._get_all_trading_dates(data)
        if not all_dates:
            logger.error("无法获取交易日期")
            return {}
        
        # 初始化回测结果
        self._init_results(all_dates)
        
        # 运行回测
        logger.info(f"开始回测，初始资金: {self.initial_capital}，交易品种数: {len(data)}")
        
        try:
            # 按日期遍历
            for date_idx, date in enumerate(all_dates):
                self.current_date = date
                
                # 更新当前数据
                current_data = self._get_data_at_date(data, date)
                
                # 生成交易信号
                signals = strategy_func(current_data, params)
                
                # 执行交易
                self._execute_trades(signals, current_data)
                
                # 更新持仓市值
                self._update_positions_value(current_data)
                
                # 记录每日回测结果
                self._record_daily_results()
                
                # 每100个交易日输出一次进度
                if date_idx % 100 == 0:
                    logger.info(f"回测进度: {date_idx}/{len(all_dates)} ({date})")
            
            # 计算性能指标
            self._calculate_performance_metrics()
            
            logger.info("回测完成")
            return self.results
        
        except Exception as e:
            logger.error(f"回测过程中发生错误: {e}")
            return {}
    
    def _reset_backtest(self):
        """重置回测状态"""
        self.current_date = None
        self.current_capital = self.initial_capital
        self.current_positions = {}
        self.current_cash = self.initial_capital
        self.results = {}
        self.trades = []
        self.positions = {}
        self.equity_curve = None
        self.metrics = {}
    
    def _get_all_trading_dates(self, data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """获取所有交易日期"""
        all_dates = set()
        
        for symbol, df in data.items():
            # 确保索引是日期类型
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    dates = df['date'].tolist()
                else:
                    logger.warning(f"无法获取交易日期 - 品种: {symbol}")
                    continue
            else:
                dates = df.index.tolist()
            
            all_dates.update(dates)
        
        # 排序日期
        return sorted(list(all_dates))
    
    def _init_results(self, dates: List[datetime]):
        """初始化回测结果"""
        # 初始化权益曲线
        self.equity_curve = pd.DataFrame(index=dates, columns=[
            'cash', 'position_value', 'total_value', 'daily_returns', 'cumulative_returns'
        ])
        
        # 初始化第一天的数据
        self.equity_curve.loc[dates[0], 'cash'] = self.initial_capital
        self.equity_curve.loc[dates[0], 'position_value'] = 0
        self.equity_curve.loc[dates[0], 'total_value'] = self.initial_capital
        self.equity_curve.loc[dates[0], 'daily_returns'] = 0
        self.equity_curve.loc[dates[0], 'cumulative_returns'] = 0
    
    def _get_data_at_date(self, data: Dict[str, pd.DataFrame], date: datetime) -> Dict[str, Dict]:
        """获取指定日期的数据"""
        result = {}
        
        for symbol, df in data.items():
            # 获取当前日期的数据
            if isinstance(df.index, pd.DatetimeIndex):
                if date in df.index:
                    row = df.loc[date]
                    result[symbol] = row.to_dict() if isinstance(row, pd.Series) else row.iloc[0].to_dict()
            else:
                if 'date' in df.columns:
                    row = df[df['date'] == date]
                    if not row.empty:
                        result[symbol] = row.iloc[0].to_dict()
        
        return result
    
    def _execute_trades(self, signals: Dict[str, str], current_data: Dict[str, Dict]):
        """
        执行交易
        
        参数:
            signals: 交易信号，格式为 {symbol: signal}，signal可以是'buy', 'sell', 'hold'
            current_data: 当前数据，格式为 {symbol: {field: value}}
        """
        # 处理卖出信号
        for symbol, signal in signals.items():
            if signal.lower() == 'sell' and symbol in self.current_positions:
                self._sell_position(symbol, current_data.get(symbol, {}))
        
        # 处理买入信号
        buy_signals = {symbol: signal for symbol, signal in signals.items() if signal.lower() == 'buy'}
        
        if buy_signals:
            # 计算每个品种的买入金额
            available_cash = self.current_cash
            current_positions_count = len(self.current_positions)
            max_new_positions = max(0, self.max_positions - current_positions_count)
            
            if max_new_positions > 0:
                # 根据仓位管理策略分配资金
                if self.position_sizing == 'equal':
                    # 平均分配资金
                    cash_per_position = available_cash / min(len(buy_signals), max_new_positions)
                    
                    # 执行买入
                    for symbol, signal in list(buy_signals.items())[:max_new_positions]:
                        if symbol not in self.current_positions and symbol in current_data:
                            self._buy_position(symbol, cash_per_position, current_data[symbol])
                
                elif self.position_sizing == 'kelly':
                    # Kelly准则分配资金（需要历史数据计算胜率和赔率）
                    # 这里简化处理，实际应用中需要更复杂的计算
                    for symbol, signal in list(buy_signals.items())[:max_new_positions]:
                        if symbol not in self.current_positions and symbol in current_data:
                            # 假设每个品种使用10%的可用资金
                            cash_amount = available_cash * 0.1
                            self._buy_position(symbol, cash_amount, current_data[symbol])
                
                else:
                    # 默认平均分配
                    cash_per_position = available_cash / min(len(buy_signals), max_new_positions)
                    
                    # 执行买入
                    for symbol, signal in list(buy_signals.items())[:max_new_positions]:
                        if symbol not in self.current_positions and symbol in current_data:
                            self._buy_position(symbol, cash_per_position, current_data[symbol])
    
    def _buy_position(self, symbol: str, cash_amount: float, data: Dict):
        """
        买入品种
        
        参数:
            symbol: 品种代码
            cash_amount: 买入金额
            data: 当前品种数据
        """
        if 'close' not in data:
            logger.warning(f"买入失败，缺少价格数据 - 品种: {symbol}")
            return
        
        # 获取买入价格（考虑滑点）
        price = data['close'] * (1 + self.slippage)
        
        # 计算买入数量（向下取整）
        quantity = int(cash_amount / price)
        
        if quantity <= 0:
            logger.warning(f"买入失败，资金不足 - 品种: {symbol}, 资金: {cash_amount}, 价格: {price}")
            return
        
        # 计算实际买入金额
        actual_cost = quantity * price
        
        # 计算佣金
        commission = max(self.min_commission, actual_cost * self.commission_rate)
        
        # 更新资金和持仓
        self.current_cash -= (actual_cost + commission)
        
        # 添加到持仓
        self.current_positions[symbol] = {
            'quantity': quantity,
            'cost_price': price,
            'cost_value': actual_cost,
            'current_price': price,
            'current_value': actual_cost,
            'entry_date': self.current_date
        }
        
        # 记录交易
        trade = {
            'date': self.current_date,
            'symbol': symbol,
            'action': 'buy',
            'quantity': quantity,
            'price': price,
            'value': actual_cost,
            'commission': commission,
            'slippage': actual_cost * self.slippage
        }
        
        self.trades.append(trade)
        logger.debug(f"买入 - 品种: {symbol}, 数量: {quantity}, 价格: {price}, 金额: {actual_cost}, 佣金: {commission}")
    
    def _sell_position(self, symbol: str, data: Dict):
        """
        卖出品种
        
        参数:
            symbol: 品种代码
            data: 当前品种数据
        """
        if symbol not in self.current_positions:
            logger.warning(f"卖出失败，无持仓 - 品种: {symbol}")
            return
        
        if 'close' not in data:
            logger.warning(f"卖出失败，缺少价格数据 - 品种: {symbol}")
            return
        
        # 获取持仓信息
        position = self.current_positions[symbol]
        quantity = position['quantity']
        
        # 获取卖出价格（考虑滑点）
        price = data['close'] * (1 - self.slippage)
        
        # 计算卖出金额
        sell_value = quantity * price
        
        # 计算佣金和印花税
        commission = max(self.min_commission, sell_value * self.commission_rate)
        tax = sell_value * self.tax_rate
        
        # 更新资金
        self.current_cash += (sell_value - commission - tax)
        
        # 计算收益
        profit = sell_value - position['cost_value'] - commission - tax
        profit_pct = profit / position['cost_value'] if position['cost_value'] > 0 else 0
        
        # 记录交易
        trade = {
            'date': self.current_date,
            'symbol': symbol,
            'action': 'sell',
            'quantity': quantity,
            'price': price,
            'value': sell_value,
            'commission': commission,
            'tax': tax,
            'profit': profit,
            'profit_pct': profit_pct,
            'hold_days': (self.current_date - position['entry_date']).days
        }
        
        self.trades.append(trade)
        
        # 从持仓中移除
        del self.current_positions[symbol]
        
        logger.debug(f"卖出 - 品种: {symbol}, 数量: {quantity}, 价格: {price}, 金额: {sell_value}, 收益: {profit}, 收益率: {profit_pct:.2%}")
    
    def _update_positions_value(self, current_data: Dict[str, Dict]):
        """更新持仓市值"""
        for symbol, position in list(self.current_positions.items()):
            if symbol in current_data and 'close' in current_data[symbol]:
                # 更新当前价格和市值
                current_price = current_data[symbol]['close']
                current_value = position['quantity'] * current_price
                
                self.current_positions[symbol]['current_price'] = current_price
                self.current_positions[symbol]['current_value'] = current_value
            else:
                logger.warning(f"无法更新持仓市值，缺少价格数据 - 品种: {symbol}")
    
    def _record_daily_results(self):
        """记录每日回测结果"""
        if self.current_date is None:
            return
        
        # 计算持仓总市值
        position_value = sum(position['current_value'] for position in self.current_positions.values())
        
        # 计算总资产
        total_value = self.current_cash + position_value
        
        # 记录到权益曲线
        self.equity_curve.loc[self.current_date, 'cash'] = self.current_cash
        self.equity_curve.loc[self.current_date, 'position_value'] = position_value
        self.equity_curve.loc[self.current_date, 'total_value'] = total_value
        
        # 计算日收益率
        if self.current_date != self.equity_curve.index[0]:
            prev_total = self.equity_curve.loc[self.equity_curve.index < self.current_date, 'total_value'].iloc[-1]
            daily_return = (total_value / prev_total) - 1 if prev_total > 0 else 0
            self.equity_curve.loc[self.current_date, 'daily_returns'] = daily_return
        else:
            self.equity_curve.loc[self.current_date, 'daily_returns'] = 0
        
        # 计算累计收益率
        self.equity_curve.loc[self.current_date, 'cumulative_returns'] = (total_value / self.initial_capital) - 1
    
    def _calculate_performance_metrics(self):
        """计算性能指标"""
        if self.equity_curve is None or self.equity_curve.empty:
            logger.warning("无法计算性能指标，权益曲线为空")
            return
        
        # 填充缺失值
        self.equity_curve = self.equity_curve.fillna(method='ffill')
        
        # 计算日收益率序列
        returns = self.equity_curve['daily_returns'].dropna()
        
        if len(returns) < 2:
            logger.warning("无法计算性能指标，收益率序列过短")
            return
        
        # 计算年化收益率
        total_days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        total_years = total_days / 365
        
        final_return = self.equity_curve['cumulative_returns'].iloc[-1]
        annual_return = (1 + final_return) ** (1 / total_years) - 1 if total_years > 0 else 0
        
        # 计算波动率（年化）
        daily_std = returns.std()
        annual_volatility = daily_std * np.sqrt(252)  # 假设一年252个交易日
        
        # 计算夏普比率
        daily_risk_free = (1 + self.risk_free_rate) ** (1 / 252) - 1
        excess_returns = returns - daily_risk_free
        sharpe_ratio = (excess_returns.mean() / daily_std) * np.sqrt(252) if daily_std > 0 else 0
        
        # 计算最大回撤
        cumulative_returns = self.equity_curve['cumulative_returns']
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / (1 + running_max)
        max_drawdown = drawdown.min()
        
        # 计算卡玛比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')
        
        # 计算索提诺比率
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else float('inf')
        
        # 计算胜率
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty and 'profit' in trades_df.columns:
            winning_trades = trades_df[trades_df['profit'] > 0]
            win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
            
            # 计算盈亏比
            avg_win = winning_trades['profit'].mean() if not winning_trades.empty else 0
            losing_trades = trades_df[trades_df['profit'] <= 0]
            avg_loss = abs(losing_trades['profit'].mean()) if not losing_trades.empty else 1
            profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
            
            # 计算期望收益
            expected_return = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        else:
            win_rate = 0
            profit_loss_ratio = 0
            expected_return = 0
        
        # 存储性能指标
        self.metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': self.equity_curve['total_value'].iloc[-1],
            'total_return': final_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'expected_return': expected_return,
            'total_trades': len(self.trades),
            'trading_days': len(self.equity_curve),
            'start_date': self.equity_curve.index[0],
            'end_date': self.equity_curve.index[-1]
        }
        
        # 更新结果
        self.results = {
            'metrics': self.metrics,
            'equity_curve': self.equity_curve,
            'trades': self.trades
        }
        
        logger.info("性能指标计算完成")
    
    def plot_results(self, save_path: str = None, show_trades: bool = True):
        """
        绘制回测结果图表
        
        参数:
            save_path: 图表保存路径，如果为None则显示图表
            show_trades: 是否显示交易点
        """
        if self.equity_curve is None or self.equity_curve.empty:
            logger.warning("无法绘制图表，权益曲线为空")
            return
        
        if not self.metrics:
            logger.warning("无法绘制图表，性能指标未计算")
            return
        
        try:
            # 设置样式
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # 创建图表
            fig = plt.figure(figsize=(16, 12))
            
            # 设置网格
            gs = plt.GridSpec(4, 2, figure=fig)
            
            # 1. 权益曲线
            ax1 = fig.add_subplot(gs[0:2, 0:2])
            ax1.set_title('权益曲线', fontsize=14)
            ax1.plot(self.equity_curve.index, self.equity_curve['total_value'], 'b-', linewidth=2)
            
            # 添加初始资金线
            ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5)
            
            # 格式化y轴为货币格式
            ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))
            
            # 添加网格
            ax1.grid(True, alpha=0.3)
            
            # 如果需要显示交易点
            if show_trades and self.trades:
                trades_df = pd.DataFrame(self.trades)
                
                # 买入点
                buy_trades = trades_df[trades_df['action'] == 'buy']
                if not buy_trades.empty:
                    for _, trade in buy_trades.iterrows():
                        ax1.scatter(trade['date'], trade['value'], color='green', marker='^', s=100, alpha=0.7)
                
                # 卖出点
                sell_trades = trades_df[trades_df['action'] == 'sell']
                if not sell_trades.empty:
                    for _, trade in sell_trades.iterrows():
                        ax1.scatter(trade['date'], trade['value'], color='red', marker='v', s=100, alpha=0.7)
            
            # 2. 累计收益率
            ax2 = fig.add_subplot(gs[2, 0])
            ax2.set_title('累计收益率', fontsize=14)
            ax2.plot(self.equity_curve.index, self.equity_curve['cumulative_returns'] * 100, 'g-', linewidth=2)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}%'))
            ax2.grid(True, alpha=0.3)
            
            # 3. 回撤
            ax3 = fig.add_subplot(gs[2, 1])
            ax3.set_title('回撤', fontsize=14)
            
            # 计算回撤
            cumulative_returns = self.equity_curve['cumulative_returns']
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - running_max) / (1 + running_max) * 100
            
            ax3.fill_between(self.equity_curve.index, 0, drawdown, color='red', alpha=0.3)
            ax3.plot(self.equity_curve.index, drawdown, 'r-', linewidth=1)
            ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax3.set_ylim(bottom=min(drawdown) * 1.1, top=5)
            ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}%'))
            ax3.grid(True, alpha=0.3)
            
            # 4. 性能指标
            ax4 = fig.add_subplot(gs[3, 0])
            ax4.set_title('性能指标', fontsize=14)
            ax4.axis('off')
            
            metrics_text = (
                f"初始资金: {self.metrics['initial_capital']:,.0f}\n"
                f"最终资金: {self.metrics['final_capital']:,.0f}\n"
                f"总收益率: {self.metrics['total_return']:.2%}\n"
                f"年化收益率: {self.metrics['annual_return']:.2%}\n"
                f"年化波动率: {self.metrics['annual_volatility']:.2%}\n"
                f"夏普比率: {self.metrics['sharpe_ratio']:.2f}\n"
                f"最大回撤: {self.metrics['max_drawdown']:.2%}\n"
                f"卡玛比率: {self.metrics['calmar_ratio']:.2f}\n"
                f"索提诺比率: {self.metrics['sortino_ratio']:.2f}\n"
            )
            
            ax4.text(0.1, 0.9, metrics_text, fontsize=12, verticalalignment='top', transform=ax4.transAxes)
            
            # 5. 交易统计
            ax5 = fig.add_subplot(gs[3, 1])
            ax5.set_title('交易统计', fontsize=14)
            ax5.axis('off')
            
            trade_text = (
                f"总交易次数: {self.metrics['total_trades']}\n"
                f"胜率: {self.metrics['win_rate']:.2%}\n"
                f"盈亏比: {self.metrics['profit_loss_ratio']:.2f}\n"
                f"期望收益: {self.metrics['expected_return']:,.2f}\n"
                f"交易天数: {self.metrics['trading_days']}\n"
                f"开始日期: {self.metrics['start_date'].strftime('%Y-%m-%d')}\n"
                f"结束日期: {self.metrics['end_date'].strftime('%Y-%m-%d')}\n"
            )
            
            ax5.text(0.1, 0.9, trade_text, fontsize=12, verticalalignment='top', transform=ax5.transAxes)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存或显示图表
            if save_path:
                plt.savefig(save_path)
                logger.info(f"回测结果图表已保存到: {save_path}")
            else:
                plt.show()
                logger.info("回测结果图表已显示")
            
            plt.close(fig)
        
        except Exception as e:
            logger.error(f"绘制回测结果图表失败: {e}")
    
    def save_results(self, file_path: str):
        """
        保存回测结果
        
        参数:
            file_path: 保存路径
        """
        if not self.results:
            logger.warning("无法保存回测结果，结果为空")
            return
        
        try:
            # 转换结果为可序列化格式
            serializable_results = {
                'metrics': self.metrics,
                'equity_curve': self.equity_curve.reset_index().to_dict('records'),
                'trades': self.trades
            }
            
            # 保存为JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=4, default=str)
            
            logger.info(f"回测结果已保存到: {file_path}")
        
        except Exception as e:
            logger.error(f"保存回测结果失败: {e}")
    
    def load_results(self, file_path: str):
        """
        加载回测结果
        
        参数:
            file_path: 加载路径
        """
        try:
            # 从JSON加载
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_results = json.load(f)
            
            # 转换回原始格式
            self.metrics = loaded_results['metrics']
            
            # 转换权益曲线
            equity_data = loaded_results['equity_curve']
            self.equity_curve = pd.DataFrame(equity_data)
            if 'index' in self.equity_curve.columns:
                self.equity_curve['index'] = pd.to_datetime(self.equity_curve['index'])
                self.equity_curve.set_index('index', inplace=True)
            
            self.trades = loaded_results['trades']
            
            # 更新结果
            self.results = {
                'metrics': self.metrics,
                'equity_curve': self.equity_curve,
                'trades': self.trades
            }
            
            logger.info(f"回测结果已从 {file_path} 加载")
        
        except Exception as e:
            logger.error(f"加载回测结果失败: {e}")


    def get_backtest_history(self) -> List[Dict]:
        """
        获取回测历史记录

        返回:
            包含历史回测结果的字典列表
        """
        # 查找保存的回测结果文件
        backtest_results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backtest_results')

        if not os.path.exists(backtest_results_dir):
            return []

        results = []
        for filename in os.listdir(backtest_results_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(backtest_results_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        # 添加文件名作为ID
                        result['id'] = os.path.splitext(filename)[0]
                        results.append(result)
                except Exception as e:
                    logger.error(f"加载回测结果失败: {file_path}, 错误: {e}")

        # 按时间倒序排序
        if results and 'metrics' in results[0] and 'end_date' in results[0]['metrics']:
            results.sort(key=lambda x: x.get('metrics', {}).get('end_date', ''), reverse=True)

        return results
    
class ParameterOptimizer:
    """参数优化器，用于优化交易策略参数"""
    
    def __init__(self, backtest_engine: BacktestEngine, data: Dict[str, pd.DataFrame], strategy_func: Callable, base_params: Dict = None):
        """
        初始化参数优化器
        
        参数:
            backtest_engine: 回测引擎实例
            data: 回测数据，格式为 {symbol: DataFrame}
            strategy_func: 策略函数，接收数据和参数，返回交易信号
            base_params: 基础策略参数
        """
        self.backtest_engine = backtest_engine
        self.data = data
        self.strategy_func = strategy_func
        self.base_params = base_params or {}
        
        # 优化结果
        self.optimization_results = []
        
        # 设置多进程
        self.n_jobs = min(multiprocessing.cpu_count(), 8)  # 最多使用8个核心
        
        logger.info(f"参数优化器初始化成功，使用 {self.n_jobs} 个CPU核心")
    
    def grid_search(self, param_grid: Dict[str, List], metric: str = 'sharpe_ratio', top_n: int = 10) -> List[Dict]:
        """
        网格搜索参数优化
        
        参数:
            param_grid: 参数网格，格式为 {param_name: [param_values]}
            metric: 优化指标，如'sharpe_ratio', 'annual_return', 'calmar_ratio'等
            top_n: 返回前N个结果
            
        返回:
            优化结果列表
        """
        # 生成参数组合
        param_combinations = list(ParameterGrid(param_grid))
        total_combinations = len(param_combinations)
        
        if total_combinations == 0:
            logger.warning("参数网格为空，无法进行优化")
            return []
        
        logger.info(f"开始网格搜索参数优化，参数组合数: {total_combinations}")
        
        # 初始化结果列表
        self.optimization_results = []
        
        try:
            # 使用多进程并行计算
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                # 提交任务
                futures = [
                    executor.submit(self._evaluate_params, {**self.base_params, **params})
                    for params in param_combinations
                ]
                
                # 收集结果
                for i, future in enumerate(tqdm(futures, total=total_combinations, desc="参数优化进度")):
                    try:
                        result = future.result()
                        self.optimization_results.append(result)
                    except Exception as e:
                        logger.error(f"参数组合 {i+1}/{total_combinations} 评估失败: {e}")
            
            # 按指定指标排序
            self.optimization_results.sort(key=lambda x: x['metrics'].get(metric, float('-inf')), reverse=True)
            
            # 返回前N个结果
            top_results = self.optimization_results[:top_n]
            
            logger.info(f"网格搜索参数优化完成，最佳 {metric}: {top_results[0]['metrics'].get(metric) if top_results else 'N/A'}")
            return top_results
        
        except Exception as e:
            logger.error(f"网格搜索参数优化失败: {e}")
            return []
    
    def bayesian_optimization(self, param_bounds: Dict[str, Tuple[float, float]], n_iter: int = 50, metric: str = 'sharpe_ratio', top_n: int = 10) -> List[Dict]:
        """
        贝叶斯优化参数
        
        参数:
            param_bounds: 参数边界，格式为 {param_name: (min_value, max_value)}
            n_iter: 迭代次数
            metric: 优化指标，如'sharpe_ratio', 'annual_return', 'calmar_ratio'等
            top_n: 返回前N个结果
            
        返回:
            优化结果列表
        """
        if not param_bounds:
            logger.warning("参数边界为空，无法进行优化")
            return []
        
        logger.info(f"开始贝叶斯优化参数，迭代次数: {n_iter}")
        
        # 初始化结果列表
        self.optimization_results = []
        
        try:
            # 定义目标函数
            def objective(**params):
                # 合并基础参数和优化参数
                full_params = {**self.base_params, **params}
                
                # 评估参数
                result = self._evaluate_params(full_params)
                
                # 保存结果
                self.optimization_results.append(result)
                
                # 返回优化指标
                return result['metrics'].get(metric, float('-inf'))
            
            # 创建贝叶斯优化器
            optimizer = BayesianOptimization(
                f=objective,
                pbounds=param_bounds,
                random_state=42
            )
            
            # 运行优化
            optimizer.maximize(init_points=5, n_iter=n_iter)
            
            # 按指定指标排序
            self.optimization_results.sort(key=lambda x: x['metrics'].get(metric, float('-inf')), reverse=True)
            
            # 返回前N个结果
            top_results = self.optimization_results[:top_n]
            
            logger.info(f"贝叶斯优化参数完成，最佳 {metric}: {top_results[0]['metrics'].get(metric) if top_results else 'N/A'}")
            return top_results
        
        except Exception as e:
            logger.error(f"贝叶斯优化参数失败: {e}")
            return []
    
    def genetic_algorithm(self, param_bounds: Dict[str, Tuple[float, float]], population_size: int = 50, generations: int = 10, metric: str = 'sharpe_ratio', top_n: int = 10) -> List[Dict]:
        """
        遗传算法优化参数
        
        参数:
            param_bounds: 参数边界，格式为 {param_name: (min_value, max_value)}
            population_size: 种群大小
            generations: 迭代代数
            metric: 优化指标，如'sharpe_ratio', 'annual_return', 'calmar_ratio'等
            top_n: 返回前N个结果
            
        返回:
            优化结果列表
        """
        if not param_bounds:
            logger.warning("参数边界为空，无法进行优化")
            return []
        
        logger.info(f"开始遗传算法优化参数，种群大小: {population_size}, 迭代代数: {generations}")
        
        # 初始化结果列表
        self.optimization_results = []
        
        try:
            # 参数名称列表
            param_names = list(param_bounds.keys())
            
            # 生成初始种群
            population = []
            for _ in range(population_size):
                individual = {}
                for param_name, (min_val, max_val) in param_bounds.items():
                    individual[param_name] = np.random.uniform(min_val, max_val)
                population.append(individual)
            
            # 迭代优化
            for generation in range(generations):
                logger.info(f"遗传算法第 {generation+1}/{generations} 代")
                
                # 评估种群
                fitness_scores = []
                for individual in tqdm(population, desc=f"评估第 {generation+1} 代"):
                    # 合并基础参数和优化参数
                    full_params = {**self.base_params, **individual}
                    
                    # 评估参数
                    result = self._evaluate_params(full_params)
                    
                    # 保存结果
                    self.optimization_results.append(result)
                    
                    # 计算适应度
                    fitness = result['metrics'].get(metric, float('-inf'))
                    fitness_scores.append(fitness)
                
                # 选择父代
                parents_indices = np.argsort(fitness_scores)[-population_size//2:]
                parents = [population[i] for i in parents_indices]
                
                # 生成下一代
                next_population = []
                
                # 精英保留
                elite_count = max(1, population_size // 10)
                for i in range(elite_count):
                    next_population.append(population[parents_indices[-i-1]])
                
                # 交叉和变异
                while len(next_population) < population_size:
                    # 选择两个父代
                    parent1, parent2 = np.random.choice(parents, 2, replace=False)
                    
                    # 交叉
                    child = {}
                    for param_name in param_names:
                        # 随机选择一个父代的基因
                        if np.random.random() < 0.5:
                            child[param_name] = parent1[param_name]
                        else:
                            child[param_name] = parent2[param_name]
                    
                    # 变异
                    for param_name, (min_val, max_val) in param_bounds.items():
                        if np.random.random() < 0.1:  # 10%的变异概率
                            mutation_range = (max_val - min_val) * 0.1  # 变异范围为参数范围的10%
                            child[param_name] += np.random.uniform(-mutation_range, mutation_range)
                            # 确保在边界内
                            child[param_name] = max(min_val, min(max_val, child[param_name]))
                    
                    next_population.append(child)
                
                # 更新种群
                population = next_population
            
            # 按指定指标排序
            self.optimization_results.sort(key=lambda x: x['metrics'].get(metric, float('-inf')), reverse=True)
            
            # 返回前N个结果
            top_results = self.optimization_results[:top_n]
            
            logger.info(f"遗传算法优化参数完成，最佳 {metric}: {top_results[0]['metrics'].get(metric) if top_results else 'N/A'}")
            return top_results
        
        except Exception as e:
            logger.error(f"遗传算法优化参数失败: {e}")
            return []
    
    def _evaluate_params(self, params: Dict) -> Dict:
        """
        评估参数组合
        
        参数:
            params: 参数字典
            
        返回:
            评估结果字典
        """
        try:
            # 创建回测引擎的副本
            backtest_engine_copy = BacktestEngine(self.backtest_engine.config)
            
            # 运行回测
            results = backtest_engine_copy.run_backtest(self.data, self.strategy_func, params)
            
            # 返回结果
            return {
                'params': params,
                'metrics': results.get('metrics', {})
            }
        
        except Exception as e:
            logger.error(f"参数评估失败: {e}")
            return {
                'params': params,
                'metrics': {}
            }
    
    def plot_optimization_results(self, param_names: List[str], metric: str = 'sharpe_ratio', save_path: str = None):
        """
        绘制优化结果
        
        参数:
            param_names: 要绘制的参数名称列表
            metric: 优化指标
            save_path: 图表保存路径，如果为None则显示图表
        """
        if not self.optimization_results:
            logger.warning("无法绘制优化结果，结果为空")
            return
        
        if len(param_names) == 0:
            logger.warning("无法绘制优化结果，参数名称列表为空")
            return
        
        try:
            # 提取参数和指标
            params_values = []
            metric_values = []
            
            for result in self.optimization_results:
                params = result['params']
                metrics = result['metrics']
                
                # 检查参数和指标是否存在
                if all(param_name in params for param_name in param_names) and metric in metrics:
                    params_values.append([params[param_name] for param_name in param_names])
                    metric_values.append(metrics[metric])
            
            if not params_values:
                logger.warning("无法绘制优化结果，没有有效的参数和指标")
                return
            
            # 转换为numpy数组
            params_values = np.array(params_values)
            metric_values = np.array(metric_values)
            
            # 创建图表
            if len(param_names) == 1:
                # 一维参数空间
                plt.figure(figsize=(10, 6))
                plt.scatter(params_values[:, 0], metric_values, alpha=0.7)
                plt.xlabel(param_names[0])
                plt.ylabel(metric)
                plt.title(f"参数优化结果 - {metric}")
                plt.grid(True, alpha=0.3)
                
                # 添加趋势线
                try:
                    z = np.polyfit(params_values[:, 0], metric_values, 2)
                    p = np.poly1d(z)
                    x_range = np.linspace(min(params_values[:, 0]), max(params_values[:, 0]), 100)
                    plt.plot(x_range, p(x_range), 'r--', alpha=0.7)
                except:
                    pass
            
            elif len(param_names) == 2:
                # 二维参数空间
                plt.figure(figsize=(10, 8))
                
                # 创建散点图
                scatter = plt.scatter(
                    params_values[:, 0],
                    params_values[:, 1],
                    c=metric_values,
                    cmap='viridis',
                    alpha=0.7,
                    s=50
                )
                
                plt.xlabel(param_names[0])
                plt.ylabel(param_names[1])
                plt.title(f"参数优化结果 - {metric}")
                plt.colorbar(scatter, label=metric)
                plt.grid(True, alpha=0.3)
                
                # 标记最佳点
                best_idx = np.argmax(metric_values)
                plt.scatter(
                    params_values[best_idx, 0],
                    params_values[best_idx, 1],
                    c='red',
                    marker='*',
                    s=200,
                    label=f'最佳: {metric_values[best_idx]:.4f}'
                )
                plt.legend()
            
            else:
                # 多维参数空间，使用平行坐标图
                plt.figure(figsize=(12, 8))
                
                # 创建DataFrame
                df = pd.DataFrame(params_values, columns=param_names)
                df[metric] = metric_values
                
                # 归一化参数
                for param in param_names:
                    df[param] = (df[param] - df[param].min()) / (df[param].max() - df[param].min())
                
                # 按指标排序
                df = df.sort_values(by=metric)
                
                # 绘制平行坐标图
                pd.plotting.parallel_coordinates(df, metric, colormap='viridis')
                plt.title(f"参数优化结果 - {metric}")
                plt.grid(True, alpha=0.3)
            
            # 保存或显示图表
            if save_path:
                plt.savefig(save_path)
                logger.info(f"优化结果图表已保存到: {save_path}")
            else:
                plt.show()
                logger.info("优化结果图表已显示")
            
            plt.close()
        
        except Exception as e:
            logger.error(f"绘制优化结果图表失败: {e}")
    
    def save_optimization_results(self, file_path: str):
        """
        保存优化结果
        
        参数:
            file_path: 保存路径
        """
        if not self.optimization_results:
            logger.warning("无法保存优化结果，结果为空")
            return
        
        try:
            # 保存为JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.optimization_results, f, ensure_ascii=False, indent=4, default=str)
            
            logger.info(f"优化结果已保存到: {file_path}")
        
        except Exception as e:
            logger.error(f"保存优化结果失败: {e}")
    
    def load_optimization_results(self, file_path: str):
        """
        加载优化结果
        
        参数:
            file_path: 加载路径
        """
        try:
            # 从JSON加载
            with open(file_path, 'r', encoding='utf-8') as f:
                self.optimization_results = json.load(f)
            
            logger.info(f"优化结果已从 {file_path} 加载")
        
        except Exception as e:
            logger.error(f"加载优化结果失败: {e}")


# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建示例数据
    dates = pd.date_range(start='2023-01-01', periods=252, freq='B')  # 252个交易日
    
    # 模拟股票数据
    np.random.seed(42)
    
    # 创建价格序列
    price = 100
    prices = [price]
    for _ in range(251):
        change = np.random.normal(0, 1) / 100  # 每日涨跌幅服从正态分布
        price *= (1 + change)
        prices.append(price)
    
    # 创建OHLCV数据
    data = pd.DataFrame({
        'open': prices[:-1],
        'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices[:-1]],
        'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices[:-1]],
        'close': prices[1:],
        'volume': np.random.normal(1000000, 200000, 251)
    }, index=dates)
    
    # 添加一些技术指标
    data['sma_10'] = data['close'].rolling(window=10).mean()
    data['sma_30'] = data['close'].rolling(window=30).mean()
    
    # 定义简单的移动平均交叉策略
    def ma_cross_strategy(current_data, params):
        signals = {}
        
        for symbol, data_point in current_data.items():
            # 检查是否有必要的数据
            if 'sma_10' not in data_point or 'sma_30' not in data_point:
                signals[symbol] = 'hold'
                continue
            
            # 获取当前的移动平均值
            sma_fast = data_point['sma_10']
            sma_slow = data_point['sma_30']
            
            # 生成信号
            if sma_fast > sma_slow * (1 + params.get('threshold', 0)):
                signals[symbol] = 'buy'
            elif sma_fast < sma_slow * (1 - params.get('threshold', 0)):
                signals[symbol] = 'sell'
            else:
                signals[symbol] = 'hold'
        
        return signals
    
    try:
        # 初始化回测引擎
        backtest_config = {
            'initial_capital': 1000000,
            'commission_rate': 0.0003,
            'slippage': 0.0001,
            'tax_rate': 0.001
        }
        
        backtest_engine = BacktestEngine(backtest_config)
        
        # 运行回测
        strategy_params = {'threshold': 0.01}
        results = backtest_engine.run_backtest({'AAPL': data}, ma_cross_strategy, strategy_params)
        
        # 打印性能指标
        if results and 'metrics' in results:
            print("回测性能指标:")
            for key, value in results['metrics'].items():
                print(f"{key}: {value}")
        
        # 绘制回测结果
        backtest_engine.plot_results()
        
        # 参数优化
        optimizer = ParameterOptimizer(backtest_engine, {'AAPL': data}, ma_cross_strategy)
        
        # 网格搜索
        param_grid = {'threshold': [0.005, 0.01, 0.015, 0.02, 0.025]}
        grid_results = optimizer.grid_search(param_grid, metric='sharpe_ratio')
        
        if grid_results:
            print("\n网格搜索最佳参数:")
            print(f"参数: {grid_results[0]['params']}")
            print(f"夏普比率: {grid_results[0]['metrics'].get('sharpe_ratio')}")
        
        # 绘制优化结果
        optimizer.plot_optimization_results(['threshold'], metric='sharpe_ratio')
        
    except Exception as e:
        logging.error(f"测试过程中发生错误: {e}")
