# -*- coding: utf-8 -*-

import re 
import os 
import sys 
import logging 
import json 
from flask import Flask, render_template, request, jsonify, send_from_directory 
from flask_cors import CORS 
 
# Try to import TA-Lib, or use mock if not available 
try: 
    import talib 
except ImportError: 
    print("Warning: TA-Lib not installed. Using mock implementation.") 
    # Create a mock talib module 
    class TALibMock: 
        def __getattr__(self, name): 
            def mock_func(*args, **kwargs): 
                print(f"Warning: Attempted to use uninstalled TA-Lib function {name}") 
                if name in ['MA', 'SMA', 'EMA']: 
                    # For moving averages, return the input data 
                    return args[0] 
                return None 
            return mock_func 
    sys.modules['talib'] = TALibMock() 
 
"""
Web界面模块 - 主控制器

本模块实现了Web界面的主控制器，负责处理用户请求和页面渲染。
主要功能包括：
1. 仪表盘展示
2. 股票市场监控
3. 加密货币市场监控
4. 回测系统
5. 交易系统
6. 系统设置

日期：2025-05-16
"""

from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional, Tuple, Any
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.express as px

# 导入自定义模块
from src.data_sources.stock_data import StockDataSource
from src.data_sources.crypto_data import CryptoDataSource
from src.strategies.smart_money_detector import SmartMoneyDetector
from src.backtesting.backtest_engine import BacktestEngine
from src.core.trade_executor import SimulatedTradeExecutor, BrokerAPIExecutor
from src.risk_management.risk_controller import RiskController

# 配置日志
logger = logging.getLogger(__name__)

class WebController:
    """Web界面控制器，负责处理用户请求和页面渲染"""
    
    def __init__(self, config_path: str = None):
        """
        初始化Web控制器

        参数:
            config_path: 配置文件路径
        """
        # 系统日志
        self.system_logs = []

        # 加载配置
        self.config = self._load_config(config_path)

        # 创建Flask应用
        self.app = Flask(__name__, 
                         template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                         static_folder=os.path.join(os.path.dirname(__file__), 'static'))

        # 设置密钥
        self.app.secret_key = self.config.get("secret_key", os.urandom(24).hex())

        # 初始化组件
        self._init_components()

        # 注册路由
        self._register_routes()

        # 系统状态
        self.system_status = {
            "data_source": "正常运行",
            "strategy_engine": "正常运行",
            "backtest_engine": "正常运行",
            "trade_executor": "正常运行",
            "risk_management": "正常运行",
            "web_interface": "正常运行"
        }

        # 系统启动时间
        self.start_time = datetime.now()

        logger.info("Web控制器初始化成功")
    
    def _load_config(self, config_path: str = None) -> Dict:
        """
        加载配置
        
        参数:
            config_path: 配置文件路径
            
        返回:
            配置字典
        """
        # 默认配置
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
                    "api_secret": ""
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
        
        # 如果提供了配置文件路径，尝试加载
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                
                # 合并配置
                self._merge_config(default_config, user_config)
                
                logger.info(f"配置已从 {config_path} 加载")
            except Exception as e:
                logger.error(f"加载配置失败: {e}")
        
        return default_config
    
    def _merge_config(self, default_config: Dict, user_config: Dict):
        """
        合并配置
        
        参数:
            default_config: 默认配置
            user_config: 用户配置
        """
        for key, value in user_config.items():
            if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                self._merge_config(default_config[key], value)
            else:
                default_config[key] = value
    
    def _init_components(self):
        """初始化组件"""
        try:
            # 初始化数据源
            self.stock_data_source = StockDataSource(self.config["data_source"]["stock"])
            self.crypto_data_source = CryptoDataSource(self.config["data_source"]["crypto"])
            
            # 初始化策略引擎
            self.smart_money_detector = SmartMoneyDetector()
            
            # 初始化回测引擎
            self.backtest_engine = BacktestEngine(self.config["backtest"])
            
            # 初始化交易执行器
            if self.config["trade"]["default_mode"] == "simulated":
                self.trade_executor = SimulatedTradeExecutor(self.config["trade"])
            else:
                self.trade_executor = BrokerAPIExecutor(self.config["trade"]["broker_api"])
            
            # 初始化风险控制器
            self.risk_controller = RiskController(self.config["risk_management"])
            
            # 记录系统日志
            self._log_system_event("系统组件初始化成功")
            
        except Exception as e:
            logger.error(f"初始化组件失败: {e}")
            self._log_system_event(f"系统组件初始化失败: {e}", "error")
    
    def _register_routes(self):
        """注册路由"""
        # 主页和仪表盘
        self.app.route('/')(self.index)
        self.app.route('/dashboard')(self.dashboard)
        
        # 股票市场
        self.app.route('/stock_market')(self.stock_market)
        self.app.route('/api/stock/search', methods=['POST'])(self.api_stock_search)
        self.app.route('/api/stock/data', methods=['POST'])(self.api_stock_data)
        self.app.route('/api/stock/analysis', methods=['POST'])(self.api_stock_analysis)
        
        # 加密货币市场
        self.app.route('/crypto_market')(self.crypto_market)
        self.app.route('/api/crypto/search', methods=['POST'])(self.api_crypto_search)
        self.app.route('/api/crypto/data', methods=['POST'])(self.api_crypto_data)
        self.app.route('/api/crypto/analysis', methods=['POST'])(self.api_crypto_analysis)
        
        # 回测系统
        self.app.route('/backtest')(self.backtest)
        self.app.route('/api/backtest/run', methods=['POST'])(self.api_backtest_run)
        self.app.route('/api/backtest/results', methods=['GET'])(self.api_backtest_results)
        self.app.route('/api/backtest/optimize', methods=['POST'])(self.api_backtest_optimize)
        
        # 交易系统
        self.app.route('/trading')(self.trading)
        self.app.route('/api/trading/account', methods=['GET'])(self.api_trading_account)
        self.app.route('/api/trading/positions', methods=['GET'])(self.api_trading_positions)
        self.app.route('/api/trading/orders', methods=['GET'])(self.api_trading_orders)
        self.app.route('/api/trading/place_order', methods=['POST'])(self.api_trading_place_order)
        self.app.route('/api/trading/cancel_order', methods=['POST'])(self.api_trading_cancel_order)
        
        # 系统设置
        self.app.route('/settings')(self.settings)
        self.app.route('/api/settings/update', methods=['POST'])(self.api_settings_update)
        self.app.route('/api/settings/broker_connect', methods=['POST'])(self.api_settings_broker_connect)
        
        # 系统状态
        self.app.route('/api/system/status', methods=['GET'])(self.api_system_status)
        self.app.route('/api/system/logs', methods=['GET'])(self.api_system_logs)
    
    def run(self, host: str = None, port: int = None, debug: bool = None):
        """
        运行Web应用
        
        参数:
            host: 主机地址
            port: 端口号
            debug: 是否开启调试模式
        """
        host = host or self.config.get("host", "0.0.0.0")
        port = port or self.config.get("port", 5000)
        debug = debug if debug is not None else self.config.get("debug", False)
        
        logger.info(f"Web应用启动 - 主机: {host}, 端口: {port}, 调试模式: {debug}")
        self._log_system_event(f"Web应用启动 - 主机: {host}, 端口: {port}")
        
        self.app.run(host=host, port=port, debug=debug)
    
    def _log_system_event(self, message: str, level: str = "info"):
        """
        记录系统事件
        
        参数:
            message: 事件消息
            level: 事件级别，可以是'info', 'warning', 'error'
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        
        self.system_logs.append(log_entry)
        
        # 限制日志数量
        if len(self.system_logs) > 1000:
            self.system_logs = self.system_logs[-1000:]
        
        # 根据级别记录日志
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
    
    # 路由处理函数
    
    def index(self):
        """主页"""
        return redirect(url_for('dashboard'))
    
    def dashboard(self):
        """仪表盘"""
        # 获取系统状态
        system_status = self.system_status
        
        # 获取运行时间
        uptime = datetime.now() - self.start_time
        uptime_str = f"{uptime.days}天 {uptime.seconds // 3600}小时 {(uptime.seconds // 60) % 60}分钟"
        
        # 获取CPU和内存使用率
        try:
            import psutil
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
        except ImportError:
            cpu_usage = 25  # 模拟值
            memory_usage = 30  # 模拟值
        
        # 获取最新日志
        latest_logs = self.system_logs[-10:] if self.system_logs else []
        
        # 获取账户信息
        account_info = self.trade_executor.get_account_info()
        
        # 获取持仓信息
        positions = self.trade_executor.get_positions()
        
        # 渲染模板
        return render_template(
            'dashboard.html',
            system_status=system_status,
            uptime=uptime_str,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            latest_logs=latest_logs,
            account_info=account_info,
            positions=positions,
            system_version="1.0.0",
            start_time=self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def stock_market(self):
        """股票市场"""
        return render_template('stock_market.html')
    
    def api_stock_search(self):
        """股票搜索API"""
        try:
            # 获取请求参数
            data = request.get_json()
            keyword = data.get('keyword', '')
            
            # 搜索股票
            results = self.stock_data_source.search_stocks(keyword)
            
            return jsonify({"success": True, "data": results})
        
        except Exception as e:
            logger.error(f"股票搜索失败: {e}")
            return jsonify({"success": False, "message": str(e)})
    
    def api_stock_data(self):
        """股票数据API"""
        try:
            # 获取请求参数
            data = request.get_json()
            stock_code = data.get('stock_code', '')
            start_date = data.get('start_date', '')
            end_date = data.get('end_date', '')
            
            # 获取股票数据
            stock_data = self.stock_data_source.get_daily_data(stock_code, start_date, end_date)
            
            # 转换为JSON格式
            stock_data_json = stock_data.to_dict(orient='records')
            
            return jsonify({"success": True, "data": stock_data_json})
        
        except Exception as e:
            logger.error(f"获取股票数据失败: {e}")
            return jsonify({"success": False, "message": str(e)})
    
    def api_stock_analysis(self):
        """股票分析API"""
        try:
            # 获取请求参数
            data = request.get_json()
            stock_code = data.get('stock_code', '')
            start_date = data.get('start_date', '')
            end_date = data.get('end_date', '')
            
            # 获取股票数据
            stock_data = self.stock_data_source.get_daily_data(stock_code, start_date, end_date)
            
            # 分析庄家行为
            analysis_result = self.smart_money_detector.analyze(stock_data)
            
            # 获取技术指标
            technical_indicators = self.stock_data_source.get_technical_indicators(stock_code, start_date, end_date)
            
            # 获取资金流向
            capital_flow = self.stock_data_source.get_capital_flow(stock_code, start_date, end_date)
            
            # 合并结果
            result = {
                "smart_money_analysis": analysis_result,
                "technical_indicators": technical_indicators.to_dict(orient='records') if isinstance(technical_indicators, pd.DataFrame) else technical_indicators,
                "capital_flow": capital_flow.to_dict(orient='records') if isinstance(capital_flow, pd.DataFrame) else capital_flow
            }
            
            return jsonify({"success": True, "data": result})
        
        except Exception as e:
            logger.error(f"股票分析失败: {e}")
            return jsonify({"success": False, "message": str(e)})
    
    def crypto_market(self):
        """加密货币市场"""
        return render_template('crypto_market.html')
    
    def api_crypto_search(self):
        """加密货币搜索API"""
        try:
            # 获取请求参数
            data = request.get_json()
            keyword = data.get('keyword', '')
            
            # 搜索加密货币
            results = self.crypto_data_source.search_cryptos(keyword)
            
            return jsonify({"success": True, "data": results})
        
        except Exception as e:
            logger.error(f"加密货币搜索失败: {e}")
            return jsonify({"success": False, "message": str(e)})
    
    def api_crypto_data(self):
        """加密货币数据API"""
        try:
            # 获取请求参数
            data = request.get_json()
            symbol = data.get('symbol', '')
            interval = data.get('interval', '1d')
            start_time = data.get('start_time', '')
            end_time = data.get('end_time', '')
            
            # 获取加密货币数据
            crypto_data = self.crypto_data_source.get_klines(symbol, interval, start_time, end_time)
            
            # 转换为JSON格式
            crypto_data_json = crypto_data.to_dict(orient='records')
            
            return jsonify({"success": True, "data": crypto_data_json})
        
        except Exception as e:
            logger.error(f"获取加密货币数据失败: {e}")
            return jsonify({"success": False, "message": str(e)})
    
    def api_crypto_analysis(self):
        """加密货币分析API"""
        try:
            # 获取请求参数
            data = request.get_json()
            symbol = data.get('symbol', '')
            interval = data.get('interval', '1d')
            start_time = data.get('start_time', '')
            end_time = data.get('end_time', '')
            
            # 获取加密货币数据
            crypto_data = self.crypto_data_source.get_klines(symbol, interval, start_time, end_time)
            
            # 分析庄家行为
            analysis_result = self.smart_money_detector.analyze(crypto_data)
            
            # 获取技术指标
            technical_indicators = self.crypto_data_source.get_technical_indicators(symbol, interval, start_time, end_time)
            
            # 获取订单簿数据
            order_book = self.crypto_data_source.get_order_book(symbol)
            
            # 合并结果
            result = {
                "smart_money_analysis": analysis_result,
                "technical_indicators": technical_indicators.to_dict(orient='records') if isinstance(technical_indicators, pd.DataFrame) else technical_indicators,
                "order_book": order_book
            }
            
            return jsonify({"success": True, "data": result})
        
        except Exception as e:
            logger.error(f"加密货币分析失败: {e}")
            return jsonify({"success": False, "message": str(e)})
    
    def backtest(self):
        """回测系统"""
        # 获取可用的策略参数
        try:
            strategy_params = self.smart_money_detector.get_params()
        except AttributeError:
            strategy_params = {
                'volume_threshold': 2.0,
                'price_threshold': 0.02,
                'ma_periods': [5, 10, 20, 60],
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'pattern_threshold': 0.7
            }
        
        # 获取可用的优化方法
        optimization_methods = [
            {"id": "grid_search", "name": "网格搜索", "description": "系统地搜索参数空间中的所有组合"},
            {"id": "random_search", "name": "随机搜索", "description": "随机采样参数空间中的组合"},
            {"id": "bayesian_optimization", "name": "贝叶斯优化", "description": "使用贝叶斯方法智能搜索最优参数"},
            {"id": "genetic_algorithm", "name": "遗传算法", "description": "使用进化算法寻找最优参数组合"}
        ]
        
        # 获取历史回测结果
        backtest_results = self.backtest_engine.get_backtest_history()
        
        return render_template(
            'backtest.html',
            strategy_params=strategy_params,
            optimization_methods=optimization_methods,
            backtest_results=backtest_results
        )
    
    def api_backtest_run(self):
        """运行回测API"""
        try:
            # 获取请求参数
            data = request.get_json()
            
            # 提取参数
            symbol = data.get('symbol', '')
            start_date = data.get('start_date', '')
            end_date = data.get('end_date', '')
            initial_capital = data.get('initial_capital', 1000000)
            strategy_params = data.get('strategy_params', {})
            
            # 记录系统日志
            self._log_system_event(f"开始回测 - 品种: {symbol}, 时间范围: {start_date} 至 {end_date}")
            
            # 运行回测
            backtest_id = self.backtest_engine.run_backtest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                strategy=self.smart_money_detector,
                strategy_params=strategy_params
            )
            
            return jsonify({"success": True, "backtest_id": backtest_id})
        
        except Exception as e:
            logger.error(f"运行回测失败: {e}")
            self._log_system_event(f"回测失败: {e}", "error")
            return jsonify({"success": False, "message": str(e)})
    
    def api_backtest_results(self):
        """获取回测结果API"""
        try:
            # 获取请求参数
            backtest_id = request.args.get('backtest_id', '')
            
            # 获取回测结果
            results = self.backtest_engine.get_backtest_results(backtest_id)
            
            if not results:
                return jsonify({"success": False, "message": "回测结果不存在"})
            
            # 生成回测图表
            equity_curve = self._generate_equity_curve(results)
            drawdown_chart = self._generate_drawdown_chart(results)
            monthly_returns = self._generate_monthly_returns(results)
            
            # 转换为JSON格式
            charts = {
                "equity_curve": json.loads(equity_curve),
                "drawdown_chart": json.loads(drawdown_chart),
                "monthly_returns": json.loads(monthly_returns)
            }
            
            # 提取性能指标
            performance_metrics = results.get("performance_metrics", {})
            
            # 提取交易记录
            trades = results.get("trades", [])
            
            return jsonify({
                "success": True, 
                "data": {
                    "charts": charts,
                    "performance_metrics": performance_metrics,
                    "trades": trades
                }
            })
        
        except Exception as e:
            logger.error(f"获取回测结果失败: {e}")
            return jsonify({"success": False, "message": str(e)})
    
    def api_backtest_optimize(self):
        """参数优化API"""
        try:
            # 获取请求参数
            data = request.get_json()
            
            # 提取参数
            symbol = data.get('symbol', '')
            start_date = data.get('start_date', '')
            end_date = data.get('end_date', '')
            initial_capital = data.get('initial_capital', 1000000)
            param_ranges = data.get('param_ranges', {})
            optimization_method = data.get('optimization_method', 'grid_search')
            optimization_target = data.get('optimization_target', 'sharpe_ratio')
            
            # 记录系统日志
            self._log_system_event(f"开始参数优化 - 品种: {symbol}, 优化方法: {optimization_method}")
            
            # 运行参数优化
            optimization_id = self.backtest_engine.optimize_parameters(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                strategy=self.smart_money_detector,
                param_ranges=param_ranges,
                optimization_method=optimization_method,
                optimization_target=optimization_target
            )
            
            return jsonify({"success": True, "optimization_id": optimization_id})
        
        except Exception as e:
            logger.error(f"参数优化失败: {e}")
            self._log_system_event(f"参数优化失败: {e}", "error")
            return jsonify({"success": False, "message": str(e)})
    
    def trading(self):
        """交易系统"""
        # 获取账户信息
        account_info = self.trade_executor.get_account_info()
        
        # 获取持仓信息
        positions = self.trade_executor.get_positions()
        
        # 获取订单信息
        orders = self.trade_executor.orders
        
        # 获取交易摘要
        trade_summary = self.trade_executor.get_trade_summary()
        
        # 获取风险报告
        risk_report = self.risk_controller.get_risk_report()
        
        return render_template(
            'trading.html',
            account_info=account_info,
            positions=positions,
            orders=orders,
            trade_summary=trade_summary,
            risk_report=risk_report,
            trading_mode=self.config["trade"]["default_mode"]
        )
    
    def api_trading_account(self):
        """获取账户信息API"""
        try:
            # 获取账户信息
            account_info = self.trade_executor.get_account_info()
            
            return jsonify({"success": True, "data": account_info})
        
        except Exception as e:
            logger.error(f"获取账户信息失败: {e}")
            return jsonify({"success": False, "message": str(e)})
    
    def api_trading_positions(self):
        """获取持仓信息API"""
        try:
            # 获取持仓信息
            positions = self.trade_executor.get_positions()
            
            # 转换为列表格式
            positions_list = [position.to_dict() for position in positions.values()]
            
            return jsonify({"success": True, "data": positions_list})
        
        except Exception as e:
            logger.error(f"获取持仓信息失败: {e}")
            return jsonify({"success": False, "message": str(e)})
    
    def api_trading_orders(self):
        """获取订单信息API"""
        try:
            # 获取订单信息
            orders = self.trade_executor.orders
            
            # 转换为列表格式
            orders_list = [order.to_dict() for order in orders.values()]
            
            return jsonify({"success": True, "data": orders_list})
        
        except Exception as e:
            logger.error(f"获取订单信息失败: {e}")
            return jsonify({"success": False, "message": str(e)})
    
    def api_trading_place_order(self):
        """下单API"""
        try:
            # 获取请求参数
            data = request.get_json()
            
            # 提取参数
            symbol = data.get('symbol', '')
            order_type = data.get('order_type', 'market')
            direction = data.get('direction', 'buy')
            quantity = float(data.get('quantity', 0))
            price = float(data.get('price', 0)) if data.get('price') else None
            stop_price = float(data.get('stop_price', 0)) if data.get('stop_price') else None
            
            # 检查参数
            if not symbol:
                return jsonify({"success": False, "message": "交易品种不能为空"})
            
            if quantity <= 0:
                return jsonify({"success": False, "message": "交易数量必须大于0"})
            
            if order_type in ['limit', 'stop_limit'] and (price is None or price <= 0):
                return jsonify({"success": False, "message": "限价单必须指定有效的价格"})
            
            if order_type in ['stop', 'stop_limit'] and (stop_price is None or stop_price <= 0):
                return jsonify({"success": False, "message": "止损单必须指定有效的止损价格"})
            
            # 获取账户信息和持仓信息
            account_info = self.trade_executor.get_account_info()
            positions = self.trade_executor.get_positions()
            
            # 检查风险
            allowed, reason = self.risk_controller.check_trade_risk(
                symbol=symbol,
                direction=direction,
                quantity=quantity,
                price=price or 0,
                account_info=account_info,
                positions=positions
            )
            
            if not allowed:
                return jsonify({"success": False, "message": f"风险控制拒绝交易: {reason}"})
            
            # 创建订单
            from src.core.trade_executor import Order
            order = Order(
                symbol=symbol,
                order_type=order_type,
                direction=direction,
                quantity=quantity,
                price=price,
                stop_price=stop_price
            )
            
            # 下单
            order_id = self.trade_executor.place_order(order)
            
            # 记录系统日志
            self._log_system_event(f"下单成功 - ID: {order_id}, 品种: {symbol}, 方向: {direction}, 数量: {quantity}")
            
            return jsonify({"success": True, "order_id": order_id})
        
        except Exception as e:
            logger.error(f"下单失败: {e}")
            self._log_system_event(f"下单失败: {e}", "error")
            return jsonify({"success": False, "message": str(e)})
    
    def api_trading_cancel_order(self):
        """取消订单API"""
        try:
            # 获取请求参数
            data = request.get_json()
            order_id = data.get('order_id', '')
            
            # 检查参数
            if not order_id:
                return jsonify({"success": False, "message": "订单ID不能为空"})
            
            # 取消订单
            success = self.trade_executor.cancel_order(order_id)
            
            if success:
                # 记录系统日志
                self._log_system_event(f"取消订单成功 - ID: {order_id}")
                return jsonify({"success": True})
            else:
                return jsonify({"success": False, "message": "取消订单失败"})
        
        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            self._log_system_event(f"取消订单失败: {e}", "error")
            return jsonify({"success": False, "message": str(e)})
    
    def settings(self):
        """系统设置"""
        return render_template('settings.html', config=self.config)
    
    def api_settings_update(self):
        """更新设置API"""
        try:
            # 获取请求参数
            data = request.get_json()
            
            # 更新配置
            self._merge_config(self.config, data)
            
            # 保存配置
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
            
            # 记录系统日志
            self._log_system_event("系统设置已更新")
            
            return jsonify({"success": True})
        
        except Exception as e:
            logger.error(f"更新设置失败: {e}")
            self._log_system_event(f"更新设置失败: {e}", "error")
            return jsonify({"success": False, "message": str(e)})
    
    def api_settings_broker_connect(self):
        """连接券商API"""
        try:
            # 获取请求参数
            data = request.get_json()
            
            # 提取参数
            broker_name = data.get('broker_name', '')
            api_key = data.get('api_key', '')
            api_secret = data.get('api_secret', '')
            api_base_url = data.get('api_base_url', '')
            
            # 检查参数
            if not broker_name:
                return jsonify({"success": False, "message": "券商名称不能为空"})
            
            if not api_key:
                return jsonify({"success": False, "message": "API密钥不能为空"})
            
            if not api_secret:
                return jsonify({"success": False, "message": "API密钥不能为空"})
            
            # 更新配置
            self.config["trade"]["default_mode"] = "broker_api"
            self.config["trade"]["broker_api"] = {
                "name": broker_name,
                "api_key": api_key,
                "api_secret": api_secret,
                "api_base_url": api_base_url
            }
            
            # 重新初始化交易执行器
            self.trade_executor = BrokerAPIExecutor(self.config["trade"]["broker_api"])
            
            # 连接券商API
            success = self.trade_executor.connect()
            
            if success:
                # 记录系统日志
                self._log_system_event(f"连接券商API成功 - {broker_name}")
                return jsonify({"success": True})
            else:
                return jsonify({"success": False, "message": "连接券商API失败"})
        
        except Exception as e:
            logger.error(f"连接券商API失败: {e}")
            self._log_system_event(f"连接券商API失败: {e}", "error")
            return jsonify({"success": False, "message": str(e)})
    
    def api_system_status(self):
        """获取系统状态API"""
        try:
            # 获取系统状态
            system_status = self.system_status
            
            # 获取运行时间
            uptime = datetime.now() - self.start_time
            uptime_str = f"{uptime.days}天 {uptime.seconds // 3600}小时 {(uptime.seconds // 60) % 60}分钟"
            
            # 获取CPU和内存使用率
            try:
                import psutil
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
            except ImportError:
                cpu_usage = 25  # 模拟值
                memory_usage = 30  # 模拟值
            
            return jsonify({
                "success": True, 
                "data": {
                    "system_status": system_status,
                    "uptime": uptime_str,
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "system_version": "1.0.0",
                    "start_time": self.start_time.isoformat()
                }
            })
        
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return jsonify({"success": False, "message": str(e)})
    
    def api_system_logs(self):
        """获取系统日志API"""
        try:
            # 获取请求参数
            limit = int(request.args.get('limit', 100))
            level = request.args.get('level', 'all')
            
            # 过滤日志
            if level == 'all':
                logs = self.system_logs[-limit:]
            else:
                logs = [log for log in self.system_logs if log["level"] == level][-limit:]
            
            return jsonify({"success": True, "data": logs})
        
        except Exception as e:
            logger.error(f"获取系统日志失败: {e}")
            return jsonify({"success": False, "message": str(e)})
    
    # 辅助方法
    
    def _generate_equity_curve(self, results: Dict) -> str:
        """
        生成权益曲线图表
        
        参数:
            results: 回测结果
            
        返回:
            JSON格式的图表
        """
        # 提取权益曲线数据
        equity_curve = results.get("equity_curve", pd.DataFrame())
        
        if isinstance(equity_curve, dict):
            equity_curve = pd.DataFrame(equity_curve)
        
        if equity_curve.empty:
            return "{}"
        
        # 创建图表
        fig = go.Figure()
        
        # 添加权益曲线
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve['total_value'],
            mode='lines',
            name='权益曲线',
            line=dict(color='blue', width=2)
        ))
        
        # 添加基准曲线（如果有）
        if 'benchmark' in equity_curve.columns:
            fig.add_trace(go.Scatter(
                x=equity_curve.index,
                y=equity_curve['benchmark'],
                mode='lines',
                name='基准',
                line=dict(color='gray', width=1, dash='dash')
            ))
        
        # 设置图表布局
        fig.update_layout(
            title='权益曲线',
            xaxis_title='日期',
            yaxis_title='价值',
            legend=dict(x=0, y=1, traceorder='normal'),
            template='plotly_white'
        )
        
        # 转换为JSON
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def _generate_drawdown_chart(self, results: Dict) -> str:
        """
        生成回撤图表
        
        参数:
            results: 回测结果
            
        返回:
            JSON格式的图表
        """
        # 提取回撤数据
        equity_curve = results.get("equity_curve", pd.DataFrame())
        
        if isinstance(equity_curve, dict):
            equity_curve = pd.DataFrame(equity_curve)
        
        if equity_curve.empty or 'drawdown' not in equity_curve.columns:
            return "{}"
        
        # 创建图表
        fig = go.Figure()
        
        # 添加回撤曲线
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve['drawdown'],
            mode='lines',
            name='回撤',
            line=dict(color='red', width=2),
            fill='tozeroy'
        ))
        
        # 设置图表布局
        fig.update_layout(
            title='回撤曲线',
            xaxis_title='日期',
            yaxis_title='回撤',
            yaxis=dict(tickformat='.1%'),
            legend=dict(x=0, y=1, traceorder='normal'),
            template='plotly_white'
        )
        
        # 转换为JSON
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def _generate_monthly_returns(self, results: Dict) -> str:
        """
        生成月度收益图表
        
        参数:
            results: 回测结果
            
        返回:
            JSON格式的图表
        """
        # 提取月度收益数据
        monthly_returns = results.get("monthly_returns", pd.DataFrame())
        
        if isinstance(monthly_returns, dict):
            monthly_returns = pd.DataFrame(monthly_returns)
        
        if monthly_returns.empty:
            return "{}"
        
        # 创建图表
        fig = go.Figure()
        
        # 添加月度收益柱状图
        for year in monthly_returns.index.levels[0]:
            year_data = monthly_returns.loc[year]
            
            fig.add_trace(go.Bar(
                x=year_data.index,
                y=year_data['returns'],
                name=str(year),
                text=[f"{r:.2%}" for r in year_data['returns']],
                textposition='auto'
            ))
        
        # 设置图表布局
        fig.update_layout(
            title='月度收益',
            xaxis_title='月份',
            yaxis_title='收益率',
            yaxis=dict(tickformat='.1%'),
            barmode='group',
            legend=dict(x=0, y=1, traceorder='normal'),
            template='plotly_white'
        )
        
        # 转换为JSON
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


# 主程序入口
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建Web控制器
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
    controller = WebController(config_path)
    
    # 运行Web应用
    controller.run(host="0.0.0.0", port=5000, debug=True)
