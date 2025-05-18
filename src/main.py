# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
from datetime import datetime
import json
import platform
import webbrowser
import time
import threading
import gc
import psutil

# Get project root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure log directory exists
log_dir = os.path.join(root_dir, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'system.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# TA-Lib Mock Class
class TALibMock:
    """TA-Lib mock implementation"""

    def __getattr__(self, name):
        def mock_func(*args, **kwargs):
            print(f"Warning: Using mocked TA-Lib function {name}")
            if name in ['MA', 'SMA', 'EMA']:
                return args[0] if args else None
            return None

        return mock_func


def check_python_version():
    """Check Python version"""
    required_version = (3, 8)
    current_version = sys.version_info

    if current_version < required_version:
        logger.error(f"Python version too old. Need Python {required_version[0]}.{required_version[1]} or higher")
        print(f"Error: Python version too old. Need Python {required_version[0]}.{required_version[1]} or higher")
        return False

    return True


def check_dependencies():
    """Check dependencies"""
    # Add project root to Python path
    sys.path.append(root_dir)

    try:
        # Try to import core dependencies
        import pandas as pd
        import numpy as np
        import flask
        import plotly

        # Try to import TA-Lib, or use mock if not available
        try:
            import talib
        except ImportError:
            logger.warning("TA-Lib not installed. Some technical analysis features will be limited.")
            print("Warning: TA-Lib not installed. Some technical analysis features will be limited.")
            # Add mock talib to modules
            sys.modules['talib'] = TALibMock()

        logger.info("Dependency check passed")
        return True

    except ImportError as e:
        logger.error(f"Dependency check failed: {e}")
        print(f"Error: Missing required dependency: {e}")
        print("Please run 'pip install -r requirements.txt' to install all dependencies")
        return False


def create_directories():
    """Create necessary directories"""
    directories = [
        os.path.join(root_dir, 'logs'),
        os.path.join(root_dir, 'data'),
        os.path.join(root_dir, 'config'),
        os.path.join(root_dir, 'backtest_results')
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")


def create_default_config():
    """Create default config file"""
    config_path = os.path.join(root_dir, 'config', 'config.json')

    if not os.path.exists(config_path):
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

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=4)

        logger.info(f"Created default config file: {config_path}")


def open_browser(url):
    """Open URL in browser"""

    def _open_browser():
        time.sleep(1.5)  # Wait for server to start
        webbrowser.open(url)

    browser_thread = threading.Thread(target=_open_browser)
    browser_thread.daemon = True
    browser_thread.start()


def optimize_memory():
    """优化内存使用"""
    gc.collect()
    process = psutil.Process(os.getpid())
    try:
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
    except:
        pass  # 忽略优先级设置错误
    return process.memory_info().rss / 1024 / 1024  # 返回当前内存使用量（MB）


def setup_memory_optimization(app):
    """为Flask应用设置内存优化"""
    from flask import request

    @app.before_request
    def before_request():
        # 在每个请求前进行轻量级的垃圾回收
        gc.collect(0)

    @app.after_request
    def after_request(response):
        # 在处理高内存页面后进行更深入的垃圾回收
        if request.path in ['/backtest', '/stock_market', '/crypto_market']:
            mem_usage = optimize_memory()
            app.logger.debug(f"内存使用: {mem_usage:.2f} MB")
        return response


def main():
    """Main function"""
    # 显示欢迎信息
    print("=" * 80)
    print("                Smart Money Follow System v1.0.0                ")
    print("=" * 80)
    
    # 记录启动时间
    start_time = datetime.now()
    print(f"System start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 显示系统信息
    print(f"Operating system: {platform.system()} {platform.release()}")
    print(f"Python version: {sys.version.split()[0]}")
    print("=" * 80)
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 创建必要的目录
    create_directories()
    
    # 创建默认配置文件
    create_default_config()
    
    # 检查API配置
    config_path = os.path.join(root_dir, 'config', 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # 检查股票API配置
        stock_api_key = config.get('data_source', {}).get('stock', {}).get('api_key', '')
        if not stock_api_key:
            logger.warning("股票API密钥未配置，部分功能将受限")
            print("提示: 股票API密钥未配置，请在系统启动后访问 API配置 页面进行设置")
        
        # 检查加密货币API配置
        crypto_api_key = config.get('data_source', {}).get('crypto', {}).get('api_key', '')
        if not crypto_api_key:
            logger.warning("加密货币API密钥未配置，部分功能将受限")
            print("提示: 加密货币API密钥未配置，请在系统启动后访问 API配置 页面进行设置")
    except Exception as e:
        logger.error(f"检查API配置失败: {e}")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Smart Money Follow System')
    parser.add_argument('--host', type=str, help='Host to run the server on')
    parser.add_argument('--port', type=int, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    args = parser.parse_args()
    
    # 加载配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        config = {}
    
    # 获取主机和端口
    host = args.host or config.get('host', '0.0.0.0')
    port = args.port or config.get('port', 5000)
    debug = args.debug or config.get('debug', False)
    
    # 导入Web控制器
    try:
        from src.web_interface.web_controller import WebController
        
        # 创建Web控制器实例
        web_controller = WebController(config_path)
        
        # 设置内存优化
        setup_memory_optimization(web_controller.app)
        
        # 自动打开浏览器
        if not args.no_browser:
            url = f"http://{'localhost' if host == '0.0.0.0' else host}:{port}"
            open_browser(url)
        
        # 启动Web服务器
        web_controller.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Failed to import Web Controller: {e}")
        print(f"Error: Failed to import Web Controller: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()