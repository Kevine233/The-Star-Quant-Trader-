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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Smart Money Follow System")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    args = parser.parse_args()

    # Display welcome message
    print("=" * 80)
    print("                Smart Money Follow System v1.0.0                ")
    print("=" * 80)
    print(f"System start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Operating system: {platform.system()} {platform.release()}")
    print(f"Python version: {platform.python_version()}")
    print("=" * 80)

    # Check Python version
    if not check_python_version():
        return

    # Check dependencies
    if not check_dependencies():
        return

    # Create necessary directories
    create_directories()

    # Create default config file
    create_default_config()

    # Import Web Controller
    try:
        from web_interface.web_controller import WebController
    except ImportError as e:
        try:
            # Try relative import
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from src.web_interface.web_controller import WebController
        except ImportError as e2:
            logger.error(f"Failed to import Web Controller: {e2}")
            print(f"Error: Failed to import Web Controller: {e2}")
            return

    # Create Web Controller
    try:
        config_path = os.path.join(root_dir, 'config', 'config.json')
        controller = WebController(config_path)

        # If not in debug mode, automatically open browser
        if not args.debug and not args.no_browser:
            url = f"http://127.0.0.1:{args.port}"
            open_browser(url)

        # Run Web application
        controller.run(host=args.host, port=args.port, debug=args.debug)

    except Exception as e:
        logger.error(f"Failed to start Web application: {e}")
        print(f"Error: Failed to start Web application: {e}")
        return


if __name__ == "__main__":
    main()