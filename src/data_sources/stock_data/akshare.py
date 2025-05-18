"""
AKShare股票数据源

实现基于AKShare的股票数据获取功能。
主要用于获取A股市场的基础数据和龙虎榜数据。

日期：2025-05-17
"""

import datetime
import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any

from .base import StockDataSource

# 配置日志
logger = logging.getLogger(__name__)

class AKShareDataSource(StockDataSource):
    """基于AKShare的股票数据源实现"""
    
    def __init__(self, config: Dict = None):
        """
        初始化AKShare数据源
        
        参数:
            config: 配置信息
        """
        super().__init__(config)
        self.name = "AKShare"
        
        # 尝试导入AKShare
        try:
            import akshare as ak
            self.ak = ak
            logger.info("AKShare导入成功")
        except ImportError:
            logger.error("AKShare导入失败，请安装: pip install akshare")
            raise ImportError("请安装AKShare: pip install akshare")
    
    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表
        
        返回:
            股票列表DataFrame，包含股票代码和名称
        """
        # 检查缓存
        cache_params = {'source': 'akshare'}
        cached_data = self._load_from_cache('stock_list', cache_params, cache_days=7)  # 股票列表缓存7天
        if cached_data is not None:
            return cached_data
        
        try:
            # 获取股票列表
            stock_info = self.ak.stock_info_a_code_name()
            
            # 重命名列
            stock_info.columns = ['股票代码', '股票名称']
            
            # 缓存数据
            self._save_to_cache(stock_info, 'stock_list', cache_params)
            
            return stock_info
            
        except Exception as e:
            logger.error(f"AKShare获取股票列表失败: {e}")
            return pd.DataFrame(columns=['股票代码', '股票名称'])
    
    def get_daily_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取日线数据
        
        参数:
            stock_code: 股票代码，如"600000"（不带前缀）
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            
        返回:
            包含OHLCV数据的DataFrame
        """
        # 检查缓存
        cache_params = {
            'stock_code': stock_code,
            'start_date': start_date,
            'end_date': end_date
        }
        cached_data = self._load_from_cache('daily_data', cache_params)
        if cached_data is not None:
            return cached_data
        
        try:
            # 处理股票代码，给上海和深圳股票添加前缀
            if stock_code.startswith('6'):
                full_code = f'sh{stock_code}'
            else:
                full_code = f'sz{stock_code}'
            
            # 获取日线数据
            daily_data = self.ak.stock_zh_a_hist(symbol=stock_code, period="daily", 
                                               start_date=start_date, end_date=end_date, 
                                               adjust="qfq")
            
            # 检查数据是否有效
            if daily_data is None or daily_data.empty:
                logger.warning(f"AKShare获取{stock_code}从{start_date}到{end_date}的日线数据为空")
                return pd.DataFrame()
            
            # 标准化列名
            rename_dict = {
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'change_percent',
                '涨跌额': 'change_amount',
                '换手率': 'turnover'
            }
            daily_data = daily_data.rename(columns=rename_dict)
            
            # 设置日期为索引
            daily_data['date'] = pd.to_datetime(daily_data['date'])
            daily_data.set_index('date', inplace=True)
            
            # 缓存数据
            self._save_to_cache(daily_data.reset_index(), 'daily_data', cache_params)
            
            return daily_data
            
        except Exception as e:
            logger.error(f"AKShare获取{stock_code}日线数据失败: {e}")
            return pd.DataFrame()
    
    def get_minute_data(self, stock_code: str, start_date: str, end_date: str, freq: str = '1min') -> pd.DataFrame:
        """
        获取分钟线数据
        
        参数:
            stock_code: 股票代码，如"600000"（不带前缀）
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            freq: 频率，如'1', '5', '15', '30', '60'
            
        返回:
            包含OHLCV数据的DataFrame
        """
        # 检查缓存
        cache_params = {
            'stock_code': stock_code,
            'start_date': start_date,
            'end_date': end_date,
            'freq': freq
        }
        cached_data = self._load_from_cache('minute_data', cache_params)
        if cached_data is not None:
            return cached_data
        
        try:
            # 将freq转换为AKShare支持的格式
            freq_map = {
                '1min': '1', 
                '5min': '5', 
                '15min': '15', 
                '30min': '30', 
                '60min': '60'
            }
            ak_freq = freq_map.get(freq, '1')
            
            # 处理股票代码，给上海和深圳股票添加前缀
            if stock_code.startswith('6'):
                full_code = f'sh{stock_code}'
            else:
                full_code = f'sz{stock_code}'
            
            # 获取分钟线数据
            minute_data = self.ak.stock_zh_a_hist_min_em(symbol=stock_code, period=ak_freq,
                                                       start_date=start_date, end_date=end_date,
                                                       adjust="qfq")
            
            # 检查数据是否有效
            if minute_data is None or minute_data.empty:
                logger.warning(f"AKShare获取{stock_code}从{start_date}到{end_date}的{freq}分钟线数据为空")
                return pd.DataFrame()
            
            # 标准化列名
            rename_dict = {
                '时间': 'datetime',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '最新价': 'last_price'
            }
            minute_data = minute_data.rename(columns=rename_dict)
            
            # 设置日期时间为索引
            minute_data['datetime'] = pd.to_datetime(minute_data['datetime'])
            minute_data.set_index('datetime', inplace=True)
            
            # 缓存数据
            self._save_to_cache(minute_data.reset_index(), 'minute_data', cache_params)
            
            return minute_data
            
        except Exception as e:
            logger.error(f"AKShare获取{stock_code}分钟线数据失败: {e}")
            return pd.DataFrame()
    
    def get_dragon_tiger_list(self, date: str = None) -> pd.DataFrame:
        """
        获取龙虎榜数据
        
        参数:
            date: 日期，格式：YYYY-MM-DD，默认为最近一个交易日
            
        返回:
            龙虎榜数据DataFrame
        """
        if date is None:
            date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # 检查缓存
        cache_params = {'date': date}
        cached_data = self._load_from_cache('dragon_tiger_list', cache_params)
        if cached_data is not None:
            return cached_data
        
        try:
            # 获取龙虎榜数据
            dragon_tiger_data = self.ak.stock_lhb_detail_em(date=date)
            
            # 缓存数据
            self._save_to_cache(dragon_tiger_data, 'dragon_tiger_list', cache_params)
            
            return dragon_tiger_data
            
        except Exception as e:
            logger.error(f"AKShare获取{date}龙虎榜数据失败: {e}")
            return pd.DataFrame()
    
    def get_big_deal(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取大单交易数据
        
        参数:
            stock_code: 股票代码，如"600000"（不带前缀）
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            
        返回:
            大单交易数据DataFrame
        """
        # 检查缓存
        cache_params = {
            'stock_code': stock_code,
            'start_date': start_date,
            'end_date': end_date
        }
        cached_data = self._load_from_cache('big_deal', cache_params)
        if cached_data is not None:
            return cached_data
        
        try:
            # 获取大单交易数据
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            big_deal_data = self.ak.stock_zh_a_tick_tx_js(code=stock_code)
            
            # 过滤日期范围
            big_deal_data['date'] = pd.to_datetime(big_deal_data['date'])
            big_deal_data = big_deal_data[
                (big_deal_data['date'] >= start_date_obj) & 
                (big_deal_data['date'] <= end_date_obj)
            ]
            
            # 缓存数据
            self._save_to_cache(big_deal_data, 'big_deal', cache_params)
            
            return big_deal_data
            
        except Exception as e:
            logger.error(f"AKShare获取{stock_code}大单交易数据失败: {e}")
            return pd.DataFrame()
    
    def get_shareholders_change(self, stock_code: str, quarter: str = None) -> pd.DataFrame:
        """
        获取股东变化数据
        
        参数:
            stock_code: 股票代码，如"600000"（不带前缀）
            quarter: 季度，格式：YYYYQ1, YYYYQ2, YYYYQ3, YYYYQ4，默认为最近一个季度
            
        返回:
            股东变化数据DataFrame
        """
        # 检查缓存
        cache_params = {
            'stock_code': stock_code,
            'quarter': quarter or 'latest'
        }
        cached_data = self._load_from_cache('shareholders_change', cache_params)
        if cached_data is not None:
            return cached_data
        
        try:
            # 获取股东变化数据
            shareholders_data = self.ak.stock_gdfx_free_holding_detail_em(symbol=stock_code)
            
            # 缓存数据
            self._save_to_cache(shareholders_data, 'shareholders_change', cache_params)
            
            return shareholders_data
            
        except Exception as e:
            logger.error(f"AKShare获取{stock_code}股东变化数据失败: {e}")
            return pd.DataFrame()
    
    def get_financial_data(self, stock_code: str, report_type: str = 'income') -> pd.DataFrame:
        """
        获取财务数据
        
        参数:
            stock_code: 股票代码，如"600000"（不带前缀）
            report_type: 报表类型，可选 'income'(利润表), 'balance'(资产负债表), 'cash'(现金流量表)
            
        返回:
            财务数据DataFrame
        """
        # 检查缓存
        cache_params = {
            'stock_code': stock_code,
            'report_type': report_type
        }
        cached_data = self._load_from_cache('financial_data', cache_params, cache_days=30)  # 财务数据缓存30天
        if cached_data is not None:
            return cached_data
        
        try:
            # 根据报表类型选择相应的函数
            if report_type == 'income':
                financial_data = self.ak.stock_financial_report_sina(symbol=stock_code, symbol_type="sh" if stock_code.startswith('6') else "sz")
            elif report_type == 'balance':
                financial_data = self.ak.stock_financial_balance_sheet_by_report_em(symbol=stock_code)
            elif report_type == 'cash':
                financial_data = self.ak.stock_financial_cash_flow_by_report_em(symbol=stock_code)
            else:
                logger.error(f"不支持的报表类型: {report_type}")
                return pd.DataFrame()
            
            # 缓存数据
            self._save_to_cache(financial_data, 'financial_data', cache_params)
            
            return financial_data
            
        except Exception as e:
            logger.error(f"AKShare获取{stock_code}财务数据失败: {e}")
            return pd.DataFrame()
    
    def get_news_sentiment(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取新闻舆情数据
        
        参数:
            stock_code: 股票代码，如"600000"（不带前缀）
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            
        返回:
            新闻舆情数据DataFrame
        """
        # AKShare目前没有直接提供单只股票的新闻舆情数据，返回空DataFrame
        logger.warning("AKShare不支持直接获取单只股票的新闻舆情数据")
        return pd.DataFrame() 