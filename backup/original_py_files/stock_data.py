"""
股票市场数据获取模块

本模块负责从多个数据源获取A股市场数据，包括：
1. 基础行情数据（日线、分钟线）
2. 龙虎榜数据
3. 大单交易数据
4. 股东数据变化
5. 财务数据
6. 新闻舆情数据

支持的数据源：
- AKShare (主要免费数据源)
- Tushare (部分特色数据)
- Baostock (历史数据补充)

日期：2025-05-16
"""

import os
import sys
import time
import logging
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any

# 配置日志
logger = logging.getLogger(__name__)

class StockDataSource:
    """股票数据源基类，定义通用接口"""
    
    def __init__(self, config: Dict = None):
        """
        初始化数据源
        
        参数:
            config: 数据源配置，包含API密钥等信息
        """
        self.config = config or {}
        self.name = "基础数据源"
        self.cache_dir = os.path.join("data", "stock", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"初始化数据源: {self.name}")
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        raise NotImplementedError("子类必须实现此方法")
    
    def get_daily_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取日线数据
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            
        返回:
            包含OHLCV数据的DataFrame
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_minute_data(self, stock_code: str, start_date: str, end_date: str, freq: str = '1min') -> pd.DataFrame:
        """
        获取分钟线数据
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            freq: 频率，如'1min', '5min', '15min', '30min', '60min'
            
        返回:
            包含OHLCV数据的DataFrame
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_dragon_tiger_list(self, date: str = None) -> pd.DataFrame:
        """
        获取龙虎榜数据
        
        参数:
            date: 日期，格式：YYYY-MM-DD，默认为最近一个交易日
            
        返回:
            龙虎榜数据DataFrame
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_big_deal(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取大单交易数据
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            
        返回:
            大单交易数据DataFrame
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_shareholders_change(self, stock_code: str, quarter: str = None) -> pd.DataFrame:
        """
        获取股东变化数据
        
        参数:
            stock_code: 股票代码
            quarter: 季度，格式：YYYYQ1, YYYYQ2, YYYYQ3, YYYYQ4，默认为最近一个季度
            
        返回:
            股东变化数据DataFrame
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_financial_data(self, stock_code: str, report_type: str = 'income') -> pd.DataFrame:
        """
        获取财务数据
        
        参数:
            stock_code: 股票代码
            report_type: 报表类型，可选 'income'(利润表), 'balance'(资产负债表), 'cash'(现金流量表)
            
        返回:
            财务数据DataFrame
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_news_sentiment(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取新闻舆情数据
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            
        返回:
            新闻舆情数据DataFrame
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def _cache_file_path(self, data_type: str, params: Dict) -> str:
        """
        生成缓存文件路径
        
        参数:
            data_type: 数据类型
            params: 请求参数
            
        返回:
            缓存文件路径
        """
        # 将参数转换为字符串作为文件名的一部分
        param_str = "_".join([f"{k}_{v}" for k, v in params.items()])
        return os.path.join(self.cache_dir, f"{data_type}_{param_str}.csv")
    
    def _load_from_cache(self, data_type: str, params: Dict, cache_days: int = 1) -> Optional[pd.DataFrame]:
        """
        从缓存加载数据
        
        参数:
            data_type: 数据类型
            params: 请求参数
            cache_days: 缓存有效天数
            
        返回:
            缓存的DataFrame或None（如果缓存不存在或已过期）
        """
        cache_file = self._cache_file_path(data_type, params)
        if os.path.exists(cache_file):
            # 检查文件修改时间
            file_time = os.path.getmtime(cache_file)
            if (time.time() - file_time) / 86400 < cache_days:  # 转换为天数
                try:
                    return pd.read_csv(cache_file, parse_dates=True)
                except Exception as e:
                    logger.warning(f"读取缓存文件失败: {e}")
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, data_type: str, params: Dict) -> None:
        """
        保存数据到缓存
        
        参数:
            data: 要缓存的DataFrame
            data_type: 数据类型
            params: 请求参数
        """
        if data is None or data.empty:
            return
        
        cache_file = self._cache_file_path(data_type, params)
        try:
            data.to_csv(cache_file, index=False)
            logger.debug(f"数据已缓存到: {cache_file}")
        except Exception as e:
            logger.warning(f"保存缓存文件失败: {e}")


class AKShareDataSource(StockDataSource):
    """基于AKShare的数据源实现"""
    
    def __init__(self, config: Dict = None):
        """
        初始化AKShare数据源
        
        参数:
            config: 配置信息，可选
        """
        super().__init__(config)
        self.name = "AKShare数据源"
        
        try:
            import akshare as ak
            self.ak = ak
            logger.info("AKShare数据源初始化成功")
        except ImportError:
            logger.error("未安装AKShare，请使用pip install akshare安装")
            raise
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取A股股票列表"""
        cache_params = {"type": "stock_list"}
        cached_data = self._load_from_cache("stock_list", cache_params, cache_days=7)  # 股票列表缓存7天
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 获取A股股票基本信息
            stock_info = self.ak.stock_info_a_code_name()
            # 标准化列名
            stock_info.columns = ['股票代码', '股票名称']
            # 缓存数据
            self._save_to_cache(stock_info, "stock_list", cache_params)
            return stock_info
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame(columns=['股票代码', '股票名称'])
    
    def get_daily_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取日线数据"""
        cache_params = {
            "stock_code": stock_code,
            "start_date": start_date,
            "end_date": end_date
        }
        cached_data = self._load_from_cache("daily", cache_params)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 确保股票代码格式正确（带市场前缀）
            if not stock_code.startswith(('sh', 'sz')):
                if stock_code.startswith('6'):
                    stock_code = f"sh{stock_code}"
                else:
                    stock_code = f"sz{stock_code}"
            
            # 获取日线数据
            daily_data = self.ak.stock_zh_a_hist(symbol=stock_code, period="daily", 
                                               start_date=start_date, end_date=end_date, adjust="qfq")
            
            # 标准化列名
            daily_data.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
            
            # 设置日期为索引
            daily_data['日期'] = pd.to_datetime(daily_data['日期'])
            
            # 缓存数据
            self._save_to_cache(daily_data, "daily", cache_params)
            
            return daily_data
        except Exception as e:
            logger.error(f"获取日线数据失败 - 股票: {stock_code}, 错误: {e}")
            return pd.DataFrame(columns=['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率'])
    
    def get_minute_data(self, stock_code: str, start_date: str, end_date: str, freq: str = '1min') -> pd.DataFrame:
        """获取分钟线数据"""
        # 转换频率格式
        period_map = {
            '1min': '1', '5min': '5', '15min': '15', '30min': '30', '60min': '60'
        }
        period = period_map.get(freq, '1')
        
        cache_params = {
            "stock_code": stock_code,
            "start_date": start_date,
            "end_date": end_date,
            "freq": freq
        }
        cached_data = self._load_from_cache("minute", cache_params)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 确保股票代码格式正确
            if not stock_code.startswith(('sh', 'sz')):
                if stock_code.startswith('6'):
                    stock_code = f"sh{stock_code}"
                else:
                    stock_code = f"sz{stock_code}"
            
            # 获取分钟线数据
            minute_data = self.ak.stock_zh_a_hist_min_em(symbol=stock_code, period=period, 
                                                      start_date=start_date, end_date=end_date, adjust="qfq")
            
            # 标准化列名
            minute_data.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '最新价']
            
            # 设置日期为索引
            minute_data['日期'] = pd.to_datetime(minute_data['日期'])
            
            # 缓存数据
            self._save_to_cache(minute_data, "minute", cache_params)
            
            return minute_data
        except Exception as e:
            logger.error(f"获取分钟线数据失败 - 股票: {stock_code}, 频率: {freq}, 错误: {e}")
            return pd.DataFrame(columns=['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '最新价'])
    
    def get_dragon_tiger_list(self, date: str = None) -> pd.DataFrame:
        """获取龙虎榜数据"""
        # 如果未指定日期，使用最近一个交易日
        if date is None:
            date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        cache_params = {"date": date}
        cached_data = self._load_from_cache("dragon_tiger", cache_params)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 获取龙虎榜数据
            dt_data = self.ak.stock_lhb_detail_em(date=date)
            
            # 缓存数据
            self._save_to_cache(dt_data, "dragon_tiger", cache_params)
            
            return dt_data
        except Exception as e:
            logger.error(f"获取龙虎榜数据失败 - 日期: {date}, 错误: {e}")
            return pd.DataFrame()
    
    def get_big_deal(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取大单交易数据"""
        cache_params = {
            "stock_code": stock_code,
            "start_date": start_date,
            "end_date": end_date
        }
        cached_data = self._load_from_cache("big_deal", cache_params)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 确保股票代码格式正确
            if not stock_code.startswith(('sh', 'sz')):
                if stock_code.startswith('6'):
                    stock_code = f"sh{stock_code}"
                else:
                    stock_code = f"sz{stock_code}"
            
            # 获取大单数据
            big_deal_data = self.ak.stock_zh_a_tick_tx_js(code=stock_code)
            
            # 过滤日期范围
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            big_deal_data['成交时间'] = pd.to_datetime(big_deal_data['成交时间'])
            big_deal_data = big_deal_data[(big_deal_data['成交时间'] >= start) & (big_deal_data['成交时间'] <= end)]
            
            # 缓存数据
            self._save_to_cache(big_deal_data, "big_deal", cache_params)
            
            return big_deal_data
        except Exception as e:
            logger.error(f"获取大单交易数据失败 - 股票: {stock_code}, 错误: {e}")
            return pd.DataFrame()
    
    def get_shareholders_change(self, stock_code: str, quarter: str = None) -> pd.DataFrame:
        """获取股东变化数据"""
        # 如果未指定季度，使用最近一个季度
        if quarter is None:
            now = datetime.datetime.now()
            year = now.year
            month = now.month
            q = (month - 1) // 3 + 1
            quarter = f"{year}Q{q}"
        
        cache_params = {
            "stock_code": stock_code,
            "quarter": quarter
        }
        cached_data = self._load_from_cache("shareholders", cache_params, cache_days=30)  # 股东数据缓存30天
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 获取十大股东数据
            shareholders_data = self.ak.stock_gdfx_top_10_em(symbol=stock_code)
            
            # 缓存数据
            self._save_to_cache(shareholders_data, "shareholders", cache_params)
            
            return shareholders_data
        except Exception as e:
            logger.error(f"获取股东变化数据失败 - 股票: {stock_code}, 季度: {quarter}, 错误: {e}")
            return pd.DataFrame()
    
    def get_financial_data(self, stock_code: str, report_type: str = 'income') -> pd.DataFrame:
        """获取财务数据"""
        cache_params = {
            "stock_code": stock_code,
            "report_type": report_type
        }
        cached_data = self._load_from_cache("financial", cache_params, cache_days=30)  # 财务数据缓存30天
        
        if cached_data is not None:
            return cached_data
        
        try:
            financial_data = None
            
            # 根据报表类型获取不同的财务数据
            if report_type == 'income':
                # 利润表
                financial_data = self.ak.stock_financial_report_sina(stock=stock_code, symbol="利润表")
            elif report_type == 'balance':
                # 资产负债表
                financial_data = self.ak.stock_financial_report_sina(stock=stock_code, symbol="资产负债表")
            elif report_type == 'cash':
                # 现金流量表
                financial_data = self.ak.stock_financial_report_sina(stock=stock_code, symbol="现金流量表")
            
            # 缓存数据
            if financial_data is not None:
                self._save_to_cache(financial_data, "financial", cache_params)
            
            return financial_data
        except Exception as e:
            logger.error(f"获取财务数据失败 - 股票: {stock_code}, 报表类型: {report_type}, 错误: {e}")
            return pd.DataFrame()
    
    def get_news_sentiment(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取新闻舆情数据"""
        # 注意：AKShare可能没有直接提供新闻舆情数据，这里提供一个简化实现
        # 实际项目中可能需要使用其他数据源或自行爬取
        
        cache_params = {
            "stock_code": stock_code,
            "start_date": start_date,
            "end_date": end_date
        }
        cached_data = self._load_from_cache("news", cache_params)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 获取股票相关新闻
            # 注意：这里使用的是股票公告数据作为替代，实际项目中应使用真实的新闻舆情数据
            news_data = self.ak.stock_notice_report(symbol=stock_code)
            
            # 过滤日期范围
            if 'date' in news_data.columns:
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                news_data['date'] = pd.to_datetime(news_data['date'])
                news_data = news_data[(news_data['date'] >= start) & (news_data['date'] <= end)]
            
            # 缓存数据
            self._save_to_cache(news_data, "news", cache_params)
            
            return news_data
        except Exception as e:
            logger.error(f"获取新闻舆情数据失败 - 股票: {stock_code}, 错误: {e}")
            return pd.DataFrame()


class TushareDataSource(StockDataSource):
    """基于Tushare的数据源实现"""
    
    def __init__(self, config: Dict = None):
        """
        初始化Tushare数据源
        
        参数:
            config: 配置信息，必须包含api_key
        """
        super().__init__(config)
        self.name = "Tushare数据源"
        
        # 检查配置中是否包含API密钥
        if not config or 'api_key' not in config:
            logger.warning("未提供Tushare API密钥，部分功能可能受限")
        
        try:
            import tushare as ts
            self.ts = ts
            
            # 设置token
            if config and 'api_key' in config:
                self.ts.set_token(config['api_key'])
                self.pro = self.ts.pro_api()
                logger.info("Tushare数据源初始化成功")
            else:
                self.pro = None
                logger.warning("Tushare未设置token，仅能使用基础功能")
        except ImportError:
            logger.error("未安装Tushare，请使用pip install tushare安装")
            raise
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取A股股票列表"""
        cache_params = {"type": "stock_list"}
        cached_data = self._load_from_cache("stock_list", cache_params, cache_days=7)  # 股票列表缓存7天
        
        if cached_data is not None:
            return cached_data
        
        try:
            if self.pro:
                # 使用pro接口获取股票列表
                stock_info = self.pro.stock_basic(exchange='', list_status='L', 
                                                fields='ts_code,symbol,name,area,industry,list_date')
                # 标准化列名
                stock_info.columns = ['TS代码', '股票代码', '股票名称', '地区', '行业', '上市日期']
            else:
                # 使用基础接口获取股票列表
                stock_info = self.ts.get_stock_basics()
                stock_info.reset_index(inplace=True)
                stock_info.columns = ['股票代码', '股票名称', '行业', '地区', '市盈率', '流通股本', '总股本', '总资产', '流动资产', '固定资产', '公积金', '每股公积金', '每股收益', '每股净资产', '市净率', '上市日期']
            
            # 缓存数据
            self._save_to_cache(stock_info, "stock_list", cache_params)
            
            return stock_info
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame(columns=['股票代码', '股票名称'])
    
    def get_daily_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取日线数据"""
        cache_params = {
            "stock_code": stock_code,
            "start_date": start_date,
            "end_date": end_date
        }
        cached_data = self._load_from_cache("daily", cache_params)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 确保股票代码格式正确
            if self.pro:
                # 转换为Tushare格式的代码
                if stock_code.startswith('6'):
                    ts_code = f"{stock_code}.SH"
                else:
                    ts_code = f"{stock_code}.SZ"
                
                # 使用pro接口获取日线数据
                daily_data = self.pro.daily(ts_code=ts_code, start_date=start_date.replace('-', ''), 
                                          end_date=end_date.replace('-', ''))
                
                # 获取复权因子
                adj_factor = self.pro.adj_factor(ts_code=ts_code, start_date=start_date.replace('-', ''), 
                                               end_date=end_date.replace('-', ''))
                
                # 合并数据并计算前复权价格
                if not adj_factor.empty:
                    daily_data = pd.merge(daily_data, adj_factor, on=['ts_code', 'trade_date'])
                    for price_col in ['open', 'high', 'low', 'close']:
                        daily_data[f"{price_col}_qfq"] = daily_data[price_col] * daily_data['adj_factor']
                    
                    # 更新列名
                    daily_data.rename(columns={
                        'trade_date': '日期',
                        'open_qfq': '开盘',
                        'close_qfq': '收盘',
                        'high_qfq': '最高',
                        'low_qfq': '最低',
                        'vol': '成交量',
                        'amount': '成交额'
                    }, inplace=True)
                else:
                    # 如果没有复权因子，使用原始价格
                    daily_data.rename(columns={
                        'trade_date': '日期',
                        'open': '开盘',
                        'close': '收盘',
                        'high': '最高',
                        'low': '最低',
                        'vol': '成交量',
                        'amount': '成交额'
                    }, inplace=True)
            else:
                # 使用基础接口获取日线数据
                daily_data = self.ts.get_hist_data(code=stock_code, start=start_date, end=end_date)
                daily_data.reset_index(inplace=True)
                daily_data.rename(columns={
                    'date': '日期',
                    'open': '开盘',
                    'close': '收盘',
                    'high': '最高',
                    'low': '最低',
                    'volume': '成交量',
                    'amount': '成交额'
                }, inplace=True)
            
            # 设置日期为索引
            daily_data['日期'] = pd.to_datetime(daily_data['日期'])
            
            # 缓存数据
            self._save_to_cache(daily_data, "daily", cache_params)
            
            return daily_data
        except Exception as e:
            logger.error(f"获取日线数据失败 - 股票: {stock_code}, 错误: {e}")
            return pd.DataFrame(columns=['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额'])
    
    def get_minute_data(self, stock_code: str, start_date: str, end_date: str, freq: str = '1min') -> pd.DataFrame:
        """获取分钟线数据"""
        # Tushare的分钟线数据可能需要较高的积分，这里提供一个简化实现
        cache_params = {
            "stock_code": stock_code,
            "start_date": start_date,
            "end_date": end_date,
            "freq": freq
        }
        cached_data = self._load_from_cache("minute", cache_params)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 转换频率格式
            freq_map = {
                '1min': '1min', '5min': '5min', '15min': '15min', '30min': '30min', '60min': '60min'
            }
            ts_freq = freq_map.get(freq, '1min')
            
            # 使用基础接口获取分钟线数据
            minute_data = self.ts.get_hist_data(code=stock_code, start=start_date, end=end_date, ktype=ts_freq)
            
            if minute_data is not None and not minute_data.empty:
                minute_data.reset_index(inplace=True)
                minute_data.rename(columns={
                    'date': '日期',
                    'open': '开盘',
                    'close': '收盘',
                    'high': '最高',
                    'low': '最低',
                    'volume': '成交量',
                    'amount': '成交额'
                }, inplace=True)
                
                # 设置日期为索引
                minute_data['日期'] = pd.to_datetime(minute_data['日期'])
                
                # 缓存数据
                self._save_to_cache(minute_data, "minute", cache_params)
                
                return minute_data
            else:
                logger.warning(f"获取分钟线数据为空 - 股票: {stock_code}, 频率: {freq}")
                return pd.DataFrame(columns=['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额'])
        except Exception as e:
            logger.error(f"获取分钟线数据失败 - 股票: {stock_code}, 频率: {freq}, 错误: {e}")
            return pd.DataFrame(columns=['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额'])
    
    def get_dragon_tiger_list(self, date: str = None) -> pd.DataFrame:
        """获取龙虎榜数据"""
        # 如果未指定日期，使用最近一个交易日
        if date is None:
            date = datetime.datetime.now().strftime('%Y%m%d')
        else:
            date = date.replace('-', '')
        
        cache_params = {"date": date}
        cached_data = self._load_from_cache("dragon_tiger", cache_params)
        
        if cached_data is not None:
            return cached_data
        
        try:
            if self.pro:
                # 使用pro接口获取龙虎榜数据
                dt_data = self.pro.top_list(trade_date=date)
                
                # 缓存数据
                self._save_to_cache(dt_data, "dragon_tiger", cache_params)
                
                return dt_data
            else:
                logger.warning("未设置Tushare token，无法获取龙虎榜数据")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取龙虎榜数据失败 - 日期: {date}, 错误: {e}")
            return pd.DataFrame()
    
    def get_big_deal(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取大单交易数据"""
        # Tushare的大单数据可能需要较高的积分，这里提供一个简化实现
        cache_params = {
            "stock_code": stock_code,
            "start_date": start_date.replace('-', ''),
            "end_date": end_date.replace('-', '')
        }
        cached_data = self._load_from_cache("big_deal", cache_params)
        
        if cached_data is not None:
            return cached_data
        
        try:
            if self.pro:
                # 转换为Tushare格式的代码
                if stock_code.startswith('6'):
                    ts_code = f"{stock_code}.SH"
                else:
                    ts_code = f"{stock_code}.SZ"
                
                # 使用pro接口获取大单数据
                big_deal_data = self.pro.moneyflow(ts_code=ts_code, 
                                                 start_date=start_date.replace('-', ''), 
                                                 end_date=end_date.replace('-', ''))
                
                # 缓存数据
                self._save_to_cache(big_deal_data, "big_deal", cache_params)
                
                return big_deal_data
            else:
                logger.warning("未设置Tushare token，无法获取大单交易数据")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取大单交易数据失败 - 股票: {stock_code}, 错误: {e}")
            return pd.DataFrame()
    
    def get_shareholders_change(self, stock_code: str, quarter: str = None) -> pd.DataFrame:
        """获取股东变化数据"""
        # 如果未指定季度，使用最近一个季度
        if quarter is None:
            now = datetime.datetime.now()
            year = now.year
            month = now.month
            q = (month - 1) // 3 + 1
            quarter = f"{year}Q{q}"
        
        # 转换为Tushare格式的日期
        year = int(quarter[:4])
        q = int(quarter[-1])
        period = f"{year}{q * 3:02d}31"
        
        cache_params = {
            "stock_code": stock_code,
            "period": period
        }
        cached_data = self._load_from_cache("shareholders", cache_params, cache_days=30)  # 股东数据缓存30天
        
        if cached_data is not None:
            return cached_data
        
        try:
            if self.pro:
                # 转换为Tushare格式的代码
                if stock_code.startswith('6'):
                    ts_code = f"{stock_code}.SH"
                else:
                    ts_code = f"{stock_code}.SZ"
                
                # 使用pro接口获取十大股东数据
                shareholders_data = self.pro.top10_holders(ts_code=ts_code, period=period)
                
                # 缓存数据
                self._save_to_cache(shareholders_data, "shareholders", cache_params)
                
                return shareholders_data
            else:
                logger.warning("未设置Tushare token，无法获取股东变化数据")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取股东变化数据失败 - 股票: {stock_code}, 季度: {quarter}, 错误: {e}")
            return pd.DataFrame()
    
    def get_financial_data(self, stock_code: str, report_type: str = 'income') -> pd.DataFrame:
        """获取财务数据"""
        cache_params = {
            "stock_code": stock_code,
            "report_type": report_type
        }
        cached_data = self._load_from_cache("financial", cache_params, cache_days=30)  # 财务数据缓存30天
        
        if cached_data is not None:
            return cached_data
        
        try:
            if self.pro:
                # 转换为Tushare格式的代码
                if stock_code.startswith('6'):
                    ts_code = f"{stock_code}.SH"
                else:
                    ts_code = f"{stock_code}.SZ"
                
                financial_data = None
                
                # 根据报表类型获取不同的财务数据
                if report_type == 'income':
                    # 利润表
                    financial_data = self.pro.income(ts_code=ts_code)
                elif report_type == 'balance':
                    # 资产负债表
                    financial_data = self.pro.balancesheet(ts_code=ts_code)
                elif report_type == 'cash':
                    # 现金流量表
                    financial_data = self.pro.cashflow(ts_code=ts_code)
                
                # 缓存数据
                if financial_data is not None:
                    self._save_to_cache(financial_data, "financial", cache_params)
                
                return financial_data
            else:
                logger.warning("未设置Tushare token，无法获取财务数据")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取财务数据失败 - 股票: {stock_code}, 报表类型: {report_type}, 错误: {e}")
            return pd.DataFrame()
    
    def get_news_sentiment(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取新闻舆情数据"""
        # Tushare可能没有直接提供新闻舆情数据，这里提供一个简化实现
        cache_params = {
            "stock_code": stock_code,
            "start_date": start_date.replace('-', ''),
            "end_date": end_date.replace('-', '')
        }
        cached_data = self._load_from_cache("news", cache_params)
        
        if cached_data is not None:
            return cached_data
        
        try:
            if self.pro:
                # 转换为Tushare格式的代码
                if stock_code.startswith('6'):
                    ts_code = f"{stock_code}.SH"
                else:
                    ts_code = f"{stock_code}.SZ"
                
                # 使用公告数据作为替代
                news_data = self.pro.anns(ts_code=ts_code, 
                                        start_date=start_date.replace('-', ''), 
                                        end_date=end_date.replace('-', ''))
                
                # 缓存数据
                self._save_to_cache(news_data, "news", cache_params)
                
                return news_data
            else:
                logger.warning("未设置Tushare token，无法获取新闻舆情数据")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取新闻舆情数据失败 - 股票: {stock_code}, 错误: {e}")
            return pd.DataFrame()


class BaostockDataSource(StockDataSource):
    """基于Baostock的数据源实现"""
    
    def __init__(self, config: Dict = None):
        """
        初始化Baostock数据源
        
        参数:
            config: 配置信息，可选
        """
        super().__init__(config)
        self.name = "Baostock数据源"
        
        try:
            import baostock as bs
            self.bs = bs
            # 登录系统
            self.login_result = self.bs.login()
            if self.login_result.error_code != '0':
                logger.error(f"Baostock登录失败: {self.login_result.error_msg}")
            else:
                logger.info("Baostock数据源初始化成功")
        except ImportError:
            logger.error("未安装Baostock，请使用pip install baostock安装")
            raise
    
    def __del__(self):
        """析构函数，确保退出登录"""
        try:
            self.bs.logout()
        except:
            pass
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取A股股票列表"""
        cache_params = {"type": "stock_list"}
        cached_data = self._load_from_cache("stock_list", cache_params, cache_days=7)  # 股票列表缓存7天
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 获取证券基本资料
            rs = self.bs.query_stock_basic()
            if rs.error_code != '0':
                logger.error(f"获取股票列表失败: {rs.error_msg}")
                return pd.DataFrame(columns=['股票代码', '股票名称'])
            
            # 处理返回的数据
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            stock_info = pd.DataFrame(data_list, columns=rs.fields)
            
            # 标准化列名
            stock_info.rename(columns={
                'code': '股票代码',
                'code_name': '股票名称'
            }, inplace=True)
            
            # 缓存数据
            self._save_to_cache(stock_info, "stock_list", cache_params)
            
            return stock_info
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame(columns=['股票代码', '股票名称'])
    
    def get_daily_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取日线数据"""
        cache_params = {
            "stock_code": stock_code,
            "start_date": start_date,
            "end_date": end_date
        }
        cached_data = self._load_from_cache("daily", cache_params)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 确保股票代码格式正确
            if not stock_code.startswith(('sh', 'sz')):
                if stock_code.startswith('6'):
                    bs_code = f"sh.{stock_code}"
                else:
                    bs_code = f"sz.{stock_code}"
            else:
                bs_code = f"{stock_code[:2]}.{stock_code[2:]}"
            
            # 获取日线数据
            rs = self.bs.query_history_k_data_plus(
                code=bs_code,
                fields="date,open,high,low,close,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2"  # 前复权
            )
            
            if rs.error_code != '0':
                logger.error(f"获取日线数据失败: {rs.error_msg}")
                return pd.DataFrame(columns=['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额'])
            
            # 处理返回的数据
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            daily_data = pd.DataFrame(data_list, columns=rs.fields)
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                if col in daily_data.columns:
                    daily_data[col] = pd.to_numeric(daily_data[col], errors='coerce')
            
            # 标准化列名
            daily_data.rename(columns={
                'date': '日期',
                'open': '开盘',
                'close': '收盘',
                'high': '最高',
                'low': '最低',
                'volume': '成交量',
                'amount': '成交额'
            }, inplace=True)
            
            # 设置日期为索引
            daily_data['日期'] = pd.to_datetime(daily_data['日期'])
            
            # 缓存数据
            self._save_to_cache(daily_data, "daily", cache_params)
            
            return daily_data
        except Exception as e:
            logger.error(f"获取日线数据失败 - 股票: {stock_code}, 错误: {e}")
            return pd.DataFrame(columns=['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额'])
    
    def get_minute_data(self, stock_code: str, start_date: str, end_date: str, freq: str = '1min') -> pd.DataFrame:
        """获取分钟线数据"""
        # 转换频率格式
        freq_map = {
            '1min': '1', '5min': '5', '15min': '15', '30min': '30', '60min': '60'
        }
        bs_freq = freq_map.get(freq, '1')
        
        cache_params = {
            "stock_code": stock_code,
            "start_date": start_date,
            "end_date": end_date,
            "freq": freq
        }
        cached_data = self._load_from_cache("minute", cache_params)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 确保股票代码格式正确
            if not stock_code.startswith(('sh', 'sz')):
                if stock_code.startswith('6'):
                    bs_code = f"sh.{stock_code}"
                else:
                    bs_code = f"sz.{stock_code}"
            else:
                bs_code = f"{stock_code[:2]}.{stock_code[2:]}"
            
            # 获取分钟线数据
            rs = self.bs.query_history_k_data_plus(
                code=bs_code,
                fields="date,time,open,high,low,close,volume,amount,adjustflag",
                start_date=start_date,
                end_date=end_date,
                frequency=f"{bs_freq}",
                adjustflag="2"  # 前复权
            )
            
            if rs.error_code != '0':
                logger.error(f"获取分钟线数据失败: {rs.error_msg}")
                return pd.DataFrame(columns=['日期', '时间', '开盘', '收盘', '最高', '最低', '成交量', '成交额'])
            
            # 处理返回的数据
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            minute_data = pd.DataFrame(data_list, columns=rs.fields)
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                if col in minute_data.columns:
                    minute_data[col] = pd.to_numeric(minute_data[col], errors='coerce')
            
            # 标准化列名
            minute_data.rename(columns={
                'date': '日期',
                'time': '时间',
                'open': '开盘',
                'close': '收盘',
                'high': '最高',
                'low': '最低',
                'volume': '成交量',
                'amount': '成交额'
            }, inplace=True)
            
            # 合并日期和时间
            minute_data['日期时间'] = pd.to_datetime(minute_data['日期'] + ' ' + minute_data['时间'])
            
            # 缓存数据
            self._save_to_cache(minute_data, "minute", cache_params)
            
            return minute_data
        except Exception as e:
            logger.error(f"获取分钟线数据失败 - 股票: {stock_code}, 频率: {freq}, 错误: {e}")
            return pd.DataFrame(columns=['日期', '时间', '开盘', '收盘', '最高', '最低', '成交量', '成交额'])
    
    # 以下方法Baostock可能不直接支持，返回空DataFrame或尝试使用其他方法替代
    
    def get_dragon_tiger_list(self, date: str = None) -> pd.DataFrame:
        """获取龙虎榜数据 - Baostock不直接支持"""
        logger.warning("Baostock不支持获取龙虎榜数据")
        return pd.DataFrame()
    
    def get_big_deal(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取大单交易数据 - Baostock不直接支持"""
        logger.warning("Baostock不支持获取大单交易数据")
        return pd.DataFrame()
    
    def get_shareholders_change(self, stock_code: str, quarter: str = None) -> pd.DataFrame:
        """获取股东变化数据 - Baostock不直接支持"""
        logger.warning("Baostock不支持获取股东变化数据")
        return pd.DataFrame()
    
    def get_financial_data(self, stock_code: str, report_type: str = 'income') -> pd.DataFrame:
        """获取财务数据"""
        cache_params = {
            "stock_code": stock_code,
            "report_type": report_type
        }
        cached_data = self._load_from_cache("financial", cache_params, cache_days=30)  # 财务数据缓存30天
        
        if cached_data is not None:
            return cached_data
        
        try:
            # 确保股票代码格式正确
            if not stock_code.startswith(('sh', 'sz')):
                if stock_code.startswith('6'):
                    bs_code = f"sh.{stock_code}"
                else:
                    bs_code = f"sz.{stock_code}"
            else:
                bs_code = f"{stock_code[:2]}.{stock_code[2:]}"
            
            financial_data = None
            
            # 根据报表类型获取不同的财务数据
            if report_type == 'income':
                # 利润表
                rs = self.bs.query_profit_data(code=bs_code)
                if rs.error_code != '0':
                    logger.error(f"获取利润表数据失败: {rs.error_msg}")
                    return pd.DataFrame()
                
                # 处理返回的数据
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                
                financial_data = pd.DataFrame(data_list, columns=rs.fields)
            elif report_type == 'balance':
                # 资产负债表 - Baostock可能不直接支持
                logger.warning("Baostock不直接支持获取资产负债表数据")
                return pd.DataFrame()
            elif report_type == 'cash':
                # 现金流量表 - Baostock可能不直接支持
                logger.warning("Baostock不直接支持获取现金流量表数据")
                return pd.DataFrame()
            
            # 缓存数据
            if financial_data is not None:
                self._save_to_cache(financial_data, "financial", cache_params)
            
            return financial_data
        except Exception as e:
            logger.error(f"获取财务数据失败 - 股票: {stock_code}, 报表类型: {report_type}, 错误: {e}")
            return pd.DataFrame()
    
    def get_news_sentiment(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取新闻舆情数据 - Baostock不直接支持"""
        logger.warning("Baostock不支持获取新闻舆情数据")
        return pd.DataFrame()


class StockDataManager:
    """股票数据管理器，整合多个数据源"""
    
    def __init__(self, config: Dict = None):
        """
        初始化数据管理器
        
        参数:
            config: 配置信息，包含各数据源的配置
        """
        self.config = config or {}
        self.data_sources = {}
        self.default_source = None
        
        # 初始化日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化数据源
        self._init_data_sources()
    
    def _init_data_sources(self):
        """初始化所有配置的数据源"""
        # 尝试初始化AKShare数据源
        try:
            ak_config = self.config.get('akshare', {})
            self.data_sources['akshare'] = AKShareDataSource(ak_config)
            if self.default_source is None:
                self.default_source = 'akshare'
            self.logger.info("AKShare数据源初始化成功")
        except Exception as e:
            self.logger.warning(f"AKShare数据源初始化失败: {e}")
        
        # 尝试初始化Tushare数据源
        try:
            ts_config = self.config.get('tushare', {})
            if ts_config and 'api_key' in ts_config:
                self.data_sources['tushare'] = TushareDataSource(ts_config)
                if self.default_source is None:
                    self.default_source = 'tushare'
                self.logger.info("Tushare数据源初始化成功")
        except Exception as e:
            self.logger.warning(f"Tushare数据源初始化失败: {e}")
        
        # 尝试初始化Baostock数据源
        try:
            bs_config = self.config.get('baostock', {})
            self.data_sources['baostock'] = BaostockDataSource(bs_config)
            if self.default_source is None:
                self.default_source = 'baostock'
            self.logger.info("Baostock数据源初始化成功")
        except Exception as e:
            self.logger.warning(f"Baostock数据源初始化失败: {e}")
        
        if not self.data_sources:
            self.logger.error("没有可用的数据源，请检查配置和依赖安装")
            raise ValueError("没有可用的数据源")
    
    def get_data_source(self, source_name: str = None) -> StockDataSource:
        """
        获取指定的数据源
        
        参数:
            source_name: 数据源名称，如果为None则返回默认数据源
            
        返回:
            数据源对象
        """
        if source_name is None:
            source_name = self.default_source
        
        if source_name not in self.data_sources:
            self.logger.warning(f"数据源 {source_name} 不存在，使用默认数据源 {self.default_source}")
            source_name = self.default_source
        
        return self.data_sources[source_name]
    
    def get_stock_list(self, source_name: str = None) -> pd.DataFrame:
        """
        获取股票列表
        
        参数:
            source_name: 数据源名称，如果为None则使用默认数据源
            
        返回:
            股票列表DataFrame
        """
        data_source = self.get_data_source(source_name)
        return data_source.get_stock_list()
    
    def get_daily_data(self, stock_code: str, start_date: str, end_date: str, source_name: str = None) -> pd.DataFrame:
        """
        获取日线数据
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            source_name: 数据源名称，如果为None则使用默认数据源
            
        返回:
            日线数据DataFrame
        """
        data_source = self.get_data_source(source_name)
        return data_source.get_daily_data(stock_code, start_date, end_date)
    
    def get_minute_data(self, stock_code: str, start_date: str, end_date: str, freq: str = '1min', source_name: str = None) -> pd.DataFrame:
        """
        获取分钟线数据
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            freq: 频率，如'1min', '5min', '15min', '30min', '60min'
            source_name: 数据源名称，如果为None则使用默认数据源
            
        返回:
            分钟线数据DataFrame
        """
        data_source = self.get_data_source(source_name)
        return data_source.get_minute_data(stock_code, start_date, end_date, freq)
    
    def get_dragon_tiger_list(self, date: str = None, source_name: str = None) -> pd.DataFrame:
        """
        获取龙虎榜数据
        
        参数:
            date: 日期，格式：YYYY-MM-DD，默认为最近一个交易日
            source_name: 数据源名称，如果为None则使用默认数据源
            
        返回:
            龙虎榜数据DataFrame
        """
        # 龙虎榜数据优先使用Tushare或AKShare
        if source_name is None:
            if 'tushare' in self.data_sources:
                source_name = 'tushare'
            elif 'akshare' in self.data_sources:
                source_name = 'akshare'
        
        data_source = self.get_data_source(source_name)
        return data_source.get_dragon_tiger_list(date)
    
    def get_big_deal(self, stock_code: str, start_date: str, end_date: str, source_name: str = None) -> pd.DataFrame:
        """
        获取大单交易数据
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            source_name: 数据源名称，如果为None则使用默认数据源
            
        返回:
            大单交易数据DataFrame
        """
        data_source = self.get_data_source(source_name)
        return data_source.get_big_deal(stock_code, start_date, end_date)
    
    def get_shareholders_change(self, stock_code: str, quarter: str = None, source_name: str = None) -> pd.DataFrame:
        """
        获取股东变化数据
        
        参数:
            stock_code: 股票代码
            quarter: 季度，格式：YYYYQ1, YYYYQ2, YYYYQ3, YYYYQ4，默认为最近一个季度
            source_name: 数据源名称，如果为None则使用默认数据源
            
        返回:
            股东变化数据DataFrame
        """
        data_source = self.get_data_source(source_name)
        return data_source.get_shareholders_change(stock_code, quarter)
    
    def get_financial_data(self, stock_code: str, report_type: str = 'income', source_name: str = None) -> pd.DataFrame:
        """
        获取财务数据
        
        参数:
            stock_code: 股票代码
            report_type: 报表类型，可选 'income'(利润表), 'balance'(资产负债表), 'cash'(现金流量表)
            source_name: 数据源名称，如果为None则使用默认数据源
            
        返回:
            财务数据DataFrame
        """
        data_source = self.get_data_source(source_name)
        return data_source.get_financial_data(stock_code, report_type)
    
    def get_news_sentiment(self, stock_code: str, start_date: str, end_date: str, source_name: str = None) -> pd.DataFrame:
        """
        获取新闻舆情数据
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            source_name: 数据源名称，如果为None则使用默认数据源
            
        返回:
            新闻舆情数据DataFrame
        """
        data_source = self.get_data_source(source_name)
        return data_source.get_news_sentiment(stock_code, start_date, end_date)
    
    def get_available_data_sources(self) -> List[str]:
        """
        获取所有可用的数据源名称
        
        返回:
            数据源名称列表
        """
        return list(self.data_sources.keys())
    
    def set_default_data_source(self, source_name: str) -> bool:
        """
        设置默认数据源
        
        参数:
            source_name: 数据源名称
            
        返回:
            是否设置成功
        """
        if source_name in self.data_sources:
            self.default_source = source_name
            self.logger.info(f"默认数据源已设置为: {source_name}")
            return True
        else:
            self.logger.warning(f"数据源 {source_name} 不存在，默认数据源未更改")
            return False


# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 测试数据管理器
    config = {
        'akshare': {},
        'tushare': {'api_key': '你的Tushare API密钥'},  # 替换为实际的API密钥
        'baostock': {}
    }
    
    try:
        # 初始化数据管理器
        data_manager = StockDataManager(config)
        
        # 获取股票列表
        stock_list = data_manager.get_stock_list()
        print(f"股票列表前5行:\n{stock_list.head()}")
        
        # 获取日线数据
        daily_data = data_manager.get_daily_data('000001', '2023-01-01', '2023-01-31')
        print(f"\n日线数据前5行:\n{daily_data.head()}")
        
        # 获取分钟线数据
        minute_data = data_manager.get_minute_data('000001', '2023-01-01', '2023-01-05', freq='5min')
        print(f"\n分钟线数据前5行:\n{minute_data.head()}")
        
        # 获取龙虎榜数据
        dt_data = data_manager.get_dragon_tiger_list('2023-01-03')
        print(f"\n龙虎榜数据前5行:\n{dt_data.head()}")
        
    except Exception as e:
        logging.error(f"测试过程中发生错误: {e}")
