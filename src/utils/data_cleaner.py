"""
数据清洗与预处理模块。
本模块提供了各种数据清洗和预处理方法，支持一键式数据清洗和多种清洗选项。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import datetime
import logging

# 配置日志
logger = logging.getLogger(__name__)

class 数据清洗器:
    """
    数据清洗器类，提供各种数据清洗和预处理方法。
    支持一键式数据清洗和多种清洗选项。
    """
    
    def __init__(self, 配置: Dict[str, Any] = None):
        """
        初始化数据清洗器。
        
        参数:
            配置: 数据清洗器配置字典，可以包含默认的清洗选项
        """
        self.配置 = 配置 if 配置 is not None else {}
        self.默认清洗选项 = self.配置.get('默认清洗选项', {
            '缺失值处理': '前向填充',
            '异常值处理': '无',
            '标准化': '无',
            '去重': True,
            '排序': True
        })
    
    def 清洗数据(self, 数据: pd.DataFrame, 选项: Dict[str, Any] = None) -> pd.DataFrame:
        """
        清洗数据，应用指定的清洗选项。
        
        参数:
            数据: 要清洗的数据
            选项: 清洗选项字典，如果为None则使用默认选项
            
        返回:
            清洗后的数据
        """
        if 数据 is None or 数据.empty:
            logger.warning("输入数据为空，无法进行清洗")
            return pd.DataFrame()
        
        # 使用默认选项或指定选项
        选项 = 选项 if 选项 is not None else self.默认清洗选项
        
        # 创建数据副本，避免修改原始数据
        清洗后数据 = 数据.copy()
        
        # 记录原始数据信息
        原始行数 = len(清洗后数据)
        原始列数 = len(清洗后数据.columns)
        原始缺失值数 = 清洗后数据.isna().sum().sum()
        
        logger.info(f"开始清洗数据: {原始行数}行, {原始列数}列, {原始缺失值数}个缺失值")
        
        # 应用各种清洗方法
        
        # 1. 去重
        if 选项.get('去重', True):
            清洗后数据 = self.去重(清洗后数据)
        
        # 2. 排序
        if 选项.get('排序', True):
            排序列 = 选项.get('排序列')
            升序 = 选项.get('升序', True)
            清洗后数据 = self.排序(清洗后数据, 排序列, 升序)
        
        # 3. 缺失值处理
        缺失值处理方法 = 选项.get('缺失值处理', '前向填充')
        if 缺失值处理方法 != '无':
            清洗后数据 = self.处理缺失值(清洗后数据, 缺失值处理方法)
        
        # 4. 异常值处理
        异常值处理方法 = 选项.get('异常值处理', '无')
        if 异常值处理方法 != '无':
            异常值阈值 = 选项.get('异常值阈值', 3.0)
            清洗后数据 = self.处理异常值(清洗后数据, 异常值处理方法, 异常值阈值)
        
        # 5. 标准化/归一化
        标准化方法 = 选项.get('标准化', '无')
        if 标准化方法 != '无':
            标准化列 = 选项.get('标准化列')
            清洗后数据 = self.标准化(清洗后数据, 标准化方法, 标准化列)
        
        # 6. 自定义处理
        自定义处理函数 = 选项.get('自定义处理')
        if 自定义处理函数 is not None and callable(自定义处理函数):
            清洗后数据 = 自定义处理函数(清洗后数据)
        
        # 记录清洗后数据信息
        清洗后行数 = len(清洗后数据)
        清洗后缺失值数 = 清洗后数据.isna().sum().sum()
        
        logger.info(f"数据清洗完成: {清洗后行数}行, {原始列数}列, {清洗后缺失值数}个缺失值")
        logger.info(f"清洗效果: 去除了{原始行数 - 清洗后行数}行, 处理了{原始缺失值数 - 清洗后缺失值数}个缺失值")
        
        return 清洗后数据
    
    def 一键清洗(self, 数据: pd.DataFrame, 市场类型: str = None, 数据类型: str = None) -> pd.DataFrame:
        """
        一键清洗数据，根据市场类型和数据类型自动选择合适的清洗选项。
        
        参数:
            数据: 要清洗的数据
            市场类型: 市场类型，'A股'或'虚拟货币'
            数据类型: 数据类型，如'日线'、'分钟线'、'龙虎榜'等
            
        返回:
            清洗后的数据
        """
        # 根据市场类型和数据类型选择合适的清洗选项
        选项 = self._选择清洗选项(市场类型, 数据类型)
        
        # 清洗数据
        return self.清洗数据(数据, 选项)
    
    def 处理缺失值(self, 数据: pd.DataFrame, 方法: str, 列: List[str] = None) -> pd.DataFrame:
        """
        处理缺失值。
        
        参数:
            数据: 要处理的数据
            方法: 处理方法，可选值：'前向填充'、'后向填充'、'均值填充'、'中位数填充'、'众数填充'、'固定值填充'、'删除'
            列: 要处理的列，如果为None则处理所有列
            
        返回:
            处理后的数据
        """
        if 数据 is None or 数据.empty:
            return 数据
        
        # 创建数据副本
        结果 = 数据.copy()
        
        # 确定要处理的列
        处理列 = 列 if 列 is not None else 结果.columns
        
        # 根据方法处理缺失值
        if 方法 == '前向填充':
            结果[处理列] = 结果[处理列].fillna(method='ffill')
        elif 方法 == '后向填充':
            结果[处理列] = 结果[处理列].fillna(method='bfill')
        elif 方法 == '均值填充':
            for 列名 in 处理列:
                if pd.api.types.is_numeric_dtype(结果[列名]):
                    结果[列名] = 结果[列名].fillna(结果[列名].mean())
        elif 方法 == '中位数填充':
            for 列名 in 处理列:
                if pd.api.types.is_numeric_dtype(结果[列名]):
                    结果[列名] = 结果[列名].fillna(结果[列名].median())
        elif 方法 == '众数填充':
            for 列名 in 处理列:
                结果[列名] = 结果[列名].fillna(结果[列名].mode()[0] if not 结果[列名].mode().empty else None)
        elif 方法 == '固定值填充':
            固定值 = self.配置.get('固定填充值', 0)
            结果[处理列] = 结果[处理列].fillna(固定值)
        elif 方法 == '删除':
            结果 = 结果.dropna(subset=处理列)
        else:
            logger.warning(f"未知的缺失值处理方法: {方法}")
        
        return 结果
    
    def 处理异常值(self, 数据: pd.DataFrame, 方法: str, 阈值: float = 3.0, 列: List[str] = None) -> pd.DataFrame:
        """
        处理异常值。
        
        参数:
            数据: 要处理的数据
            方法: 处理方法，可选值：'标准差法'、'四分位法'、'均值法'、'删除'、'替换'
            阈值: 异常值阈值，用于标准差法和四分位法
            列: 要处理的列，如果为None则处理所有数值列
            
        返回:
            处理后的数据
        """
        if 数据 is None or 数据.empty:
            return 数据
        
        # 创建数据副本
        结果 = 数据.copy()
        
        # 确定要处理的列
        if 列 is not None:
            处理列 = 列
        else:
            处理列 = 结果.select_dtypes(include=[np.number]).columns
        
        # 根据方法处理异常值
        for 列名 in 处理列:
            if not pd.api.types.is_numeric_dtype(结果[列名]):
                continue
            
            if 方法 == '标准差法':
                # 使用均值和标准差识别异常值
                均值 = 结果[列名].mean()
                标准差 = 结果[列名].std()
                下限 = 均值 - 阈值 * 标准差
                上限 = 均值 + 阈值 * 标准差
                
                # 替换或删除异常值
                if self.配置.get('异常值替换', True):
                    结果.loc[结果[列名] < 下限, 列名] = 下限
                    结果.loc[结果[列名] > 上限, 列名] = 上限
                else:
                    结果 = 结果[(结果[列名] >= 下限) & (结果[列名] <= 上限)]
            
            elif 方法 == '四分位法':
                # 使用四分位数识别异常值
                Q1 = 结果[列名].quantile(0.25)
                Q3 = 结果[列名].quantile(0.75)
                IQR = Q3 - Q1
                下限 = Q1 - 阈值 * IQR
                上限 = Q3 + 阈值 * IQR
                
                # 替换或删除异常值
                if self.配置.get('异常值替换', True):
                    结果.loc[结果[列名] < 下限, 列名] = 下限
                    结果.loc[结果[列名] > 上限, 列名] = 上限
                else:
                    结果 = 结果[(结果[列名] >= 下限) & (结果[列名] <= 上限)]
            
            elif 方法 == '均值法':
                # 使用均值替换异常值
                均值 = 结果[列名].mean()
                标准差 = 结果[列名].std()
                下限 = 均值 - 阈值 * 标准差
                上限 = 均值 + 阈值 * 标准差
                
                结果.loc[结果[列名] < 下限, 列名] = 均值
                结果.loc[结果[列名] > 上限, 列名] = 均值
            
            elif 方法 == '删除':
                # 删除异常值
                均值 = 结果[列名].mean()
                标准差 = 结果[列名].std()
                下限 = 均值 - 阈值 * 标准差
                上限 = 均值 + 阈值 * 标准差
                
                结果 = 结果[(结果[列名] >= 下限) & (结果[列名] <= 上限)]
            
            elif 方法 == '替换':
                # 使用指定值替换异常值
                替换值 = self.配置.get('异常值替换值', None)
                均值 = 结果[列名].mean()
                标准差 = 结果[列名].std()
                下限 = 均值 - 阈值 * 标准差
                上限 = 均值 + 阈值 * 标准差
                
                if 替换值 is not None:
                    结果.loc[结果[列名] < 下限, 列名] = 替换值
                    结果.loc[结果[列名] > 上限, 列名] = 替换值
                else:
                    结果.loc[结果[列名] < 下限, 列名] = 均值
                    结果.loc[结果[列名] > 上限, 列名] = 均值
            
            else:
                logger.warning(f"未知的异常值处理方法: {方法}")
        
        return 结果
    
    def 标准化(self, 数据: pd.DataFrame, 方法: str, 列: List[str] = None) -> pd.DataFrame:
        """
        标准化/归一化数据。
        
        参数:
            数据: 要处理的数据
            方法: 处理方法，可选值：'Z-Score'、'Min-Max'、'最大绝对值'、'稳健'
            列: 要处理的列，如果为None则处理所有数值列
            
        返回:
            处理后的数据
        """
        if 数据 is None or 数据.empty:
            return 数据
        
        # 创建数据副本
        结果 = 数据.copy()
        
        # 确定要处理的列
        if 列 is not None:
            处理列 = 列
        else:
            处理列 = 结果.select_dtypes(include=[np.number]).columns
        
        # 根据方法标准化数据
        for 列名 in 处理列:
            if not pd.api.types.is_numeric_dtype(结果[列名]):
                continue
            
            if 方法 == 'Z-Score':
                # Z-Score标准化: (x - mean) / std
                均值 = 结果[列名].mean()
                标准差 = 结果[列名].std()
                if 标准差 != 0:
                    结果[列名] = (结果[列名] - 均值) / 标准差
            
            elif 方法 == 'Min-Max':
                # Min-Max归一化: (x - min) / (max - min)
                最小值 = 结果[列名].min()
                最大值 = 结果[列名].max()
                if 最大值 != 最小值:
                    结果[列名] = (结果[列名] - 最小值) / (最大值 - 最小值)
            
            elif 方法 == '最大绝对值':
                # 最大绝对值缩放: x / max(abs(x))
                最大绝对值 = 结果[列名].abs().max()
                if 最大绝对值 != 0:
                    结果[列名] = 结果[列名] / 最大绝对值
            
            elif 方法 == '稳健':
                # 稳健缩放: (x - median) / IQR
                中位数 = 结果[列名].median()
                Q1 = 结果[列名].quantile(0.25)
                Q3 = 结果[列名].quantile(0.75)
                IQR = Q3 - Q1
                if IQR != 0:
                    结果[列名] = (结果[列名] - 中位数) / IQR
            
            else:
                logger.warning(f"未知的标准化方法: {方法}")
        
        return 结果
    
    def 去重(self, 数据: pd.DataFrame, 列: List[str] = None) -> pd.DataFrame:
        """
        去除重复行。
        
        参数:
            数据: 要处理的数据
            列: 用于判断重复的列，如果为None则使用所有列
            
        返回:
            处理后的数据
        """
        if 数据 is None or 数据.empty:
            return 数据
        
        # 去除重复行
        if 列 is not None:
            return 数据.drop_duplicates(subset=列)
        else:
            return 数据.drop_duplicates()
    
    def 排序(self, 数据: pd.DataFrame, 列: Union[str, List[str]] = None, 升序: bool = True) -> pd.DataFrame:
        """
        排序数据。
        
        参数:
            数据: 要处理的数据
            列: 用于排序的列，如果为None则尝试使用索引或时间列
            升序: 是否升序排序
            
        返回:
            处理后的数据
        """
        if 数据 is None or 数据.empty:
            return 数据
        
        # 确定排序列
        if 列 is None:
            # 尝试使用索引或时间列
            if isinstance(数据.index, pd.DatetimeIndex):
                # 如果索引是时间索引，则按索引排序
                return 数据.sort_index(ascending=升序)
            else:
                # 尝试查找时间列
                时间列候选 = [col for col in 数据.columns if '时间' in col or '日期' in col]
                if 时间列候选:
                    return 数据.sort_values(时间列候选[0], ascending=升序)
                else:
                    # 无法确定排序列，返回原始数据
                    logger.warning("无法确定排序列，返回原始数据")
                    return 数据
        else:
            # 使用指定列排序
            return 数据.sort_values(列, ascending=升序)
    
    def _选择清洗选项(self, 市场类型: str = None, 数据类型: str = None) -> Dict[str, Any]:
        """
        根据市场类型和数据类型选择合适的清洗选项。
        
        参数:
            市场类型: 市场类型，'A股'或'虚拟货币'
            数据类型: 数据类型，如'日线'、'分钟线'、'龙虎榜'等
            
        返回:
            清洗选项字典
        """
        # 默认选项
        选项 = self.默认清洗选项.copy()
        
        # 根据市场类型和数据类型调整选项
        if 市场类型 == 'A股':
            if 数据类型 == '日线':
                选项.update({
                    '缺失值处理': '前向填充',
                    '异常值处理': '标准差法',
                    '异常值阈值': 3.0,
                    '标准化': '无'
                })
            elif 数据类型 == '分钟线':
                选项.update({
                    '缺失值处理': '前向填充',
                    '异常值处理': '标准差法',
                    '异常值阈值': 4.0,
                    '标准化': '无'
                })
            elif 数据类型 == '龙虎榜':
                选项.update({
                    '缺失值处理': '删除',
                    '异常值处理': '无',
                    '标准化': '无'
                })
        elif 市场类型 == '虚拟货币':
            if 数据类型 == '日线':
                选项.update({
                    '缺失值处理': '前向填充',
                    '异常值处理': '四分位法',
                    '异常值阈值': 1.5,
                    '标准化': '无'
                })
            elif 数据类型 == '分钟线':
                选项.update({
                    '缺失值处理': '前向填充',
                    '异常值处理': '四分位法',
                    '异常值阈值': 2.0,
                    '标准化': '无'
                })
            elif 数据类型 == '链上数据':
                选项.update({
                    '缺失值处理': '后向填充',
                    '异常值处理': '标准差法',
                    '异常值阈值': 5.0,
                    '标准化': '无'
                })
        
        return 选项
