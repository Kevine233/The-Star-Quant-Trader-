"""
策略引擎模块，负责实现"跟随庄家"的核心算法和交易信号生成。
本模块包含策略基类和具体的"跟随庄家"策略实现。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import datetime
import logging
import time
import json
import os
from abc import ABC, abstractmethod

# 配置日志
logger = logging.getLogger(__name__)

class 策略基类(ABC):
    """
    策略抽象基类，定义了所有策略的通用方法。
    所有具体的策略实现都应该继承这个类。
    """
    
    def __init__(self, 配置: Dict[str, Any] = None):
        """
        初始化策略。
        
        参数:
            配置: 策略配置字典
        """
        self.配置 = 配置 if 配置 is not None else {}
        self.名称 = self.配置.get('名称', self.__class__.__name__)
        self.描述 = self.配置.get('描述', '')
        self.参数 = self.配置.get('参数', {})
        self.市场类型 = self.配置.get('市场类型', 'A股')
        
        # 策略状态
        self.已初始化 = False
        self.运行中 = False
        
        # 数据和信号
        self.数据 = {}
        self.信号 = []
        self.最新信号 = None
    
    @abstractmethod
    def 初始化(self, 数据管理器=None) -> bool:
        """
        初始化策略，加载必要的数据和模型。
        
        参数:
            数据管理器: 数据管理器实例，用于获取数据
            
        返回:
            初始化是否成功
        """
        pass
    
    @abstractmethod
    def 计算信号(self, 数据: pd.DataFrame) -> Dict[str, Any]:
        """
        计算交易信号。
        
        参数:
            数据: 用于计算信号的数据
            
        返回:
            交易信号字典
        """
        pass
    
    def 运行(self, 数据: pd.DataFrame = None) -> Dict[str, Any]:
        """
        运行策略，计算最新的交易信号。
        
        参数:
            数据: 用于计算信号的数据，如果为None则使用策略内部数据
            
        返回:
            最新的交易信号字典
        """
        if not self.已初始化:
            logger.warning(f"策略 {self.名称} 尚未初始化，无法运行")
            return {}
        
        self.运行中 = True
        
        try:
            # 使用提供的数据或内部数据
            使用数据 = 数据 if 数据 is not None else self.获取最新数据()
            
            if 使用数据 is None or 使用数据.empty:
                logger.warning(f"策略 {self.名称} 没有可用数据，无法计算信号")
                return {}
            
            # 计算信号
            信号 = self.计算信号(使用数据)
            
            # 记录信号
            if 信号:
                self.信号.append(信号)
                self.最新信号 = 信号
                logger.info(f"策略 {self.名称} 生成新信号: {信号}")
            
            return 信号
        
        except Exception as e:
            logger.error(f"策略 {self.名称} 运行时发生错误: {str(e)}")
            return {}
        
        finally:
            self.运行中 = False
    
    def 获取最新数据(self) -> pd.DataFrame:
        """
        获取策略使用的最新数据。
        
        返回:
            最新数据
        """
        # 默认实现，子类可以覆盖
        if not self.数据:
            return None
        
        # 返回最新的数据
        return list(self.数据.values())[-1]
    
    def 获取所有信号(self) -> List[Dict[str, Any]]:
        """
        获取策略生成的所有信号。
        
        返回:
            信号列表
        """
        return self.信号
    
    def 获取最新信号(self) -> Dict[str, Any]:
        """
        获取策略生成的最新信号。
        
        返回:
            最新信号字典
        """
        return self.最新信号
    
    def 设置参数(self, 参数: Dict[str, Any]) -> bool:
        """
        设置策略参数。
        
        参数:
            参数: 参数字典
            
        返回:
            设置是否成功
        """
        try:
            self.参数.update(参数)
            logger.info(f"策略 {self.名称} 参数已更新: {参数}")
            return True
        except Exception as e:
            logger.error(f"设置策略 {self.名称} 参数时发生错误: {str(e)}")
            return False
    
    def 获取参数(self) -> Dict[str, Any]:
        """
        获取策略参数。
        
        返回:
            参数字典
        """
        return self.参数
    
    def 保存参数(self, 文件路径: str) -> bool:
        """
        保存策略参数到文件。
        
        参数:
            文件路径: 保存参数的文件路径
            
        返回:
            保存是否成功
        """
        try:
            with open(文件路径, 'w', encoding='utf-8') as f:
                json.dump(self.参数, f, ensure_ascii=False, indent=4)
            logger.info(f"策略 {self.名称} 参数已保存到 {文件路径}")
            return True
        except Exception as e:
            logger.error(f"保存策略 {self.名称} 参数时发生错误: {str(e)}")
            return False
    
    def 加载参数(self, 文件路径: str) -> bool:
        """
        从文件加载策略参数。
        
        参数:
            文件路径: 参数文件路径
            
        返回:
            加载是否成功
        """
        try:
            with open(文件路径, 'r', encoding='utf-8') as f:
                参数 = json.load(f)
            self.参数 = 参数
            logger.info(f"策略 {self.名称} 参数已从 {文件路径} 加载")
            return True
        except Exception as e:
            logger.error(f"加载策略 {self.名称} 参数时发生错误: {str(e)}")
            return False


class 跟随庄家策略(策略基类):
    """
    跟随庄家策略，通过识别庄家行为和操纵迹象生成交易信号。
    """
    
    def __init__(self, 配置: Dict[str, Any] = None):
        """
        初始化跟随庄家策略。
        
        参数:
            配置: 策略配置字典
        """
        super().__init__(配置)
        
        # 设置默认参数
        默认参数 = {
            # 操纵可能性评分阈值
            '操纵可能性阈值': 0.7,
            
            # 龙虎榜特征权重
            '龙虎榜买入权重': 0.3,
            '龙虎榜卖出权重': -0.2,
            '龙虎榜净买入权重': 0.4,
            
            # 成交量特征权重
            '成交量突增权重': 0.2,
            '成交量连续放大权重': 0.15,
            
            # 价格特征权重
            '价格突破权重': 0.25,
            '价格回踩权重': 0.1,
            '价格波动率权重': 0.15,
            
            # 大单特征权重
            '大单买入权重': 0.3,
            '大单卖出权重': -0.2,
            '大单净买入权重': 0.35,
            
            # 机构特征权重
            '机构买入权重': 0.4,
            '机构卖出权重': -0.3,
            '机构净买入权重': 0.5,
            
            # 入场条件
            '入场条件': {
                '操纵可能性': 0.7,
                '价格上涨': True,
                '成交量放大': True
            },
            
            # 出场条件
            '出场条件': {
                '止盈百分比': 0.2,
                '止损百分比': 0.05,
                '最大持有天数': 10,
                '操纵可能性下降阈值': 0.3
            },
            
            # 仓位管理
            '最大单一持仓比例': 0.1,
            '最大总持仓比例': 0.5,
            '资金分配方式': '等权重'  # '等权重', '按信号强度', '按市值'
        }
        
        # 更新参数
        for 键, 值 in 默认参数.items():
            if 键 not in self.参数:
                self.参数[键] = 值
        
        # 策略特定状态
        self.持仓 = {}
        self.历史操纵可能性 = {}
    
    def 初始化(self, 数据管理器=None) -> bool:
        """
        初始化策略，加载必要的数据和模型。
        
        参数:
            数据管理器: 数据管理器实例，用于获取数据
            
        返回:
            初始化是否成功
        """
        try:
            logger.info(f"初始化策略 {self.名称}")
            
            # 如果提供了数据管理器，加载初始数据
            if 数据管理器 is not None:
                # 加载A股市场的日线数据
                if self.市场类型 == 'A股':
                    # 获取最近30天的日线数据
                    结束日期 = datetime.datetime.now().strftime('%Y-%m-%d')
                    开始日期 = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
                    
                    try:
                        日线数据 = 数据管理器.一键获取数据(
                            市场类型='A股',
                            数据类型='日线',
                            开始日期=开始日期,
                            结束日期=结束日期
                        )
                        self.数据['日线'] = 日线数据
                        logger.info(f"已加载A股日线数据: {len(日线数据)}行")
                    except Exception as e:
                        logger.warning(f"加载A股日线数据失败: {str(e)}")
                    
                    # 获取最近的龙虎榜数据
                    try:
                        龙虎榜数据 = 数据管理器.一键获取数据(
                            市场类型='A股',
                            数据类型='龙虎榜',
                            开始日期=开始日期,
                            结束日期=结束日期
                        )
                        self.数据['龙虎榜'] = 龙虎榜数据
                        logger.info(f"已加载A股龙虎榜数据: {len(龙虎榜数据)}行")
                    except Exception as e:
                        logger.warning(f"加载A股龙虎榜数据失败: {str(e)}")
                
                # 加载虚拟货币市场的数据
                elif self.市场类型 == '虚拟货币':
                    # 获取最近30天的日线数据
                    结束日期 = datetime.datetime.now().strftime('%Y-%m-%d')
                    开始日期 = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
                    
                    try:
                        日线数据 = 数据管理器.一键获取数据(
                            市场类型='虚拟货币',
                            数据类型='日线',
                            开始日期=开始日期,
                            结束日期=结束日期
                        )
                        self.数据['日线'] = 日线数据
                        logger.info(f"已加载虚拟货币日线数据: {len(日线数据)}行")
                    except Exception as e:
                        logger.warning(f"加载虚拟货币日线数据失败: {str(e)}")
                    
                    # 获取链上数据
                    try:
                        链上数据 = 数据管理器.一键获取数据(
                            市场类型='虚拟货币',
                            数据类型='链上数据',
                            开始日期=开始日期,
                            结束日期=结束日期
                        )
                        self.数据['链上数据'] = 链上数据
                        logger.info(f"已加载虚拟货币链上数据: {len(链上数据)}行")
                    except Exception as e:
                        logger.warning(f"加载虚拟货币链上数据失败: {str(e)}")
            
            self.已初始化 = True
            logger.info(f"策略 {self.名称} 初始化成功")
            return True
        
        except Exception as e:
            logger.error(f"初始化策略 {self.名称} 时发生错误: {str(e)}")
            self.已初始化 = False
            return False
    
    def 计算信号(self, 数据: pd.DataFrame) -> Dict[str, Any]:
        """
        计算交易信号。
        
        参数:
            数据: 用于计算信号的数据
            
        返回:
            交易信号字典
        """
        if 数据 is None or 数据.empty:
            logger.warning("没有数据可用于计算信号")
            return {}
        
        try:
            # 按股票代码分组处理数据
            分组数据 = 数据.groupby('股票代码') if '股票代码' in 数据.columns else {'all': 数据}
            
            信号列表 = []
            
            for 股票代码, 股票数据 in 分组数据:
                # 计算操纵可能性分数
                操纵可能性 = self._计算操纵可能性(股票数据)
                
                # 记录历史操纵可能性
                if 股票代码 not in self.历史操纵可能性:
                    self.历史操纵可能性[股票代码] = []
                self.历史操纵可能性[股票代码].append((datetime.datetime.now(), 操纵可能性))
                
                # 检查是否满足入场条件
                if self._检查入场条件(股票代码, 股票数据, 操纵可能性):
                    # 生成买入信号
                    信号 = self._生成买入信号(股票代码, 股票数据, 操纵可能性)
                    信号列表.append(信号)
                
                # 检查是否满足出场条件
                elif 股票代码 in self.持仓 and self._检查出场条件(股票代码, 股票数据, 操纵可能性):
                    # 生成卖出信号
                    信号 = self._生成卖出信号(股票代码, 股票数据, 操纵可能性)
                    信号列表.append(信号)
            
            # 如果有多个信号，选择操纵可能性最高的
            if 信号列表:
                最佳信号 = max(信号列表, key=lambda x: x.get('操纵可能性', 0))
                return 最佳信号
            
            return {}
        
        except Exception as e:
            logger.error(f"计算信号时发生错误: {str(e)}")
            return {}
    
    def _计算操纵可能性(self, 数据: pd.DataFrame) -> float:
        """
        计算操纵可能性分数。
        
        参数:
            数据: 股票数据
            
        返回:
            操纵可能性分数，范围[0, 1]
        """
        特征分数 = {}
        
        # 1. 龙虎榜特征
        if '龙虎榜买入金额' in 数据.columns and '龙虎榜卖出金额' in 数据.columns:
            买入金额 = 数据['龙虎榜买入金额'].iloc[-1] if not pd.isna(数据['龙虎榜买入金额'].iloc[-1]) else 0
            卖出金额 = 数据['龙虎榜卖出金额'].iloc[-1] if not pd.isna(数据['龙虎榜卖出金额'].iloc[-1]) else 0
            净买入金额 = 买入金额 - 卖出金额
            
            特征分数['龙虎榜买入'] = min(买入金额 / 1e8, 1) * self.参数['龙虎榜买入权重']
            特征分数['龙虎榜卖出'] = min(卖出金额 / 1e8, 1) * self.参数['龙虎榜卖出权重']
            特征分数['龙虎榜净买入'] = (0.5 + 0.5 * np.tanh(净买入金额 / 1e8)) * self.参数['龙虎榜净买入权重']
        
        # 2. 成交量特征
        if '成交量' in 数据.columns:
            # 计算成交量变化率
            数据['成交量变化率'] = 数据['成交量'].pct_change()
            
            # 成交量突增
            最近成交量变化率 = 数据['成交量变化率'].iloc[-1] if not pd.isna(数据['成交量变化率'].iloc[-1]) else 0
            特征分数['成交量突增'] = min(max(最近成交量变化率, 0), 3) / 3 * self.参数['成交量突增权重']
            
            # 成交量连续放大
            if len(数据) >= 3:
                连续放大 = all(数据['成交量变化率'].iloc[-3:] > 0)
                特征分数['成交量连续放大'] = self.参数['成交量连续放大权重'] if 连续放大 else 0
        
        # 3. 价格特征
        if '收盘价' in 数据.columns:
            # 计算价格变化率
            数据['价格变化率'] = 数据['收盘价'].pct_change()
            
            # 价格突破
            if '最高价' in 数据.columns and len(数据) >= 20:
                二十日最高价 = 数据['最高价'].iloc[-20:-1].max()
                最新收盘价 = 数据['收盘价'].iloc[-1]
                价格突破 = 最新收盘价 > 二十日最高价
                特征分数['价格突破'] = self.参数['价格突破权重'] if 价格突破 else 0
            
            # 价格回踩
            if '最低价' in 数据.columns and '最高价' in 数据.columns and len(数据) >= 5:
                五日最高价 = 数据['最高价'].iloc[-5:].max()
                五日最低价 = 数据['最低价'].iloc[-5:].min()
                最新收盘价 = 数据['收盘价'].iloc[-1]
                前一日收盘价 = 数据['收盘价'].iloc[-2]
                价格回踩 = 前一日收盘价 < 最新收盘价 and 前一日收盘价 <= 五日最低价 * 1.05
                特征分数['价格回踩'] = self.参数['价格回踩权重'] if 价格回踩 else 0
            
            # 价格波动率
            if len(数据) >= 20:
                二十日波动率 = 数据['价格变化率'].iloc[-20:].std() * np.sqrt(252)
                特征分数['价格波动率'] = min(二十日波动率 / 0.5, 1) * self.参数['价格波动率权重']
        
        # 4. 大单特征
        if '大单买入金额' in 数据.columns and '大单卖出金额' in 数据.columns:
            买入金额 = 数据['大单买入金额'].iloc[-1] if not pd.isna(数据['大单买入金额'].iloc[-1]) else 0
            卖出金额 = 数据['大单卖出金额'].iloc[-1] if not pd.isna(数据['大单卖出金额'].iloc[-1]) else 0
            净买入金额 = 买入金额 - 卖出金额
            
            特征分数['大单买入'] = min(买入金额 / 1e8, 1) * self.参数['大单买入权重']
            特征分数['大单卖出'] = min(卖出金额 / 1e8, 1) * self.参数['大单卖出权重']
            特征分数['大单净买入'] = (0.5 + 0.5 * np.tanh(净买入金额 / 1e8)) * self.参数['大单净买入权重']
        
        # 5. 机构特征
        if '机构买入金额' in 数据.columns and '机构卖出金额' in 数据.columns:
            买入金额 = 数据['机构买入金额'].iloc[-1] if not pd.isna(数据['机构买入金额'].iloc[-1]) else 0
            卖出金额 = 数据['机构卖出金额'].iloc[-1] if not pd.isna(数据['机构卖出金额'].iloc[-1]) else 0
            净买入金额 = 买入金额 - 卖出金额
            
            特征分数['机构买入'] = min(买入金额 / 1e8, 1) * self.参数['机构买入权重']
            特征分数['机构卖出'] = min(卖出金额 / 1e8, 1) * self.参数['机构卖出权重']
            特征分数['机构净买入'] = (0.5 + 0.5 * np.tanh(净买入金额 / 1e8)) * self.参数['机构净买入权重']
        
        # 计算总分
        总分 = sum(特征分数.values())
        
        # 归一化到[0, 1]范围
        操纵可能性 = 1 / (1 + np.exp(-总分))  # Sigmoid函数
        
        return 操纵可能性
    
    def _检查入场条件(self, 股票代码: str, 数据: pd.DataFrame, 操纵可能性: float) -> bool:
        """
        检查是否满足入场条件。
        
        参数:
            股票代码: 股票代码
            数据: 股票数据
            操纵可能性: 操纵可能性分数
            
        返回:
            是否满足入场条件
        """
        # 如果已经持有该股票，不再入场
        if 股票代码 in self.持仓:
            return False
        
        入场条件 = self.参数['入场条件']
        
        # 检查操纵可能性是否超过阈值
        if 操纵可能性 < 入场条件['操纵可能性']:
            return False
        
        # 检查价格是否上涨
        if 入场条件['价格上涨'] and '收盘价' in 数据.columns and len(数据) >= 2:
            价格上涨 = 数据['收盘价'].iloc[-1] > 数据['收盘价'].iloc[-2]
            if not 价格上涨:
                return False
        
        # 检查成交量是否放大
        if 入场条件['成交量放大'] and '成交量' in 数据.columns and len(数据) >= 2:
            成交量放大 = 数据['成交量'].iloc[-1] > 数据['成交量'].iloc[-2]
            if not 成交量放大:
                return False
        
        # 检查仓位限制
        当前总持仓比例 = sum(持仓['持仓比例'] for 持仓 in self.持仓.values())
        if 当前总持仓比例 >= self.参数['最大总持仓比例']:
            return False
        
        return True
    
    def _检查出场条件(self, 股票代码: str, 数据: pd.DataFrame, 操纵可能性: float) -> bool:
        """
        检查是否满足出场条件。
        
        参数:
            股票代码: 股票代码
            数据: 股票数据
            操纵可能性: 操纵可能性分数
            
        返回:
            是否满足出场条件
        """
        if 股票代码 not in self.持仓:
            return False
        
        持仓信息 = self.持仓[股票代码]
        出场条件 = self.参数['出场条件']
        
        # 检查止盈条件
        if '收盘价' in 数据.columns:
            最新价格 = 数据['收盘价'].iloc[-1]
            买入价格 = 持仓信息['买入价格']
            价格变化率 = (最新价格 - 买入价格) / 买入价格
            
            if 价格变化率 >= 出场条件['止盈百分比']:
                logger.info(f"触发止盈条件: {股票代码} 价格变化率 {价格变化率:.2%} >= {出场条件['止盈百分比']:.2%}")
                return True
            
            # 检查止损条件
            if 价格变化率 <= -出场条件['止损百分比']:
                logger.info(f"触发止损条件: {股票代码} 价格变化率 {价格变化率:.2%} <= -{出场条件['止损百分比']:.2%}")
                return True
        
        # 检查持有时间
        买入时间 = 持仓信息['买入时间']
        当前时间 = datetime.datetime.now()
        持有天数 = (当前时间 - 买入时间).days
        
        if 持有天数 >= 出场条件['最大持有天数']:
            logger.info(f"触发最大持有天数条件: {股票代码} 持有天数 {持有天数} >= {出场条件['最大持有天数']}")
            return True
        
        # 检查操纵可能性下降
        历史操纵可能性 = self.历史操纵可能性.get(股票代码, [])
        if 历史操纵可能性:
            最高操纵可能性 = max(分数 for _, 分数 in 历史操纵可能性)
            if 最高操纵可能性 - 操纵可能性 >= 出场条件['操纵可能性下降阈值']:
                logger.info(f"触发操纵可能性下降条件: {股票代码} 下降幅度 {最高操纵可能性 - 操纵可能性:.2f} >= {出场条件['操纵可能性下降阈值']:.2f}")
                return True
        
        return False
    
    def _生成买入信号(self, 股票代码: str, 数据: pd.DataFrame, 操纵可能性: float) -> Dict[str, Any]:
        """
        生成买入信号。
        
        参数:
            股票代码: 股票代码
            数据: 股票数据
            操纵可能性: 操纵可能性分数
            
        返回:
            买入信号字典
        """
        # 计算买入数量和价格
        买入价格 = 数据['收盘价'].iloc[-1] if '收盘价' in 数据.columns else 0
        
        # 根据资金分配方式计算持仓比例
        if self.参数['资金分配方式'] == '等权重':
            持仓比例 = self.参数['最大单一持仓比例']
        elif self.参数['资金分配方式'] == '按信号强度':
            持仓比例 = self.参数['最大单一持仓比例'] * 操纵可能性
        elif self.参数['资金分配方式'] == '按市值':
            # 这里需要市值数据，如果没有则使用等权重
            持仓比例 = self.参数['最大单一持仓比例']
        else:
            持仓比例 = self.参数['最大单一持仓比例']
        
        # 更新持仓信息
        self.持仓[股票代码] = {
            '买入时间': datetime.datetime.now(),
            '买入价格': 买入价格,
            '持仓比例': 持仓比例,
            '操纵可能性': 操纵可能性
        }
        
        # 生成信号
        信号 = {
            '时间': datetime.datetime.now(),
            '标的': 股票代码,
            '方向': '买入',
            '价格类型': '限价',
            '价格': 买入价格,
            '持仓比例': 持仓比例,
            '操纵可能性': 操纵可能性,
            '信号来源': self.名称,
            '信号解释': f"操纵可能性分数 {操纵可能性:.2f} 超过阈值 {self.参数['操纵可能性阈值']:.2f}，"
                    f"价格和成交量形态符合庄家操作特征，建议买入。"
        }
        
        return 信号
    
    def _生成卖出信号(self, 股票代码: str, 数据: pd.DataFrame, 操纵可能性: float) -> Dict[str, Any]:
        """
        生成卖出信号。
        
        参数:
            股票代码: 股票代码
            数据: 股票数据
            操纵可能性: 操纵可能性分数
            
        返回:
            卖出信号字典
        """
        持仓信息 = self.持仓[股票代码]
        卖出价格 = 数据['收盘价'].iloc[-1] if '收盘价' in 数据.columns else 0
        买入价格 = 持仓信息['买入价格']
        持仓比例 = 持仓信息['持仓比例']
        
        # 计算收益率
        收益率 = (卖出价格 - 买入价格) / 买入价格
        
        # 生成卖出原因
        卖出原因 = ""
        出场条件 = self.参数['出场条件']
        
        if 收益率 >= 出场条件['止盈百分比']:
            卖出原因 = f"触发止盈条件，收益率 {收益率:.2%} >= {出场条件['止盈百分比']:.2%}"
        elif 收益率 <= -出场条件['止损百分比']:
            卖出原因 = f"触发止损条件，收益率 {收益率:.2%} <= -{出场条件['止损百分比']:.2%}"
        else:
            买入时间 = 持仓信息['买入时间']
            当前时间 = datetime.datetime.now()
            持有天数 = (当前时间 - 买入时间).days
            
            if 持有天数 >= 出场条件['最大持有天数']:
                卖出原因 = f"触发最大持有天数条件，持有天数 {持有天数} >= {出场条件['最大持有天数']}"
            else:
                历史操纵可能性 = self.历史操纵可能性.get(股票代码, [])
                if 历史操纵可能性:
                    最高操纵可能性 = max(分数 for _, 分数 in 历史操纵可能性)
                    if 最高操纵可能性 - 操纵可能性 >= 出场条件['操纵可能性下降阈值']:
                        卖出原因 = f"触发操纵可能性下降条件，下降幅度 {最高操纵可能性 - 操纵可能性:.2f} >= {出场条件['操纵可能性下降阈值']:.2f}"
        
        # 从持仓中移除
        del self.持仓[股票代码]
        
        # 生成信号
        信号 = {
            '时间': datetime.datetime.now(),
            '标的': 股票代码,
            '方向': '卖出',
            '价格类型': '限价',
            '价格': 卖出价格,
            '持仓比例': 持仓比例,
            '操纵可能性': 操纵可能性,
            '收益率': 收益率,
            '信号来源': self.名称,
            '信号解释': f"卖出原因: {卖出原因}，当前操纵可能性分数 {操纵可能性:.2f}，收益率 {收益率:.2%}。"
        }
        
        return 信号


class 策略管理器:
    """
    策略管理器类，负责管理多个策略并提供统一的接口。
    """
    
    def __init__(self):
        """初始化策略管理器。"""
        self.策略 = {}
        self.数据管理器 = None
        self.交易执行器 = None
    
    def 设置数据管理器(self, 数据管理器):
        """
        设置数据管理器。
        
        参数:
            数据管理器: 数据管理器实例
        """
        self.数据管理器 = 数据管理器
        logger.info("已设置数据管理器")
    
    def 设置交易执行器(self, 交易执行器):
        """
        设置交易执行器。
        
        参数:
            交易执行器: 交易执行器实例
        """
        self.交易执行器 = 交易执行器
        logger.info("已设置交易执行器")
    
    def 注册策略(self, 名称: str, 策略: 策略基类):
        """
        注册策略。
        
        参数:
            名称: 策略名称
            策略: 策略实例
        """
        self.策略[名称] = 策略
        logger.info(f"已注册策略: {名称}")
    
    def 注销策略(self, 名称: str) -> bool:
        """
        注销策略。
        
        参数:
            名称: 策略名称
            
        返回:
            注销是否成功
        """
        if 名称 in self.策略:
            del self.策略[名称]
            logger.info(f"已注销策略: {名称}")
            return True
        else:
            logger.warning(f"策略不存在: {名称}")
            return False
    
    def 获取策略(self, 名称: str) -> Optional[策略基类]:
        """
        获取策略实例。
        
        参数:
            名称: 策略名称
            
        返回:
            策略实例，如果不存在则返回None
        """
        return self.策略.get(名称)
    
    def 获取所有策略(self) -> Dict[str, 策略基类]:
        """
        获取所有策略。
        
        返回:
            策略字典，键为策略名称，值为策略实例
        """
        return self.策略
    
    def 初始化策略(self, 名称: str) -> bool:
        """
        初始化指定策略。
        
        参数:
            名称: 策略名称
            
        返回:
            初始化是否成功
        """
        if 名称 not in self.策略:
            logger.warning(f"策略不存在: {名称}")
            return False
        
        策略 = self.策略[名称]
        return 策略.初始化(self.数据管理器)
    
    def 初始化所有策略(self) -> Dict[str, bool]:
        """
        初始化所有策略。
        
        返回:
            初始化结果字典，键为策略名称，值为初始化是否成功
        """
        结果 = {}
        
        for 名称, 策略 in self.策略.items():
            结果[名称] = 策略.初始化(self.数据管理器)
        
        return 结果
    
    def 运行策略(self, 名称: str, 数据: pd.DataFrame = None) -> Dict[str, Any]:
        """
        运行指定策略。
        
        参数:
            名称: 策略名称
            数据: 用于计算信号的数据，如果为None则使用策略内部数据
            
        返回:
            策略生成的信号
        """
        if 名称 not in self.策略:
            logger.warning(f"策略不存在: {名称}")
            return {}
        
        策略 = self.策略[名称]
        
        if not 策略.已初始化:
            初始化成功 = 策略.初始化(self.数据管理器)
            if not 初始化成功:
                logger.warning(f"策略 {名称} 初始化失败，无法运行")
                return {}
        
        信号 = 策略.运行(数据)
        
        # 如果设置了交易执行器且信号有效，则执行信号
        if self.交易执行器 is not None and 信号:
            self.交易执行器.处理信号(信号)
        
        return 信号
    
    def 运行所有策略(self, 数据: pd.DataFrame = None) -> Dict[str, Dict[str, Any]]:
        """
        运行所有策略。
        
        参数:
            数据: 用于计算信号的数据，如果为None则使用策略内部数据
            
        返回:
            策略信号字典，键为策略名称，值为策略生成的信号
        """
        结果 = {}
        
        for 名称, 策略 in self.策略.items():
            结果[名称] = self.运行策略(名称, 数据)
        
        return 结果
    
    def 获取策略信号(self, 名称: str) -> Dict[str, Any]:
        """
        获取指定策略的最新信号。
        
        参数:
            名称: 策略名称
            
        返回:
            策略的最新信号
        """
        if 名称 not in self.策略:
            logger.warning(f"策略不存在: {名称}")
            return {}
        
        策略 = self.策略[名称]
        return 策略.获取最新信号()
    
    def 获取所有策略信号(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有策略的最新信号。
        
        返回:
            策略信号字典，键为策略名称，值为策略的最新信号
        """
        结果 = {}
        
        for 名称, 策略 in self.策略.items():
            结果[名称] = 策略.获取最新信号()
        
        return 结果
    
    def 设置策略参数(self, 名称: str, 参数: Dict[str, Any]) -> bool:
        """
        设置指定策略的参数。
        
        参数:
            名称: 策略名称
            参数: 参数字典
            
        返回:
            设置是否成功
        """
        if 名称 not in self.策略:
            logger.warning(f"策略不存在: {名称}")
            return False
        
        策略 = self.策略[名称]
        return 策略.设置参数(参数)
    
    def 获取策略参数(self, 名称: str) -> Dict[str, Any]:
        """
        获取指定策略的参数。
        
        参数:
            名称: 策略名称
            
        返回:
            策略参数字典
        """
        if 名称 not in self.策略:
            logger.warning(f"策略不存在: {名称}")
            return {}
        
        策略 = self.策略[名称]
        return 策略.获取参数()
    
    def 保存策略参数(self, 名称: str, 文件路径: str) -> bool:
        """
        保存指定策略的参数到文件。
        
        参数:
            名称: 策略名称
            文件路径: 保存参数的文件路径
            
        返回:
            保存是否成功
        """
        if 名称 not in self.策略:
            logger.warning(f"策略不存在: {名称}")
            return False
        
        策略 = self.策略[名称]
        return 策略.保存参数(文件路径)
    
    def 加载策略参数(self, 名称: str, 文件路径: str) -> bool:
        """
        从文件加载指定策略的参数。
        
        参数:
            名称: 策略名称
            文件路径: 参数文件路径
            
        返回:
            加载是否成功
        """
        if 名称 not in self.策略:
            logger.warning(f"策略不存在: {名称}")
            return False
        
        策略 = self.策略[名称]
        return 策略.加载参数(文件路径)
