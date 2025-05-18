"""
Web界面模块，提供本地Web服务和远程监控功能。
本模块使用Flask框架实现简洁的Web界面，支持查看系统状态、收益率、交易日志，
以及远程启动/暂停交易功能。
"""

import os
import json
import datetime
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Union, Tuple

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 配置日志
logger = logging.getLogger(__name__)

class Web界面:
    """
    Web界面类，负责提供本地Web服务和远程监控功能。
    """
    
    def __init__(self, 
               系统管理器 = None,
               配置: Dict[str, Any] = None,
               静态文件夹: str = None,
               模板文件夹: str = None):
        """
        初始化Web界面。
        
        参数:
            系统管理器: 系统管理器实例，用于控制系统各个模块
            配置: Web界面配置字典
            静态文件夹: 静态文件夹路径
            模板文件夹: 模板文件夹路径
        """
        self.系统管理器 = 系统管理器
        self.配置 = 配置 if 配置 is not None else {}
        
        # 设置默认配置
        self.配置.setdefault('主机', '0.0.0.0')
        self.配置.setdefault('端口', 5000)
        self.配置.setdefault('调试模式', False)
        self.配置.setdefault('密钥', os.urandom(24).hex())
        self.配置.setdefault('用户名', 'admin')
        self.配置.setdefault('密码', 'admin')  # 默认密码，建议修改
        
        # 创建Flask应用
        self.应用 = Flask(__name__, 
                      static_folder=静态文件夹,
                      template_folder=模板文件夹)
        
        # 设置密钥
        self.应用.secret_key = self.配置['密钥']
        
        # 注册路由
        self._注册路由()
        
        # Web服务器线程
        self.服务器线程 = None
        self.运行中 = False
    
    def 启动(self) -> bool:
        """
        启动Web服务器。
        
        返回:
            启动是否成功
        """
        if self.运行中:
            logger.warning("Web服务器已经在运行中")
            return True
        
        # 启动服务器线程
        self.运行中 = True
        self.服务器线程 = threading.Thread(target=self._服务器线程函数, daemon=True)
        self.服务器线程.start()
        
        logger.info(f"Web服务器已启动，访问地址: http://{self.配置['主机']}:{self.配置['端口']}")
        return True
    
    def 停止(self) -> bool:
        """
        停止Web服务器。
        
        返回:
            停止是否成功
        """
        if not self.运行中:
            logger.warning("Web服务器已经停止")
            return True
        
        # 停止服务器线程
        self.运行中 = False
        # Flask服务器无法优雅关闭，只能等待线程自然结束
        
        logger.info("Web服务器已停止")
        return True
    
    def 是否运行中(self) -> bool:
        """
        检查Web服务器是否在运行中。
        
        返回:
            是否在运行中
        """
        return self.运行中
    
    def _服务器线程函数(self):
        """Web服务器线程的主函数。"""
        try:
            self.应用.run(
                host=self.配置['主机'],
                port=self.配置['端口'],
                debug=self.配置['调试模式'],
                use_reloader=False  # 禁用重载器，避免在线程中启动时出现问题
            )
        except Exception as e:
            logger.error(f"Web服务器运行时发生错误: {str(e)}")
            self.运行中 = False
    
    def _注册路由(self):
        """注册Flask路由。"""
        应用 = self.应用
        
        # 登录相关路由
        @应用.route('/login', methods=['GET', 'POST'])
        def 登录():
            if request.method == 'POST':
                用户名 = request.form.get('username')
                密码 = request.form.get('password')
                
                if 用户名 == self.配置['用户名'] and 密码 == self.配置['密码']:
                    session['已登录'] = True
                    return redirect(url_for('首页'))
                else:
                    flash('用户名或密码错误')
            
            return render_template('login.html')
        
        @应用.route('/logout')
        def 登出():
            session.pop('已登录', None)
            return redirect(url_for('登录'))
        
        # 登录检查装饰器
        def 需要登录(f):
            def 装饰函数(*args, **kwargs):
                if not session.get('已登录'):
                    return redirect(url_for('登录'))
                return f(*args, **kwargs)
            装饰函数.__name__ = f.__name__
            return 装饰函数
        
        # 主页路由
        @应用.route('/')
        @需要登录
        def 首页():
            return render_template('index.html')
        
        # API路由 - 系统状态
        @应用.route('/api/status')
        @需要登录
        def 获取状态():
            if not self.系统管理器:
                return jsonify({
                    'status': 'error',
                    'message': '系统管理器未初始化'
                })
            
            try:
                状态 = self.系统管理器.获取状态()
                return jsonify({
                    'status': 'success',
                    'data': 状态
                })
            except Exception as e:
                logger.error(f"获取系统状态时发生错误: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': f'获取系统状态时发生错误: {str(e)}'
                })
        
        # API路由 - 收益率
        @应用.route('/api/performance')
        @需要登录
        def 获取收益率():
            if not self.系统管理器:
                return jsonify({
                    'status': 'error',
                    'message': '系统管理器未初始化'
                })
            
            try:
                收益数据 = self.系统管理器.获取收益数据()
                
                # 创建收益率图表
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # 添加策略收益曲线
                fig.add_trace(
                    go.Scatter(
                        x=收益数据['日期'],
                        y=收益数据['策略收益率'],
                        name="策略收益率",
                        line=dict(color='rgb(0, 128, 255)', width=2)
                    ),
                    secondary_y=False
                )
                
                # 添加基准收益曲线（如有）
                if '基准收益率' in 收益数据:
                    fig.add_trace(
                        go.Scatter(
                            x=收益数据['日期'],
                            y=收益数据['基准收益率'],
                            name="基准收益率",
                            line=dict(color='rgb(128, 128, 128)', width=2, dash='dash')
                        ),
                        secondary_y=False
                    )
                
                # 添加买入持有收益曲线（如有）
                if '买入持有收益率' in 收益数据:
                    fig.add_trace(
                        go.Scatter(
                            x=收益数据['日期'],
                            y=收益数据['买入持有收益率'],
                            name="买入持有收益率",
                            line=dict(color='rgb(255, 128, 0)', width=2, dash='dot')
                        ),
                        secondary_y=False
                    )
                
                # 添加回撤曲线（如有）
                if '回撤' in 收益数据:
                    fig.add_trace(
                        go.Scatter(
                            x=收益数据['日期'],
                            y=收益数据['回撤'],
                            name="回撤",
                            line=dict(color='rgb(255, 0, 0)', width=1.5),
                            fill='tozeroy',
                            fillcolor='rgba(255, 0, 0, 0.1)'
                        ),
                        secondary_y=True
                    )
                
                # 更新图表布局
                fig.update_layout(
                    title='策略收益率',
                    xaxis_title='日期',
                    yaxis_title='收益率 (%)',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(l=50, r=50, t=80, b=50),
                    height=500
                )
                
                # 更新y轴标题
                fig.update_yaxes(title_text="收益率 (%)", secondary_y=False)
                fig.update_yaxes(title_text="回撤 (%)", secondary_y=True)
                
                # 转换为JSON
                图表JSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                
                # 计算关键指标
                指标 = {
                    '总收益率': f"{收益数据['策略收益率'].iloc[-1]:.2f}%",
                    '年化收益率': f"{收益数据.get('年化收益率', 0):.2f}%",
                    '夏普比率': f"{收益数据.get('夏普比率', 0):.2f}",
                    '最大回撤': f"{收益数据.get('最大回撤', 0):.2f}%",
                    '收益回撤比': f"{收益数据.get('收益回撤比', 0):.2f}",
                    '胜率': f"{收益数据.get('胜率', 0) * 100:.2f}%",
                    '盈亏比': f"{收益数据.get('盈亏比', 0):.2f}"
                }
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'chart': 图表JSON,
                        'metrics': 指标
                    }
                })
            except Exception as e:
                logger.error(f"获取收益率数据时发生错误: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': f'获取收益率数据时发生错误: {str(e)}'
                })
        
        # API路由 - 交易日志
        @应用.route('/api/trade_logs')
        @需要登录
        def 获取交易日志():
            if not self.系统管理器:
                return jsonify({
                    'status': 'error',
                    'message': '系统管理器未初始化'
                })
            
            try:
                # 获取分页参数
                页码 = int(request.args.get('page', 1))
                每页数量 = int(request.args.get('per_page', 20))
                
                交易日志 = self.系统管理器.获取交易日志(页码, 每页数量)
                return jsonify({
                    'status': 'success',
                    'data': 交易日志
                })
            except Exception as e:
                logger.error(f"获取交易日志时发生错误: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': f'获取交易日志时发生错误: {str(e)}'
                })
        
        # API路由 - 持仓信息
        @应用.route('/api/positions')
        @需要登录
        def 获取持仓信息():
            if not self.系统管理器:
                return jsonify({
                    'status': 'error',
                    'message': '系统管理器未初始化'
                })
            
            try:
                持仓信息 = self.系统管理器.获取持仓信息()
                return jsonify({
                    'status': 'success',
                    'data': 持仓信息
                })
            except Exception as e:
                logger.error(f"获取持仓信息时发生错误: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': f'获取持仓信息时发生错误: {str(e)}'
                })
        
        # API路由 - 启动交易
        @应用.route('/api/start_trading', methods=['POST'])
        @需要登录
        def 启动交易():
            if not self.系统管理器:
                return jsonify({
                    'status': 'error',
                    'message': '系统管理器未初始化'
                })
            
            try:
                # 获取参数
                参数组 = request.json.get('param_group')
                
                结果 = self.系统管理器.启动交易(参数组)
                if 结果:
                    return jsonify({
                        'status': 'success',
                        'message': '交易已启动'
                    })
                else:
                    return jsonify({
                        'status': 'error',
                        'message': '启动交易失败'
                    })
            except Exception as e:
                logger.error(f"启动交易时发生错误: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': f'启动交易时发生错误: {str(e)}'
                })
        
        # API路由 - 暂停交易
        @应用.route('/api/stop_trading', methods=['POST'])
        @需要登录
        def 暂停交易():
            if not self.系统管理器:
                return jsonify({
                    'status': 'error',
                    'message': '系统管理器未初始化'
                })
            
            try:
                结果 = self.系统管理器.暂停交易()
                if 结果:
                    return jsonify({
                        'status': 'success',
                        'message': '交易已暂停'
                    })
                else:
                    return jsonify({
                        'status': 'error',
                        'message': '暂停交易失败'
                    })
            except Exception as e:
                logger.error(f"暂停交易时发生错误: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': f'暂停交易时发生错误: {str(e)}'
                })
        
        # API路由 - 获取参数组列表
        @应用.route('/api/param_groups')
        @需要登录
        def 获取参数组列表():
            if not self.系统管理器:
                return jsonify({
                    'status': 'error',
                    'message': '系统管理器未初始化'
                })
            
            try:
                参数组列表 = self.系统管理器.获取参数组列表()
                return jsonify({
                    'status': 'success',
                    'data': 参数组列表
                })
            except Exception as e:
                logger.error(f"获取参数组列表时发生错误: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': f'获取参数组列表时发生错误: {str(e)}'
                })
        
        # 错误处理
        @应用.errorhandler(404)
        def 页面未找到(e):
            return render_template('404.html'), 404
        
        @应用.errorhandler(500)
        def 服务器错误(e):
            return render_template('500.html'), 500


class 系统管理器:
    """
    系统管理器类，负责协调系统各个模块的运行。
    这个类是Web界面与后端模块之间的桥梁。
    """
    
    def __init__(self,
               数据模块 = None,
               策略模块 = None,
               回测模块 = None,
               交易模块 = None,
               风控模块 = None,
               通知模块 = None,
               配置: Dict[str, Any] = None):
        """
        初始化系统管理器。
        
        参数:
            数据模块: 数据模块实例
            策略模块: 策略模块实例
            回测模块: 回测模块实例
            交易模块: 交易模块实例
            风控模块: 风控模块实例
            通知模块: 通知模块实例
            配置: 系统配置字典
        """
        self.数据模块 = 数据模块
        self.策略模块 = 策略模块
        self.回测模块 = 回测模块
        self.交易模块 = 交易模块
        self.风控模块 = 风控模块
        self.通知模块 = 通知模块
        self.配置 = 配置 if 配置 is not None else {}
        
        # 系统状态
        self.状态 = {
            '运行状态': '已停止',
            '启动时间': None,
            '运行时长': 0,
            '最后更新': datetime.datetime.now(),
            '交易状态': '已停止',
            '当前策略': None,
            '当前参数组': None
        }
        
        # 收益数据
        self.收益数据 = pd.DataFrame(columns=['日期', '策略收益率', '基准收益率', '买入持有收益率', '回撤'])
        
        # 交易日志
        self.交易日志 = []
        
        # 持仓信息
        self.持仓信息 = {}
        
        # 参数组列表
        self.参数组列表 = []
        
        # 状态更新线程
        self.状态更新线程 = None
        self.运行中 = False
    
    def 启动(self) -> bool:
        """
        启动系统管理器。
        
        返回:
            启动是否成功
        """
        if self.运行中:
            logger.warning("系统管理器已经在运行中")
            return True
        
        # 启动状态更新线程
        self.运行中 = True
        self.状态更新线程 = threading.Thread(target=self._状态更新线程函数, daemon=True)
        self.状态更新线程.start()
        
        # 更新系统状态
        self.状态['运行状态'] = '运行中'
        self.状态['启动时间'] = datetime.datetime.now()
        
        logger.info("系统管理器已启动")
        return True
    
    def 停止(self) -> bool:
        """
        停止系统管理器。
        
        返回:
            停止是否成功
        """
        if not self.运行中:
            logger.warning("系统管理器已经停止")
            return True
        
        # 停止状态更新线程
        self.运行中 = False
        if self.状态更新线程:
            self.状态更新线程.join(timeout=5.0)
            self.状态更新线程 = None
        
        # 更新系统状态
        self.状态['运行状态'] = '已停止'
        
        logger.info("系统管理器已停止")
        return True
    
    def 启动交易(self, 参数组: str = None) -> bool:
        """
        启动交易。
        
        参数:
            参数组: 要使用的参数组名称
            
        返回:
            启动是否成功
        """
        if not self.交易模块:
            logger.error("交易模块未初始化，无法启动交易")
            return False
        
        try:
            # 如果指定了参数组，加载参数
            if 参数组:
                self._加载参数组(参数组)
            
            # 启动交易
            结果 = self.交易模块.启动()
            
            # 更新系统状态
            if 结果:
                self.状态['交易状态'] = '运行中'
                self.状态['当前参数组'] = 参数组
            
            return 结果
        except Exception as e:
            logger.error(f"启动交易时发生错误: {str(e)}")
            return False
    
    def 暂停交易(self) -> bool:
        """
        暂停交易。
        
        返回:
            暂停是否成功
        """
        if not self.交易模块:
            logger.error("交易模块未初始化，无法暂停交易")
            return False
        
        try:
            # 暂停交易
            结果 = self.交易模块.停止()
            
            # 更新系统状态
            if 结果:
                self.状态['交易状态'] = '已停止'
            
            return 结果
        except Exception as e:
            logger.error(f"暂停交易时发生错误: {str(e)}")
            return False
    
    def 获取状态(self) -> Dict[str, Any]:
        """
        获取系统状态。
        
        返回:
            系统状态字典
        """
        # 更新运行时长
        if self.状态['启动时间']:
            self.状态['运行时长'] = (datetime.datetime.now() - self.状态['启动时间']).total_seconds()
        
        # 更新最后更新时间
        self.状态['最后更新'] = datetime.datetime.now()
        
        return self.状态
    
    def 获取收益数据(self) -> pd.DataFrame:
        """
        获取收益数据。
        
        返回:
            收益数据DataFrame
        """
        # 如果有回测模块，从回测模块获取最新收益数据
        if self.回测模块 and hasattr(self.回测模块, '获取收益数据'):
            try:
                self.收益数据 = self.回测模块.获取收益数据()
            except Exception as e:
                logger.error(f"从回测模块获取收益数据时发生错误: {str(e)}")
        
        # 如果有交易模块，从交易模块获取实盘收益数据
        if self.交易模块 and hasattr(self.交易模块, '获取收益数据'):
            try:
                实盘收益数据 = self.交易模块.获取收益数据()
                # 如果有实盘数据，优先使用实盘数据
                if not 实盘收益数据.empty:
                    self.收益数据 = 实盘收益数据
            except Exception as e:
                logger.error(f"从交易模块获取收益数据时发生错误: {str(e)}")
        
        return self.收益数据
    
    def 获取交易日志(self, 页码: int = 1, 每页数量: int = 20) -> Dict[str, Any]:
        """
        获取交易日志。
        
        参数:
            页码: 页码，从1开始
            每页数量: 每页显示的日志数量
            
        返回:
            交易日志字典，包含分页信息和日志数据
        """
        # 如果有交易模块，从交易模块获取最新交易日志
        if self.交易模块 and hasattr(self.交易模块, '获取交易日志'):
            try:
                self.交易日志 = self.交易模块.获取交易日志()
            except Exception as e:
                logger.error(f"从交易模块获取交易日志时发生错误: {str(e)}")
        
        # 计算分页
        总数量 = len(self.交易日志)
        总页数 = (总数量 + 每页数量 - 1) // 每页数量
        
        # 确保页码在有效范围内
        页码 = max(1, min(页码, 总页数))
        
        # 计算起始和结束索引
        起始索引 = (页码 - 1) * 每页数量
        结束索引 = min(起始索引 + 每页数量, 总数量)
        
        # 获取当前页的日志
        当前页日志 = self.交易日志[起始索引:结束索引]
        
        return {
            'total': 总数量,
            'pages': 总页数,
            'current_page': 页码,
            'per_page': 每页数量,
            'logs': 当前页日志
        }
    
    def 获取持仓信息(self) -> Dict[str, Any]:
        """
        获取持仓信息。
        
        返回:
            持仓信息字典
        """
        # 如果有交易模块，从交易模块获取最新持仓信息
        if self.交易模块 and hasattr(self.交易模块, '获取持仓信息'):
            try:
                self.持仓信息 = self.交易模块.获取持仓信息()
            except Exception as e:
                logger.error(f"从交易模块获取持仓信息时发生错误: {str(e)}")
        
        return self.持仓信息
    
    def 获取参数组列表(self) -> List[Dict[str, Any]]:
        """
        获取参数组列表。
        
        返回:
            参数组列表
        """
        # 如果有回测模块，从回测模块获取参数组列表
        if self.回测模块 and hasattr(self.回测模块, '获取参数组列表'):
            try:
                self.参数组列表 = self.回测模块.获取参数组列表()
            except Exception as e:
                logger.error(f"从回测模块获取参数组列表时发生错误: {str(e)}")
        
        return self.参数组列表
    
    def _状态更新线程函数(self):
        """状态更新线程的主函数。"""
        logger.info("状态更新线程已启动")
        
        while self.运行中:
            try:
                # 更新系统状态
                self._更新系统状态()
                
                # 更新收益数据
                self._更新收益数据()
                
                # 更新交易日志
                self._更新交易日志()
                
                # 更新持仓信息
                self._更新持仓信息()
                
                # 更新参数组列表
                self._更新参数组列表()
                
                # 休眠一段时间
                time.sleep(5.0)
            except Exception as e:
                logger.error(f"状态更新线程发生错误: {str(e)}")
                time.sleep(10.0)  # 发生错误时暂停更长时间
        
        logger.info("状态更新线程已停止")
    
    def _更新系统状态(self):
        """更新系统状态。"""
        # 更新交易状态
        if self.交易模块 and hasattr(self.交易模块, '是否运行中'):
            try:
                if self.交易模块.是否运行中():
                    self.状态['交易状态'] = '运行中'
                else:
                    self.状态['交易状态'] = '已停止'
            except Exception as e:
                logger.error(f"更新交易状态时发生错误: {str(e)}")
        
        # 更新当前策略
        if self.策略模块 and hasattr(self.策略模块, '获取当前策略'):
            try:
                self.状态['当前策略'] = self.策略模块.获取当前策略()
            except Exception as e:
                logger.error(f"更新当前策略时发生错误: {str(e)}")
    
    def _更新收益数据(self):
        """更新收益数据。"""
        # 在状态更新线程中调用获取收益数据方法
        self.获取收益数据()
    
    def _更新交易日志(self):
        """更新交易日志。"""
        # 在状态更新线程中调用获取交易日志方法
        if self.交易模块 and hasattr(self.交易模块, '获取交易日志'):
            try:
                self.交易日志 = self.交易模块.获取交易日志()
            except Exception as e:
                logger.error(f"更新交易日志时发生错误: {str(e)}")
    
    def _更新持仓信息(self):
        """更新持仓信息。"""
        # 在状态更新线程中调用获取持仓信息方法
        self.获取持仓信息()
    
    def _更新参数组列表(self):
        """更新参数组列表。"""
        # 在状态更新线程中调用获取参数组列表方法
        self.获取参数组列表()
    
    def _加载参数组(self, 参数组: str):
        """
        加载指定的参数组。
        
        参数:
            参数组: 参数组名称
        """
        if not self.回测模块 or not hasattr(self.回测模块, '加载参数组'):
            logger.warning("回测模块未初始化或不支持加载参数组")
            return
        
        try:
            self.回测模块.加载参数组(参数组)
            logger.info(f"已加载参数组: {参数组}")
        except Exception as e:
            logger.error(f"加载参数组时发生错误: {str(e)}")
            raise
