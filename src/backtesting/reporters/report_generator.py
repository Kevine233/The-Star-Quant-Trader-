"""
回测报告生成器模块

用于将回测结果转换为HTML、PDF等格式的报告。

日期：2025-05-17
"""

import os
import datetime
import json
import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# 配置日志
logger = logging.getLogger(__name__)

class ReportGenerator:
    """回测报告生成器，将回测结果转换为格式化报告"""
    
    def __init__(self, 
                output_dir: str = None, 
                template_dir: str = None):
        """
        初始化报告生成器。
        
        参数:
            output_dir: 报告输出目录，默认为项目根目录的reports文件夹
            template_dir: 报告模板目录，默认为当前模块目录下的templates文件夹
        """
        # 设置输出目录
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'reports')
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置模板目录
        if template_dir is None:
            self.template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        else:
            self.template_dir = template_dir
            
        # 确保模板目录存在
        os.makedirs(self.template_dir, exist_ok=True)
        
        logger.info(f"报告生成器初始化完成，输出目录: {self.output_dir}")
    
    def generate_html_report(self, 
                           backtest_results: Dict[str, Any], 
                           report_name: str = None) -> str:
        """
        生成HTML格式的回测报告。
        
        参数:
            backtest_results: 回测结果字典
            report_name: 报告名称，默认为自动生成
            
        返回:
            HTML报告文件路径
        """
        # 生成报告文件名
        if report_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            symbol = backtest_results.get('basic_info', {}).get('symbol', 'unknown')
            report_name = f"{symbol}_backtest_report_{timestamp}"
        
        if not report_name.endswith('.html'):
            report_name += '.html'
            
        report_path = os.path.join(self.output_dir, report_name)
        
        # 构建HTML内容
        html_content = self._build_html_content(backtest_results)
        
        # 写入文件
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"HTML报告已生成: {report_path}")
        return report_path
    
    def generate_pdf_report(self, 
                          backtest_results: Dict[str, Any],
                          report_name: str = None) -> str:
        """
        生成PDF格式的回测报告。
        
        参数:
            backtest_results: 回测结果字典
            report_name: 报告名称，默认为自动生成
            
        返回:
            PDF报告文件路径
        """
        try:
            import pdfkit
        except ImportError:
            logger.error("未安装pdfkit。请运行: pip install pdfkit")
            logger.error("另外还需要安装wkhtmltopdf: https://wkhtmltopdf.org/downloads.html")
            return ""
            
        # 先生成HTML报告
        html_path = self.generate_html_report(backtest_results, report_name)
        
        # 设置PDF文件路径
        pdf_path = html_path.replace('.html', '.pdf')
        
        # 转换为PDF
        try:
            pdfkit.from_file(html_path, pdf_path)
            logger.info(f"PDF报告已生成: {pdf_path}")
            return pdf_path
        except Exception as e:
            logger.error(f"PDF生成失败: {e}")
            return ""
    
    def _build_html_content(self, backtest_results: Dict[str, Any]) -> str:
        """构建HTML报告内容。"""
        # 提取基本信息
        basic_info = backtest_results.get('basic_info', {})
        results = backtest_results.get('backtest_results', {})
        trade_records = backtest_results.get('trade_records', [])
        chart_path = backtest_results.get('chart_path', '')
        
        # 开始构建HTML
        html = [
            '<!DOCTYPE html>',
            '<html lang="zh-cn">',
            '<head>',
            '    <meta charset="UTF-8">',
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f'    <title>回测报告 - {basic_info.get("symbol", "")}</title>',
            '    <style>',
            '        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }',
            '        .container { max-width: 1200px; margin: 0 auto; }',
            '        h1, h2, h3 { color: #2c3e50; }',
            '        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }',
            '        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }',
            '        th { background-color: #f2f2f2; }',
            '        tr:nth-child(even) { background-color: #f9f9f9; }',
            '        .chart { width: 100%; margin: 20px 0; }',
            '        .positive { color: #27ae60; }',
            '        .negative { color: #e74c3c; }',
            '        .summary { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }',
            '        .metric { padding: 15px; border-radius: 5px; background-color: #f8f9fa; flex: 1; min-width: 200px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }',
            '        .metric h3 { margin-top: 0; color: #7f8c8d; font-size: 14px; font-weight: normal; }',
            '        .metric p { margin-bottom: 0; font-size: 24px; font-weight: bold; }',
            '    </style>',
            '</head>',
            '<body>',
            '    <div class="container">',
            f'        <h1>回测报告 - {basic_info.get("symbol", "未知")} ({basic_info.get("strategy_name", "未知策略")})</h1>',
            f'        <p>回测期间: {basic_info.get("backtest_start_date", "")} 至 {basic_info.get("backtest_end_date", "")} (共 {basic_info.get("backtest_days", 0)} 个交易日)</p>',
            '        <h2>策略信息</h2>',
            '        <table>',
            '            <tr><th>策略名称</th><td>' + str(basic_info.get("strategy_name", "")) + '</td></tr>',
            '            <tr><th>策略描述</th><td>' + str(basic_info.get("strategy_description", "")) + '</td></tr>',
            '            <tr><th>策略参数</th><td>' + self._format_parameters(basic_info.get("strategy_parameters", {})) + '</td></tr>',
            '        </table>',
            '',
            '        <h2>回测结果摘要</h2>',
            '        <div class="summary">',
            '            <div class="metric">',
            '                <h3>初始资产</h3>',
            f'                <p>{self._format_number(results.get("initial_equity", 0))}</p>',
            '            </div>',
            '            <div class="metric">',
            '                <h3>最终资产</h3>',
            f'                <p>{self._format_number(results.get("final_equity", 0))}</p>',
            '            </div>',
            '            <div class="metric">',
            '                <h3>收益率</h3>',
            f'                <p class="{self._get_sign_class(results.get("total_return", 0))}">{self._format_percentage(results.get("total_return", 0))}</p>',
            '            </div>',
            '            <div class="metric">',
            '                <h3>年化收益率</h3>',
            f'                <p class="{self._get_sign_class(results.get("annual_return", 0))}">{self._format_percentage(results.get("annual_return", 0))}</p>',
            '            </div>',
            '        </div>',
            '',
            '        <div class="summary">',
            '            <div class="metric">',
            '                <h3>最大回撤</h3>',
            f'                <p class="negative">{self._format_percentage(results.get("max_drawdown", 0))}</p>',
            '            </div>',
            '            <div class="metric">',
            '                <h3>收益回撤比</h3>',
            f'                <p>{self._format_number(results.get("calmar_ratio", 0), 2)}</p>',
            '            </div>',
            '            <div class="metric">',
            '                <h3>夏普比率</h3>',
            f'                <p>{self._format_number(results.get("sharpe_ratio", 0), 2)}</p>',
            '            </div>',
            '            <div class="metric">',
            '                <h3>交易次数</h3>',
            f'                <p>{results.get("trade_count", 0)}</p>',
            '            </div>',
            '        </div>',
            '',
            '        <div class="summary">',
            '            <div class="metric">',
            '                <h3>盈利次数</h3>',
            f'                <p class="positive">{results.get("profit_count", 0)}</p>',
            '            </div>',
            '            <div class="metric">',
            '                <h3>亏损次数</h3>',
            f'                <p class="negative">{results.get("loss_count", 0)}</p>',
            '            </div>',
            '            <div class="metric">',
            '                <h3>胜率</h3>',
            f'                <p>{self._format_percentage(results.get("win_rate", 0))}</p>',
            '            </div>',
            '            <div class="metric">',
            '                <h3>盈亏比</h3>',
            f'                <p>{self._format_number(results.get("profit_loss_ratio", 0), 2)}</p>',
            '            </div>',
            '        </div>',
            '',
            '        <h2>回测图表</h2>'
        ]
        
        # 添加图表
        if chart_path and os.path.exists(chart_path):
            # 将图片转为base64嵌入
            with open(chart_path, 'rb') as f:
                img_data = f.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                html.append(f'        <img src="data:image/png;base64,{img_base64}" alt="回测图表" class="chart">')
        else:
            html.append('        <p>未生成图表或图表文件不存在。</p>')
        
        # 添加交易记录表格
        html.extend([
            '        <h2>交易记录</h2>',
            '        <table>',
            '            <tr>',
            '                <th>日期</th>',
            '                <th>类型</th>',
            '                <th>股票代码</th>',
            '                <th>价格</th>',
            '                <th>数量</th>',
            '                <th>交易金额</th>',
            '                <th>手续费</th>',
            '                <th>盈亏</th>',
            '                <th>剩余资金</th>',
            '            </tr>'
        ])
        
        # 添加交易记录
        for trade in trade_records[:100]:  # 限制显示前100条记录
            date = trade.get('date', trade.get('日期', ''))
            if isinstance(date, (datetime.datetime, pd.Timestamp)):
                date = date.strftime('%Y-%m-%d %H:%M:%S')
                
            trade_type = trade.get('type', trade.get('类型', ''))
            symbol = trade.get('symbol', trade.get('股票代码', ''))
            price = trade.get('price', trade.get('价格', 0))
            quantity = trade.get('quantity', trade.get('数量', 0))
            amount = trade.get('amount', trade.get('交易金额', 0))
            commission = trade.get('commission', trade.get('手续费', 0))
            pnl = trade.get('pnl', trade.get('盈亏', 0)) if trade_type.lower() in ['sell', '卖出'] else '-'
            remaining = trade.get('remaining_capital', trade.get('剩余资金', 0))
            
            pnl_class = ''
            if pnl != '-':
                pnl_class = 'positive' if pnl > 0 else 'negative'
                
            html.append('            <tr>')
            html.append(f'                <td>{date}</td>')
            html.append(f'                <td>{trade_type}</td>')
            html.append(f'                <td>{symbol}</td>')
            html.append(f'                <td>{self._format_number(price, 4)}</td>')
            html.append(f'                <td>{quantity}</td>')
            html.append(f'                <td>{self._format_number(amount)}</td>')
            html.append(f'                <td>{self._format_number(commission)}</td>')
            html.append(f'                <td class="{pnl_class}">{self._format_number(pnl) if pnl != "-" else "-"}</td>')
            html.append(f'                <td>{self._format_number(remaining)}</td>')
            html.append('            </tr>')
            
        html.extend([
            '        </table>',
            '        <p>注：如果交易记录过多，可能只显示部分记录。</p>',
            '    </div>',
            '</body>',
            '</html>'
        ])
        
        return '\n'.join(html)
    
    def _format_number(self, number, decimals=2):
        """格式化数字。"""
        if isinstance(number, (int, float)):
            if number == float('inf') or number == float('-inf'):
                return "∞"
            return f"{number:,.{decimals}f}"
        return str(number)
    
    def _format_percentage(self, number, decimals=2):
        """格式化百分比。"""
        if isinstance(number, (int, float)):
            if number == float('inf') or number == float('-inf'):
                return "∞"
            return f"{number * 100:,.{decimals}f}%"
        return str(number)
    
    def _format_parameters(self, params):
        """格式化参数字典。"""
        if not params:
            return "无"
            
        if isinstance(params, dict):
            return "<br>".join([f"{k}: {v}" for k, v in params.items()])
        
        return str(params)
    
    def _get_sign_class(self, number):
        """根据数字的正负返回CSS类名。"""
        if isinstance(number, (int, float)):
            if number > 0:
                return "positive"
            elif number < 0:
                return "negative"
        return ""


# 中文命名版
class 报告生成器(ReportGenerator):
    """回测报告生成器，将回测结果转换为格式化报告（中文版）"""
    
    def 生成HTML报告(self, 回测结果: Dict[str, Any], 报告名称: str = None) -> str:
        """生成HTML格式的回测报告。"""
        return self.generate_html_report(回测结果, 报告名称)
    
    def 生成PDF报告(self, 回测结果: Dict[str, Any], 报告名称: str = None) -> str:
        """生成PDF格式的回测报告。"""
        return self.generate_pdf_report(回测结果, 报告名称)