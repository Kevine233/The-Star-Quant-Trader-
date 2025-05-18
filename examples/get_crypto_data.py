"""
加密货币数据获取示例
演示如何使用CryptoDataProvider类获取不同交易所的数据
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入CryptoDataProvider
from src.data_sources.crypto_data import CryptoDataProvider

def load_config():
    """
    加载配置文件
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'config', 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取配置文件失败: {str(e)}")
        return {}

def test_all_sources(symbol='BTC_USDT', interval='1h', limit=100):
    """
    测试所有可用的数据源
    
    参数:
        symbol: 交易对符号
        interval: K线间隔
        limit: 获取的K线数量
    """
    # 加载配置
    config = load_config()
    crypto_config = config.get('data_source', {}).get('crypto', {})
    
    # 获取所有可用数据源
    available_sources = crypto_config.get('available_sources', 
                                         ['gateio', 'huobi', 'okex', 'binance'])
    
    # 测试每个数据源
    results = {}
    failed_sources = []
    
    for source in available_sources:
        print(f"\n===== 测试数据源: {source} =====")
        
        # 创建数据提供商实例
        provider_config = crypto_config.copy()
        provider_config['api_source'] = source
        data_provider = CryptoDataProvider(provider_config)
        
        try:
            # 获取数据
            start_time = datetime.now()
            df = data_provider.get_market_data(symbol, interval, limit)
            end_time = datetime.now()
            
            if df is not None and not df.empty:
                duration = (end_time - start_time).total_seconds()
                print(f"成功! 获取到 {len(df)} 条记录")
                print(f"耗时: {duration:.2f} 秒")
                print("\n数据示例:")
                print(df.head(3))
                
                results[source] = {
                    "success": True,
                    "records": len(df),
                    "duration": duration,
                    "data": df
                }
            else:
                print(f"失败! 未获取到数据")
                failed_sources.append(source)
                results[source] = {
                    "success": False
                }
        except Exception as e:
            print(f"出错! {str(e)}")
            failed_sources.append(source)
            results[source] = {
                "success": False,
                "error": str(e)
            }
    
    return results, failed_sources

def plot_data(results):
    """
    绘制各数据源的数据对比图
    
    参数:
        results: 测试结果字典
    """
    # 获取成功的数据源
    successful_sources = [source for source, result in results.items() 
                         if result.get('success', False)]
    
    if not successful_sources:
        print("没有成功获取数据的数据源，无法绘图")
        return
    
    # 创建图表
    plt.figure(figsize=(14, 8))
    
    # 绘制每个数据源的收盘价
    for source in successful_sources:
        df = results[source]['data']
        plt.plot(df['timestamp'], df['close'], label=source)
    
    # 设置图表属性
    plt.title('不同数据源的比特币价格对比')
    plt.xlabel('时间')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'crypto_data_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
    
    print(f"\n图表已保存到 {output_dir} 目录")
    
    # 显示图表
    plt.show()

def save_data(results):
    """
    保存各数据源的数据到CSV文件
    
    参数:
        results: 测试结果字典
    """
    # 获取成功的数据源
    successful_sources = [source for source, result in results.items() 
                         if result.get('success', False)]
    
    if not successful_sources:
        print("没有成功获取数据的数据源，无法保存")
        return
    
    # 创建保存目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存每个数据源的数据
    for source in successful_sources:
        df = results[source]['data']
        filename = os.path.join(output_dir, f'crypto_data_{source}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        df.to_csv(filename, index=False)
        print(f"数据已保存到 {filename}")

def recommend_best_source(results, failed_sources):
    """
    根据测试结果推荐最佳数据源
    
    参数:
        results: 测试结果字典
        failed_sources: 失败的数据源列表
    """
    # 获取成功的数据源
    successful_sources = [source for source, result in results.items() 
                         if result.get('success', False)]
    
    if not successful_sources:
        print("\n===== 推荐数据源 =====")
        print("所有数据源都失败了，请检查网络连接或尝试使用代理")
        return None
    
    # 计算评分 (基于响应时间和数据记录数)
    scores = {}
    for source in successful_sources:
        duration = results[source]['duration']
        records = results[source]['records']
        
        # 评分公式: 记录数越多越好，响应时间越短越好
        score = records / (duration + 0.1)  # 加0.1避免除零错误
        scores[source] = score
    
    # 找出最高分的数据源
    best_source = max(scores, key=scores.get)
    
    print("\n===== 推荐数据源 =====")
    print(f"测试成功的数据源: {', '.join(successful_sources)}")
    print(f"测试失败的数据源: {', '.join(failed_sources)}")
    print(f"推荐使用的数据源: {best_source}")
    
    # 打印详细评分
    print("\n各数据源评分:")
    for source, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{source}: {score:.2f} (记录数: {results[source]['records']}, 响应时间: {results[source]['duration']:.2f}秒)")
    
    return best_source

def update_config(best_source):
    """
    更新配置文件，设置推荐的数据源为默认数据源
    
    参数:
        best_source: 推荐的数据源名称
    """
    if not best_source:
        return
    
    # 加载配置
    config = load_config()
    
    # 更新配置
    if 'data_source' in config and 'crypto' in config['data_source']:
        config['data_source']['crypto']['api_source'] = best_source
        config['data_source']['crypto']['default_provider'] = best_source
        
        # 保存配置
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'config', 'config.json')
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            print(f"\n已更新配置文件，设置 {best_source} 为默认数据源")
        except Exception as e:
            print(f"\n更新配置文件失败: {str(e)}")

def main():
    """
    主函数
    """
    print("===== 加密货币数据源测试工具 =====")
    print("此工具将测试所有可用的加密货币数据源，并推荐最佳数据源。")
    
    # 测试所有数据源
    results, failed_sources = test_all_sources(symbol='BTC_USDT', interval='1h', limit=100)
    
    # 保存数据到CSV
    save_data(results)
    
    # 绘制数据对比图
    plot_data(results)
    
    # 推荐最佳数据源
    best_source = recommend_best_source(results, failed_sources)
    
    # 提示用户是否更新配置
    if best_source:
        answer = input(f"\n是否将 {best_source} 设置为默认数据源? (y/n): ")
        if answer.lower() == 'y':
            update_config(best_source)

if __name__ == "__main__":
    main() 