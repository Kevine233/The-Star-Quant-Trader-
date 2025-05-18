"""
庄家行为检测器单元测试

测试庄家行为检测器的各项功能。
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategies.smart_money_detector import SmartMoneyDetector, DataValidationError

class TestSmartMoneyDetector(unittest.TestCase):
    """测试庄家行为检测器类"""
    
    def setUp(self):
        """测试前准备工作"""
        # 创建一个测试配置
        self.test_config = {
            'volume_threshold': 2.5,
            'price_manipulation_window': 15,
            'big_order_threshold': 500000,
            'concentration_threshold': 0.5,
            'manipulation_score_threshold': 65
        }
        
        # 创建检测器实例
        self.detector = SmartMoneyDetector(self.test_config)
        
        # 创建测试数据
        self.create_test_data()
    
    def create_test_data(self):
        """创建测试数据集"""
        # 创建日期索引
        dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
        
        # 创建基本数据
        data = {
            'open': np.random.uniform(10, 15, size=100),
            'high': np.random.uniform(11, 16, size=100),
            'low': np.random.uniform(9, 14, size=100),
            'close': np.random.uniform(10, 15, size=100),
            'volume': np.random.uniform(1000, 10000, size=100)
        }
        
        # 确保high > open, close 且 low < open, close
        for i in range(100):
            high = max(data['open'][i], data['close'][i], data['high'][i])
            low = min(data['open'][i], data['close'][i], data['low'][i])
            data['high'][i] = high + 0.5
            data['low'][i] = low - 0.5
        
        # 创建DataFrame
        self.test_data = pd.DataFrame(data, index=dates)
        
        # 创建一个异常数据点 - 成交量突然增加10倍
        self.test_data.loc[dates[50], 'volume'] = self.test_data.loc[dates[49], 'volume'] * 10
        
        # 创建一个价格操纵模式 - 价格快速上涨后下跌
        for i in range(60, 65):
            self.test_data.loc[dates[i], 'close'] = self.test_data.loc[dates[i-1], 'close'] * 1.03
            self.test_data.loc[dates[i], 'high'] = self.test_data.loc[dates[i], 'close'] * 1.01
            self.test_data.loc[dates[i], 'volume'] = self.test_data.loc[dates[i-1], 'volume'] * 1.5
        
        for i in range(65, 70):
            self.test_data.loc[dates[i], 'close'] = self.test_data.loc[dates[i-1], 'close'] * 0.98
            self.test_data.loc[dates[i], 'low'] = self.test_data.loc[dates[i], 'close'] * 0.99
            self.test_data.loc[dates[i], 'volume'] = self.test_data.loc[dates[i-1], 'volume'] * 1.2
        
        # 创建空数据集和无效数据集用于测试边界情况
        self.empty_data = pd.DataFrame()
        self.invalid_data = pd.DataFrame({'date': dates, 'price': np.random.uniform(10, 15, size=100)})
    
    def test_initialization(self):
        """测试初始化"""
        # 测试默认配置
        detector = SmartMoneyDetector()
        self.assertIsNotNone(detector)
        
        # 测试传入配置
        detector = SmartMoneyDetector(self.test_config)
        self.assertEqual(detector.volume_threshold, 2.5)
        self.assertEqual(detector.price_manipulation_window, 15)
        self.assertEqual(detector.big_order_threshold, 500000)
        self.assertEqual(detector.concentration_threshold, 0.5)
        self.assertEqual(detector.manipulation_score_threshold, 65)
    
    def test_validate_input_data(self):
        """测试数据验证功能"""
        # 测试空数据
        with self.assertRaises(DataValidationError):
            self.detector._validate_input_data(self.empty_data, ['volume'])
        
        # 测试缺少所需列
        with self.assertRaises(DataValidationError):
            self.detector._validate_input_data(self.invalid_data, ['volume'])
        
        # 测试有效数据
        try:
            self.detector._validate_input_data(self.test_data, ['volume'])
            self.detector._validate_input_data(self.test_data, ['open', 'high', 'low', 'close', 'volume'])
        except DataValidationError:
            self.fail("_validate_input_data() raised DataValidationError unexpectedly!")
    
    def test_detect_volume_anomalies(self):
        """测试成交量异常检测功能"""
        # 测试正常流程
        result = self.detector.detect_volume_anomalies(self.test_data)
        
        # 验证结果是否包含预期的列
        self.assertIn('volume_zscore', result.columns)
        self.assertIn('volume_anomaly', result.columns)
        self.assertIn('consecutive_anomalies', result.columns)
        
        # 验证异常数据点是否被检测到
        self.assertTrue(abs(result.iloc[50]['volume_zscore']) > self.detector.volume_threshold)
        self.assertNotEqual(result.iloc[50]['volume_anomaly'], 0)
        
        # 测试无效数据处理
        result = self.detector.detect_volume_anomalies(self.empty_data)
        self.assertTrue(result.empty)
        
        result = self.detector.detect_volume_anomalies(self.invalid_data)
        self.assertEqual(len(result), len(self.invalid_data))
    
    def test_detect_price_manipulation(self):
        """测试价格操纵检测功能"""
        # 先检测成交量异常
        data_with_volume_anomalies = self.detector.detect_volume_anomalies(self.test_data)
        
        # 测试正常流程
        result = self.detector.detect_price_manipulation(data_with_volume_anomalies)
        
        # 验证结果是否包含预期的列
        self.assertIn('price_volatility', result.columns)
        self.assertIn('price_volatility_zscore', result.columns)
        self.assertIn('price_volume_corr', result.columns)
        self.assertIn('price_manipulation_score', result.columns)
        self.assertIn('price_manipulation', result.columns)
        
        # 验证异常数据点是否被检测到（前面创建的价格操纵模式）
        manipulation_detected = False
        for i in range(60, 70):
            if result.iloc[i]['price_manipulation'] > 0:
                manipulation_detected = True
                break
        
        self.assertTrue(manipulation_detected, "价格操纵模式未被检测到")
        
        # 测试无效数据处理
        result = self.detector.detect_price_manipulation(self.empty_data)
        self.assertTrue(result.empty)
        
        result = self.detector.detect_price_manipulation(self.invalid_data)
        self.assertEqual(len(result), len(self.invalid_data))
    
    def test_smart_money_detection_end_to_end(self):
        """测试端到端的庄家行为检测"""
        # 测试正常流程
        result = self.detector.detect_smart_money(self.test_data, stock_code="TEST001")
        
        # 验证结果是否包含预期的列
        expected_columns = [
            'volume_zscore', 'volume_anomaly', 'consecutive_anomalies',
            'price_volatility', 'price_volume_corr', 'price_manipulation_score',
            'price_manipulation', 'manipulation_score', 'manipulation_probability',
            'manipulation_direction'
        ]
        
        for col in expected_columns:
            self.assertIn(col, result.columns, f"结果中缺少列: {col}")
        
        # 验证缓存机制
        # 首先确认缓存已创建
        self.assertIn("TEST001", self.detector.result_cache)
        
        # 再次调用，应该使用缓存
        cached_result = self.detector.detect_smart_money(self.test_data, stock_code="TEST001")
        
        # 缓存的结果应该与原始结果相同
        pd.testing.assert_frame_equal(result, cached_result)
        
        # 测试无效数据处理
        result = self.detector.detect_smart_money(self.empty_data)
        self.assertTrue(result.empty)
    
    def test_get_params(self):
        """测试获取参数功能"""
        params = self.detector.get_params()
        
        # 验证返回的参数是否正确
        self.assertEqual(params['volume_threshold'], 2.5)
        self.assertEqual(params['price_manipulation_window'], 15)
        self.assertEqual(params['big_order_threshold'], 500000)
        self.assertEqual(params['concentration_threshold'], 0.5)
        self.assertEqual(params['manipulation_score_threshold'], 65)

if __name__ == '__main__':
    unittest.main() 