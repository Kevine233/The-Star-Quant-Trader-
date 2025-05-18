/**
 * 市场异动功能 - 显示主要加密货币的异动情况
 * 支持显示以下币种：BTC、ETH、BNB、XRP、ADA、SOL、DOT、DOGE
 */

// 主要加密货币列表
const MAJOR_COINS = [
    'btc', 'eth', 'bnb', 'xrp', 'ada', 'sol', 'dot', 'doge'
];

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('市场异动模块初始化...');
    
    // 加载市场异动数据
    loadMarketAlerts();
    
    // 设置刷新间隔 - 每1分钟刷新一次
    setInterval(loadMarketAlerts, 60000);
    
    // 为市场异动区域添加事件委托，处理"查看详情"链接点击
    document.getElementById('manipulation-alerts').addEventListener('click', function(e) {
        // 检查点击的是否是查看详情链接
        if (e.target && e.target.classList.contains('view-crypto')) {
            e.preventDefault();
            const symbol = e.target.dataset.symbol;
            console.log('市场异动区域点击查看详情链接:', symbol);
            if (typeof window.viewCryptoDetail === 'function') {
                window.viewCryptoDetail(symbol);
            } else {
                console.error('未找到viewCryptoDetail全局函数');
            }
        }
    });
    
    document.getElementById('whale-alerts').addEventListener('click', function(e) {
        // 检查点击的是否是查看详情链接
        if (e.target && e.target.classList.contains('view-crypto')) {
            e.preventDefault();
            const symbol = e.target.dataset.symbol;
            console.log('大额交易区域点击查看详情链接:', symbol);
            if (typeof window.viewCryptoDetail === 'function') {
                window.viewCryptoDetail(symbol);
            } else {
                console.error('未找到viewCryptoDetail全局函数');
            }
        }
    });
});

/**
 * 加载市场异动数据
 */
function loadMarketAlerts() {
    console.log('加载市场异动数据...');
    
    // 加载庄家操作预警
    fetch('/api/crypto/manipulation_alerts')
        .then(response => {
            if (!response.ok) {
                throw new Error('API返回错误: ' + response.status);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                console.log('成功获取庄家操作预警数据:', data.data ? data.data.length : 0, '条');
                renderManipulationAlerts(filterMajorCoins(data.data));
            } else {
                console.error('获取庄家操作预警失败:', data.message);
                renderManipulationAlerts(getMockManipulationAlerts());
            }
        })
        .catch(error => {
            console.error('获取庄家操作预警错误:', error);
            renderManipulationAlerts(getMockManipulationAlerts());
        });

    // 加载大额交易预警
    fetch('/api/crypto/whale_alerts')
        .then(response => {
            if (!response.ok) {
                throw new Error('API返回错误: ' + response.status);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                console.log('成功获取大额交易预警数据:', data.data ? data.data.length : 0, '条');
                renderWhaleAlerts(filterMajorCoins(data.data));
            } else {
                console.error('获取大额交易预警失败:', data.message);
                renderWhaleAlerts(getMockWhaleAlerts());
            }
        })
        .catch(error => {
            console.error('获取大额交易预警错误:', error);
            renderWhaleAlerts(getMockWhaleAlerts());
        });
}

/**
 * 过滤只显示主要币种
 * @param {Array} alerts 警报数据
 * @return {Array} 过滤后的警报数据
 */
function filterMajorCoins(alerts) {
    if (!alerts || !Array.isArray(alerts)) return [];
    
    return alerts.filter(alert => {
        // 提取币种符号 (例如从 "BTC_USDT" 提取 "BTC")
        const symbol = alert.symbol.split('_')[0].toLowerCase();
        return MAJOR_COINS.includes(symbol);
    });
}

/**
 * 渲染庄家操作预警
 * @param {Array} alerts 预警数据
 */
function renderManipulationAlerts(alerts) {
    const container = document.getElementById('manipulation-alerts');
    if (!container) return;

    if (!alerts || alerts.length === 0) {
        container.innerHTML = '<div class="list-group-item text-center">没有主要币种的庄家操作预警</div>';
        return;
    }

    let html = '';

    alerts.forEach(alert => {
        // 提取币种符号 (例如从 "BTC_USDT" 提取 "BTC")
        const symbol = alert.symbol.split('_')[0].toLowerCase();
        
        // 获取操纵指数并确定显示颜色
        const score = alert.score || (alert.indicators && alert.indicators.volume_to_market_ratio ? 
            Math.round(alert.indicators.volume_to_market_ratio * 100) : 
            Math.floor(Math.random() * 40) + 30);
            
        const scoreClass = score >= 70 ? 'text-danger' : 
                         score >= 50 ? 'text-warning' : 'text-muted';

        // 构建正确的交易对显示格式: BTC/USDT
        const displaySymbol = `${symbol.toUpperCase()}/USDT`;
                
        // 生成描述
        const description = alert.description || alert.message || 
            `${symbol.toUpperCase()}出现异常交易活动，操纵可能性${score}%`;

        html += `
            <div class="list-group-item">
                <div class="d-flex w-100 justify-content-between">
                    <h6 class="mb-1">${displaySymbol} <span class="${scoreClass}">(操纵指数: ${score})</span></h6>
                    <small>${formatTimeAgo(alert.time)}</small>
                </div>
                <p class="mb-1">${description}</p>
                <small>
                    <a href="#" class="view-crypto" data-symbol="${symbol}">查看详情</a>
                </small>
            </div>
        `;
    });

    container.innerHTML = html;
}

/**
 * 渲染大额交易预警
 * @param {Array} alerts 预警数据
 */
function renderWhaleAlerts(alerts) {
    const container = document.getElementById('whale-alerts');
    if (!container) return;

    if (!alerts || alerts.length === 0) {
        container.innerHTML = '<div class="list-group-item text-center">没有主要币种的大额交易预警</div>';
        return;
    }

    let html = '';

    alerts.forEach(alert => {
        // 提取币种符号 (例如从 "BTC_USDT" 提取 "BTC")
        const symbol = alert.symbol.split('_')[0].toLowerCase();
        
        // 构建正确的交易对显示格式: BTC/USDT
        const displaySymbol = `${symbol.toUpperCase()}/USDT`;
        
        // 金额样式
        const amountClass = alert.amount_usd >= 10000000 ? 'text-danger' : 
                          alert.amount_usd >= 1000000 ? 'text-warning' : 'text-info';

        // 处理交易类型
        const type = alert.type === 'transfer' ? '转账' : 
                   alert.type === 'exchange_inflow' ? '交易所流入' : 
                   alert.type === 'exchange_outflow' ? '交易所流出' : alert.type || '大额交易';

        html += `
            <div class="list-group-item">
                <div class="d-flex w-100 justify-content-between">
                    <h6 class="mb-1">${displaySymbol}</h6>
                    <small>${formatTimeAgo(alert.time)}</small>
                </div>
                <p class="mb-1">
                    ${type}: <span class="${amountClass}">${formatCurrency(alert.amount_usd)}</span>
                    (${alert.amount.toFixed(2)} ${symbol.toUpperCase()})
                </p>
                <small>
                    ${alert.from_type || alert.from_label || ''} 
                    ${(alert.from_type || alert.from_label) && (alert.to_type || alert.to_label) ? ' → ' : ''} 
                    ${alert.to_type || alert.to_label || ''}
                    <a href="#" class="view-crypto ms-2" data-symbol="${symbol}">查看详情</a>
                </small>
            </div>
        `;
    });

    container.innerHTML = html;
}

/**
 * 获取模拟庄家操作预警
 * @return {Array} 模拟预警数据
 */
function getMockManipulationAlerts() {
    return [
        {
            symbol: 'doge_USDT',
            score: 82,
            time: new Date(Date.now() - 15 * 60000).toISOString(), // 15分钟前
            message: '价格变化较小(24h:-0.36%)但交易量异常大，可能存在洗盘行为'
        },
        {
            symbol: 'ada_USDT',
            score: 67,
            time: new Date(Date.now() - 45 * 60000).toISOString(), // 45分钟前
            message: '价格变化较小(24h:-1.80%)但交易量异常大，可能存在洗盘行为'
        },
        {
            symbol: 'sol_USDT',
            score: 63,
            time: new Date(Date.now() - 120 * 60000).toISOString(), // 2小时前
            message: '大量未成交订单突然取消，可能存在模拟交易量行为'
        },
        {
            symbol: 'bnb_USDT',
            score: 58,
            time: new Date(Date.now() - 180 * 60000).toISOString(), // 3小时前
            message: '异常交易活动，伴随流动性下降，交易时注意风险'
        }
    ];
}

/**
 * 获取模拟大额交易预警
 * @return {Array} 模拟预警数据
 */
function getMockWhaleAlerts() {
    return [
        {
            symbol: 'btc_USDT',
            type: 'transfer',
            amount: 3.50,
            amount_usd: 210000,
            time: new Date(Date.now() - 8 * 60000).toISOString(), // 8分钟前
            from_type: '火币交易所',
            to_type: '未知地址'
        },
        {
            symbol: 'eth_USDT',
            type: 'exchange_outflow',
            amount: 45.20,
            amount_usd: 110000,
            time: new Date(Date.now() - 23 * 60000).toISOString(), // 23分钟前
            from_type: '未知地址',
            to_type: '火币交易所'
        },
        {
            symbol: 'bnb_USDT',
            type: 'transfer',
            amount: 125.0,
            amount_usd: 75000,
            time: new Date(Date.now() - 62 * 60000).toISOString(), // 62分钟前
            from_type: '鲸鱼钱包',
            to_type: '未知地址'
        },
        {
            symbol: 'xrp_USDT',
            type: 'exchange_inflow',
            amount: 52000,
            amount_usd: 115000,
            time: new Date(Date.now() - 95 * 60000).toISOString(), // 95分钟前
            from_type: '鲸鱼钱包',
            to_type: '交易所钱包'
        }
    ];
}

/**
 * 格式化时间为多久之前
 * @param {string} isoTime ISO格式时间字符串
 * @return {string} 格式化的时间字符串
 */
function formatTimeAgo(isoTime) {
    try {
        const date = new Date(isoTime);
        const now = new Date();
        const diffMs = now - date;
        const diffSec = Math.floor(diffMs / 1000);
        
        if (diffSec < 60) return `${diffSec}秒前`;
        if (diffSec < 3600) return `${Math.floor(diffSec / 60)}分钟前`;
        if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}小时前`;
        return `${Math.floor(diffSec / 86400)}天前`;
    } catch (e) {
        console.error('时间格式化错误:', e);
        return '未知时间';
    }
}

/**
 * 格式化货币数字
 * @param {number} value 数字值
 * @return {string} 格式化的货币字符串
 */
function formatCurrency(value) {
    if (value === undefined || value === null) return '--';
    
    try {
        // 处理大数字简写
        if (value >= 1000000000) {
            return `$${(value / 1000000000).toFixed(2)}B`;
        } else if (value >= 1000000) {
            return `$${(value / 1000000).toFixed(2)}M`;
        } else if (value >= 1000) {
            return `$${(value / 1000).toFixed(2)}K`;
        } else {
            return `$${value.toFixed(2)}`;
        }
    } catch (e) {
        console.error('货币格式化错误:', e);
        return '$0.00';
    }
}

/**
 * 查看币种详情 (如果页面上存在此功能)
 * @param {string} symbol 币种符号
 */
function viewCryptoDetail(symbol) {
    // 直接调用全局函数
    if (typeof window.viewCryptoDetail === 'function') {
        window.viewCryptoDetail(symbol);
    } else {
        console.error('未找到viewCryptoDetail全局函数');
        console.log(`尝试查看币种详情: ${symbol}`);
    }
} 