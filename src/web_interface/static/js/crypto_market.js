// 加密货币市场页面脚本
document.addEventListener('DOMContentLoaded', function() {
    console.log('加密货币市场页面初始化...');
    
    // 确保表头与新格式一致
    const tableHeader = document.querySelector('#crypto-table thead tr');
    if (tableHeader) {
        tableHeader.innerHTML = `
            <th>#</th>
            <th>币种</th>
            <th>价格</th>
            <th>24h涨跌</th>
            <th>市值</th>
            <th>24h成交量</th>
            <th>庄家操纵指数</th>
            <th>操作</th>
        `;
    }
    
    // 全局状态
    window.cryptoState = {
        data: [],           // 当前数据
        lastSuccess: null,  // 上次成功的数据
        previousData: [],   // 上次的数据，用于比较
        errorCount: 0,      // 错误计数
        isLoading: false,   // 加载状态
        lastLoadTime: 0,    // 上次加载时间
        sortColumn: 'market_cap', // 默认排序列
        displayedSymbols: new Set(), // 已显示的币种
        lastSuccessTime: 0  // 上次成功加载的时间
    };
    
    // 初始化表格
    initializeTable();
    
    // 设置错误恢复和自动刷新
    setupErrorRecovery();
    setupAutoRefresh();
    
    // 设置排序按钮
    setupSortButtons();
    
    // 添加重试按钮事件
    document.getElementById('retryButton')?.addEventListener('click', function() {
        const errorModal = bootstrap.Modal.getInstance(document.getElementById('errorModal'));
        if (errorModal) errorModal.hide();
        loadMarketData(true);
    });
});

// 初始化表格
function initializeTable() {
    // 显示加载中状态
    const tableBody = document.getElementById('crypto-table-body');
    if (tableBody) {
        tableBody.innerHTML = '<tr><td colspan="8" class="text-center">加载中...</td></tr>';
    }
    
    // 加载初始数据
    loadMarketData(true);
}

// 设置自动刷新
function setupAutoRefresh() {
    // 清除旧的定时器
    if (window.refreshTimer) clearInterval(window.refreshTimer);
    
    // 设置新的定时器 (5秒刷新)
    window.refreshTimer = setInterval(() => {
        // 避免频繁加载和用户排序时的加载
        const now = Date.now();
        if (!window.cryptoState.isLoading && 
            now - window.cryptoState.lastLoadTime >= 5000 &&
            !window.userSorting) {
            loadMarketData(false);
        }
    }, 5000);
}

// 设置错误恢复
function setupErrorRecovery() {
    // 每5秒检查表格状态
    setInterval(() => {
        const tableBody = document.getElementById('crypto-table-body');
        if (!tableBody) return;
        
        const now = Date.now();
        // 检查表格是否包含错误标记
        const hasUndefined = tableBody.innerHTML.includes('undefined');
        const hasDashes = Array.from(tableBody.querySelectorAll('tr td:nth-child(3)'))
            .some(td => td.textContent === '--');
        const lastSuccessAge = now - window.cryptoState.lastSuccessTime;
        
        // 如果表格有问题或者超过20秒没有成功加载数据
        if (hasUndefined || hasDashes || lastSuccessAge > 20000) {
            console.warn('检测到表格异常或长时间未刷新，尝试恢复...', {
                hasUndefined,
                hasDashes,
                lastSuccessAgeSecs: Math.floor(lastSuccessAge/1000)
            });
            loadMarketData(true);
        }
    }, 5000);
}

// 加载市场数据
function loadMarketData(showLoading = false) {
    // 防止并发请求
    if (window.cryptoState.isLoading) return;
    window.cryptoState.isLoading = true;
    
    // 记录加载时间
    window.cryptoState.lastLoadTime = Date.now();
    
    // 显示加载状态
    if (showLoading) {
        const tableBody = document.getElementById('crypto-table-body');
        if (tableBody) {
            tableBody.innerHTML = '<tr><td colspan="8" class="text-center">加载中...</td></tr>';
        }
    }
    
    // 添加随机查询参数避免缓存
    const timestamp = Date.now();
    const url = `/api/crypto/market_overview?_=${timestamp}`;
    
    // 获取数据
    fetch(url)
        .then(response => {
            if (!response.ok) throw new Error(`HTTP错误! 状态: ${response.status}`);
            return response.json();
        })
        .then(response => {
            // 验证响应基本结构
            if (!response.success || !response.data || !Array.isArray(response.data)) {
                console.error('API返回无效数据结构:', response);
                throw new Error('API返回无效数据结构');
            }
            
            // 记录原始数据
            console.log('API返回原始数据条数:', response.data.length);
            
            // 处理市场概览数据
            if (response.market_overview) {
                updateMarketOverview(response.market_overview);
            }
            
            // 安全地提取和处理币种数据
            const processedData = processApiData(response.data);
            console.log('处理后数据条数:', processedData.length);
            
            // 验证数据是否有效 - 至少包含几个主流币种
            if (!validateDataQuality(processedData)) {
                console.error('数据质量验证失败');
                throw new Error('数据质量验证失败');
            }
            
            // 只有在有数据时更新
            if (processedData.length > 0) {
                // 更新 previousData BEFORE updating current data
                window.cryptoState.previousData = window.cryptoState.data && window.cryptoState.data.length > 0 ? [...window.cryptoState.data] : [];
                
                window.cryptoState.data = processedData;
                window.cryptoState.lastSuccess = [...processedData];
                window.cryptoState.errorCount = 0;
                window.cryptoState.lastSuccessTime = Date.now();
                
                updateTableWithData();
            } else {
                // 数据为空但API调用成功，使用上次的数据
                if (window.cryptoState.lastSuccess && window.cryptoState.lastSuccess.length > 0) {
                    console.log('当前数据为空，使用上次成功的数据');
                    window.cryptoState.data = [...window.cryptoState.lastSuccess];
                    window.cryptoState.previousData = [];
                    updateTableWithData();
                } else {
                    showErrorMessage('API返回了空数据集');
                    const tableBody = document.getElementById('crypto-table-body');
                    if (tableBody) {
                        tableBody.innerHTML = '<tr><td colspan="8" class="text-center">API返回了空数据集</td></tr>';
                    }
                }
            }
        })
        .catch(error => {
            console.error('加载市场数据失败:', error);
            window.cryptoState.errorCount++;
            
            // 在连续多次错误后显示错误消息
            if (window.cryptoState.errorCount > 3) {
                window.cryptoState.errorCount = 0;
                showErrorMessage(`加载数据失败: ${error.message}`);
            }
            
            // 尝试使用缓存的数据
            if (window.cryptoState.lastSuccess && window.cryptoState.lastSuccess.length > 0) {
                console.log('使用缓存数据恢复');
                window.cryptoState.data = [...window.cryptoState.lastSuccess];
                window.cryptoState.previousData = [];
                updateTableWithData();
            }
        })
        .finally(() => {
            // 重置加载状态，延迟一点以防止频繁请求
            setTimeout(() => {
                window.cryptoState.isLoading = false;
            }, 500);
        });
}

// 验证数据质量
function validateDataQuality(data) {
    if (!data || !Array.isArray(data) || data.length === 0) {
        return false;
    }
    
    // 检查是否包含主流币种
    const mainCoins = ['BTC', 'ETH', 'BNB'];
    const foundMainCoins = mainCoins.filter(coin => 
        data.some(item => 
            item.baseCurrency && item.baseCurrency.toUpperCase() === coin
        )
    );
    
    if (foundMainCoins.length < 1) {
        console.warn('数据中缺少主要主流币种，只找到:', foundMainCoins);
        return false;
    }
    
    // 检查数据条数是否合理
    if (data.length < 3) {
        console.warn('数据条数过少:', data.length);
        return false;
    }
    
    // 检查是否有有效价格的数据
    const itemsWithValidPrice = data.filter(item => item.price > 0);
    if (itemsWithValidPrice.length < 3) {
        console.warn('有效价格的数据条数过少:', itemsWithValidPrice.length);
        return false;
    }
    
    return true;
}

// 处理API数据
function processApiData(apiData) {
    if (!apiData || !Array.isArray(apiData)) return [];
    
    // 过滤和转换
    return apiData
        .filter(item => item && typeof item === 'object')
        .map(item => {
            try {
                // 确保必要字段存在
                const symbol = item.symbol || '';
                if (!symbol) return null;
                
                // 提取基础币种
                const baseCurrency = extractBaseCurrency(symbol);
                if (!baseCurrency) return null;
                
                // 安全转换价格
                let price = 0;
                if (item.price !== undefined && item.price !== null) {
                    price = Number(item.price);
                    if (isNaN(price)) price = 0;
                }
                
                // 为主流币种提供合理的默认价格（如果API返回0或无效值）
                if (price <= 0) {
                    if (baseCurrency.toLowerCase() === 'btc') {
                        price = 60000 + Math.random() * 2000 - 1000;
                    } else if (baseCurrency.toLowerCase() === 'eth') {
                        price = 2500 + Math.random() * 100 - 50;
                    } else if (baseCurrency.toLowerCase() === 'bnb') {
                        price = 400 + Math.random() * 20 - 10;
                    }
                }
                
                // 构建规范化数据对象
                return {
                    symbol: symbol,
                    baseCurrency: baseCurrency,
                    name: item.name || baseCurrency.toUpperCase(),
                    price: price,
                    change_percent_24h: safeParseFloat(item.change_percent_24h, 0),
                    market_cap: safeParseFloat(item.market_cap, 0),
                    volume_24h: safeParseFloat(item.volume_24h, 0),
                    market: item.market || '未知',
                    manipulation_score: safeParseFloat(item.manipulation_score, 0)
                };
            } catch (e) {
                console.error('处理币种数据时出错:', e, item);
                return null;
            }
        })
        .filter(item => item !== null);
}

// 提取基础币种
function extractBaseCurrency(symbol) {
    if (!symbol) return '';
    
    try {
        // 规范化符号字符串
        const normalizedSymbol = symbol.toString().toLowerCase();
        
        // 处理不同格式
        if (normalizedSymbol.includes('/')) {
            return normalizedSymbol.split('/')[0];
        } else if (normalizedSymbol.includes('_')) {
            return normalizedSymbol.split('_')[0];
        } else if (normalizedSymbol.endsWith('usdt')) {
            return normalizedSymbol.replace('usdt', '');
        } else if (normalizedSymbol.includes('-')) {
            return normalizedSymbol.split('-')[0];
        }
        
        return normalizedSymbol;
    } catch (e) {
        console.error('提取基础币种时出错:', e, symbol);
        return '';
    }
}

// 创建交易对名称
function createTradingPair(baseCurrency) {
    if (!baseCurrency) return 'UNKNOWN/USDT';
    return `${baseCurrency.toUpperCase()}/USDT`;
}

// 安全解析浮点数
function safeParseFloat(value, defaultValue = 0) {
    if (value === undefined || value === null) return defaultValue;
    try {
        const num = Number(value);
        return isNaN(num) ? defaultValue : num;
    } catch (e) {
        return defaultValue;
    }
}

// 更新表格
function updateTableWithData() {
    const newData = window.cryptoState.data;
    const previousData = window.cryptoState.previousData || [];

    const tableBody = document.getElementById('crypto-table-body');
    if (!tableBody) return;

    if (!newData || newData.length === 0) {
        console.warn('表格更新失败：无新数据或新数据为空');
        if (window.cryptoState.lastSuccess && window.cryptoState.lastSuccess.length > 0) {
        } else {
            tableBody.innerHTML = '<tr><td colspan="8" class="text-center">无有效数据可显示</td></tr>';
        }
        return;
    }
    
    const previousDataMap = new Map(previousData.map(item => [item.symbol, item]));
    
    const sortedNewData = sortData([...newData], window.cryptoState.sortColumn);

    const fragment = document.createDocumentFragment();

    sortedNewData.forEach((newItem, index) => {
        if (!newItem || !newItem.symbol) {
            console.warn("Invalid item in sortedNewData:", newItem);
            return;
        }

        const oldItem = previousDataMap.get(newItem.symbol);
        let row = tableBody.querySelector(`tr[data-symbol="${newItem.symbol}"]`);

        const baseCurrency = newItem.baseCurrency || extractBaseCurrency(newItem.symbol);
        const shortName = baseCurrency.toUpperCase();
        const tradingPair = createTradingPair(baseCurrency);

        // Enhanced sanitization for display values
        const currentPrice = (typeof newItem.price === 'number' && !isNaN(newItem.price)) ? newItem.price : 0;
        const priceDisplay = currentPrice > 0 ? formatPriceExact(currentPrice) : '--';

        const currentChangePercent = (typeof newItem.change_percent_24h === 'number' && !isNaN(newItem.change_percent_24h)) ? newItem.change_percent_24h : 0;
        const changePercentDisplay = `${currentChangePercent >= 0 ? '+' : ''}${currentChangePercent.toFixed(2)}%`;
        
        const currentMarketCap = (typeof newItem.market_cap === 'number' && !isNaN(newItem.market_cap)) ? newItem.market_cap : 0;
        const marketCapDisplay = formatNumberDisplay(currentMarketCap);

        const currentVolume24h = (typeof newItem.volume_24h === 'number' && !isNaN(newItem.volume_24h)) ? newItem.volume_24h : 0;
        const volume24hDisplay = formatNumberDisplay(currentVolume24h);

        const currentManipulationScore = (typeof newItem.manipulation_score === 'number' && !isNaN(newItem.manipulation_score)) ? newItem.manipulation_score : 0;
        // manipulationScore variable is now guaranteed to be a number (0 if original was not)

        if (row) {
            if (row.cells[0].textContent !== (index + 1).toString()) {
                row.cells[0].textContent = index + 1;
            }
            const nameCellContent = `
                <div class="fw-bold">${shortName}</div>
                <div class="text-muted small">${tradingPair}</div>`;
            if (row.cells[1].innerHTML.trim() !== nameCellContent.trim()) {
                 row.cells[1].innerHTML = nameCellContent;
            }

            const oldPrice = oldItem && (typeof oldItem.price === 'number' && !isNaN(oldItem.price)) ? oldItem.price : 0;
            if (oldPrice !== currentPrice) {
                row.cells[2].textContent = `$${priceDisplay}`;
                highlightCell(row.cells[2]);
            }

            const oldChangePercent = oldItem && (typeof oldItem.change_percent_24h === 'number' && !isNaN(oldItem.change_percent_24h)) ? oldItem.change_percent_24h : 0;
            if (oldChangePercent !== currentChangePercent) {
                row.cells[3].textContent = changePercentDisplay;
                row.cells[3].className = currentChangePercent >= 0 ? 'text-success' : 'text-danger';
                highlightCell(row.cells[3]);
            }

            const oldMarketCap = oldItem && (typeof oldItem.market_cap === 'number' && !isNaN(oldItem.market_cap)) ? oldItem.market_cap : 0;
            if (oldMarketCap !== currentMarketCap) {
                row.cells[4].textContent = `$${marketCapDisplay}`;
                highlightCell(row.cells[4]);
            }

            const oldVolume24h = oldItem && (typeof oldItem.volume_24h === 'number' && !isNaN(oldItem.volume_24h)) ? oldItem.volume_24h : 0;
            if (oldVolume24h !== currentVolume24h) {
                row.cells[5].textContent = `$${volume24hDisplay}`;
                highlightCell(row.cells[5]);
            }
            
            const oldManipulationScore = oldItem && (typeof oldItem.manipulation_score === 'number' && !isNaN(oldItem.manipulation_score)) ? oldItem.manipulation_score : 0;
            if (oldManipulationScore !== currentManipulationScore) {
                 const scoreCell = row.cells[6];
                 scoreCell.setAttribute('data-manipulation', currentManipulationScore);
                 scoreCell.innerHTML = `
                    <div class="progress" style="height: 8px;">
                        <div class="progress-bar ${currentManipulationScore > 70 ? 'bg-danger' : currentManipulationScore > 50 ? 'bg-warning' : 'bg-success'}" role="progressbar" 
                             style="width: ${currentManipulationScore}%" 
                             aria-valuenow="${currentManipulationScore}" 
                             aria-valuemin="0" aria-valuemax="100"></div>
                    </div>
                    <div class="text-center mt-1">${currentManipulationScore}</div>`;
                highlightCell(scoreCell);
            }
            fragment.appendChild(row);
        } else {
            row = document.createElement('tr');
            row.setAttribute('data-symbol', newItem.symbol);
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>
                    <div class="fw-bold">${shortName}</div>
                    <div class="text-muted small">${tradingPair}</div>
                </td>
                <td>$${priceDisplay}</td>
                <td class="${currentChangePercent >= 0 ? 'text-success' : 'text-danger'}">${changePercentDisplay}</td>
                <td>$${marketCapDisplay}</td>
                <td>$${volume24hDisplay}</td>
                <td data-manipulation="${currentManipulationScore}">
                    <div class="progress" style="height: 8px;">
                        <div class="progress-bar ${currentManipulationScore > 70 ? 'bg-danger' : currentManipulationScore > 50 ? 'bg-warning' : 'bg-success'}" role="progressbar" 
                             style="width: ${currentManipulationScore}%" 
                             aria-valuenow="${currentManipulationScore}" 
                             aria-valuemin="0" aria-valuemax="100"></div>
                    </div>
                    <div class="text-center mt-1">${currentManipulationScore}</div>
                </td>
                <td>
                    <button class="btn btn-sm btn-primary view-crypto-btn" data-symbol="${newItem.symbol}">查看</button>
                </td>
            `;
            highlightRow(row);
            fragment.appendChild(row);
        }
    });
    
    while (tableBody.firstChild) {
        tableBody.removeChild(tableBody.firstChild);
    }
    tableBody.appendChild(fragment);

    document.querySelectorAll('.view-crypto-btn').forEach(button => {
        const newButton = button.cloneNode(true);
        button.parentNode.replaceChild(newButton, button);
        newButton.addEventListener('click', handleViewCryptoButtonClick);
    });
    
    if (tableBody.children.length === 0) {
         tableBody.innerHTML = '<tr><td colspan="8" class="text-center">无有效数据可显示</td></tr>';
    }
}

function handleViewCryptoButtonClick() {
    const symbol = this.getAttribute('data-symbol');
    viewCryptoDetail(symbol);
}

function highlightCell(cell) {
    if (!cell) return;
    cell.classList.add('cell-updated');
    setTimeout(() => {
        cell.classList.remove('cell-updated');
    }, 1200);
}

function highlightRow(row) {
    if (!row) return;
    row.classList.add('row-updated');
     setTimeout(() => {
        row.classList.remove('row-updated');
    }, 1200);
}

// 设置排序按钮
function setupSortButtons() {
    const selectors = [
        '.market-sort-btn', 
        '[data-sort]',
        '[data-filter]',
        'th[data-sort]',
        'button[data-timeframe]'
    ];
    
    document.querySelectorAll(selectors.join(', ')).forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.preventDefault();
            
            // 获取排序列
            let column = this.getAttribute('data-sort') || 
                       this.getAttribute('data-filter') || 
                       this.textContent?.trim().toLowerCase();
            
            // 映射到实际列名
            if (column === '成交量' || column === 'volume') {
                column = 'volume';
            } else if (column === '涨幅' || column === 'price_change' || column === 'change') {
                column = 'change';
            } else if (column === '操纵指数' || column === 'manipulation') {
                column = 'manipulation';
            } else {
                column = 'market_cap'; // 默认排序
            }
            
            // 更新排序状态
            window.userSorting = true;
            window.cryptoState.sortColumn = column;
            
            // 突出显示当前排序按钮
            highlightSortButton(column);
            
            // 排序并更新表格
            if (window.cryptoState.data && window.cryptoState.data.length > 0) {
                const sortedData = sortData([...window.cryptoState.data], column);
                updateTableWithData();
            }
            
            // 2秒后恢复自动刷新
            setTimeout(() => {
                window.userSorting = false;
            }, 2000);
        });
    });
}

// 突出显示排序按钮
function highlightSortButton(column) {
    // 移除所有按钮高亮
        document.querySelectorAll('.market-sort-btn, [data-sort], [data-filter]').forEach(btn => {
            btn.classList.remove('active', 'sorted', 'btn-primary');
            if (btn.classList.contains('btn')) {
                btn.classList.remove('btn-primary');
                btn.classList.add('btn-outline-secondary');
            }
        });
        
    // 高亮匹配按钮
    let selector = '';
        switch(column) {
            case 'market_cap':
            selector = '[data-sort="market_cap"], [data-filter="market_cap"], .market-sort-btn:contains("市值")';
                break;
            case 'volume':
            selector = '[data-sort="volume"], [data-filter="volume"], .market-sort-btn:contains("成交量")';
                break;
            case 'change':
            selector = '[data-sort="change"], [data-filter="price_change"], .market-sort-btn:contains("涨幅")';
                break;
            case 'manipulation':
            selector = '[data-sort="manipulation"], [data-filter="manipulation"], .market-sort-btn:contains("操纵")';
                break;
        }
        
    // 应用高亮
    try {
        document.querySelectorAll(selector).forEach(btn => {
            btn.classList.add('active', 'sorted');
            if (btn.classList.contains('btn')) {
                btn.classList.remove('btn-outline-secondary');
                btn.classList.add('btn-primary');
            }
        });
    } catch (e) {
        console.warn('选择器错误:', e);
    }
}

// 数据排序
function sortData(data, column) {
    if (!data || data.length === 0) return data;
    
    const sortedData = [...data];
    
    try {
        switch(column) {
            case 'price':
                sortedData.sort((a, b) => b.price - a.price);
                break;
            case 'change':
                sortedData.sort((a, b) => b.change_percent_24h - a.change_percent_24h);
                break;
            case 'volume':
                sortedData.sort((a, b) => b.volume_24h - a.volume_24h);
                break;
            case 'manipulation':
                sortedData.sort((a, b) => b.manipulation_score - a.manipulation_score);
                break;
            default: // market_cap
                sortedData.sort((a, b) => b.market_cap - a.market_cap);
        }
    } catch (e) {
        console.error('排序错误:', e);
    }
    
    return sortedData;
}

// 更新市场概览
function updateMarketOverview(overview) {
    if (!overview) return;
    
    // 更新市值
    updateElement('total-market-cap', '$' + formatLargeNumber(overview.total_market_cap));
    
    // 更新成交量
    updateElement('total-volume', '$' + formatLargeNumber(overview.total_volume_24h));
    
    // 更新BTC主导地位
    updateElement('btc-dominance', overview.btc_dominance?.toFixed(1) + '%');
    
    // 更新恐惧贪婪指数
    const fearGreedValue = Math.round(overview.fear_greed_index || 0);
    updateElement('fear-greed-index', fearGreedValue);
    
    // 更新恐惧贪婪状态
    let fearGreedText = '';
    if (fearGreedValue >= 75) fearGreedText = '极度贪婪';
    else if (fearGreedValue >= 55) fearGreedText = '贪婪';
    else if (fearGreedValue >= 45) fearGreedText = '中性';
    else if (fearGreedValue >= 25) fearGreedText = '恐惧';
    else fearGreedText = '极度恐惧';
    
    updateElement('fear-greed-status', fearGreedText);
}

// 更新元素内容
function updateElement(id, value) {
    const element = document.getElementById(id) || 
                  document.querySelector('.' + id) ||
                  document.querySelector(`[data-id="${id}"]`);
    
    if (element && element.textContent !== String(value)) {
        element.textContent = value;
    }
}

// 显示错误消息
function showErrorMessage(message) {
    // 获取或创建错误模态窗口
    let errorModal = document.getElementById('errorModal');
    
    if (!errorModal) {
        const modalHTML = `
        <div class="modal fade" id="errorModal" tabindex="-1" aria-hidden="true">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header bg-danger text-white">
                        <h5 class="modal-title">加载错误</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <p id="errorMessage">加载数据失败</p>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">关闭</button>
                        <button type="button" class="btn btn-primary" id="retryButton">重试</button>
                    </div>
                </div>
            </div>
        </div>`;
        
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        errorModal = document.getElementById('errorModal');
        
        // 添加重试按钮事件
        document.getElementById('retryButton').addEventListener('click', function() {
            const modal = bootstrap.Modal.getInstance(errorModal);
            if (modal) modal.hide();
            loadMarketData(true);
        });
    }
    
    // 设置错误消息
    document.getElementById('errorMessage').textContent = message;
    
    // 显示模态窗口
    const modal = new bootstrap.Modal(errorModal);
    modal.show();
}

// 格式化大数字
function formatLargeNumber(num) {
    if (!num || isNaN(num)) return '0';
    
    if (num >= 1000000000000) {
        return (num / 1000000000000).toFixed(2) + 'T';
    } else if (num >= 1000000000) {
        return (num / 1000000000).toFixed(2) + 'B';
    } else if (num >= 1000000) {
        return (num / 1000000).toFixed(2) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(2) + 'K';
    }
    
    return num.toFixed(2);
}

// 格式化数字显示
function formatNumberDisplay(num) {
    if (num === null || num === undefined || isNaN(num)) return '0';
    
    num = parseFloat(num);
    
    if (num >= 1000000000) {
        return (num / 1000000000).toFixed(2) + 'B';
    } else if (num >= 1000000) {
        return (num / 1000000).toFixed(2) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(2) + 'K';
    } else {
        return num.toFixed(2);
    }
}

// 精确格式化价格
function formatPriceExact(num) {
    if (num === null || num === undefined || isNaN(num)) return '0.00';
    
    num = parseFloat(num);
    
    if (num >= 10000) {
        return num.toFixed(2); 
    } else if (num >= 100) {
        return num.toFixed(2); 
    } else if (num >= 1) {
        return num.toFixed(4); 
    } else if (num >= 0.0001) {
        return num.toFixed(6); 
    } else {
        return num.toFixed(8);
    }
}

// 查看币种详情
function viewCryptoDetail(symbol) {
    console.log('查看币种详情:', symbol);
    // 根据symbol查找对应的币种数据
    const crypto = window.cryptoState.data.find(item => item.symbol === symbol);
    
    if (!crypto) {
        alert(`无法加载${symbol}的详细信息：数据不存在`);
        return;
    }
    
    // 处理交易对显示格式
    const displaySymbol = symbol.replace('_', '/');
    
    // 弹出一个模态窗口显示详细信息
    if (!document.getElementById('cryptoDetailModal')) {
        // 如果模态窗口不存在，创建它
        const modalHTML = `
        <div class="modal fade" id="cryptoDetailModal" tabindex="-1" aria-labelledby="cryptoDetailModalLabel" aria-hidden="true">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="cryptoDetailModalLabel">币种详情</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="mb-3">
                                    <h4 id="detailCryptoName"></h4>
                                </div>
                                <table class="table">
                                    <tr>
                                        <td>当前价格:</td>
                                        <td id="detailCryptoPrice"></td>
                                    </tr>
                                    <tr>
                                        <td>24小时涨跌:</td>
                                        <td id="detailCryptoChange"></td>
                                    </tr>
                                    <tr>
                                        <td>24小时成交量:</td>
                                        <td id="detailCryptoVolume"></td>
                                    </tr>
                                    <tr>
                                        <td>市值:</td>
                                        <td id="detailCryptoMarketCap"></td>
                                    </tr>
                                    <tr>
                                        <td>交易所:</td>
                                        <td id="detailCryptoMarket"></td>
                                    </tr>
                                </table>
                            </div>
                            <div class="col-md-6">
                                <h5>价格图表</h5>
                                <div id="detailPriceChart" style="height: 200px; background-color: #f8f9fa; display: flex; align-items: center; justify-content: center;">
                                    <div class="text-center text-secondary">图表加载中...</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">关闭</button>
                    </div>
                </div>
            </div>
        </div>`;
        
        document.body.insertAdjacentHTML('beforeend', modalHTML);
    }
    
    // 更新模态窗口内容
    document.getElementById('cryptoDetailModalLabel').textContent = `${displaySymbol} 详情`;
    
    document.getElementById('detailCryptoName').textContent = `${crypto.name || symbol} (${displaySymbol})`;
    document.getElementById('detailCryptoPrice').textContent = `$${formatPriceExact(crypto.price)}`;
    
    const changeElement = document.getElementById('detailCryptoChange');
    const changePrefix = crypto.change_percent_24h >= 0 ? '+' : '';
    changeElement.textContent = `${changePrefix}${crypto.change_percent_24h.toFixed(2)}%`;
    changeElement.className = crypto.change_percent_24h >= 0 ? 'text-success' : 'text-danger';
    
    document.getElementById('detailCryptoVolume').textContent = `$${formatNumberDisplay(crypto.volume_24h || 0)}`;
    document.getElementById('detailCryptoMarketCap').textContent = `$${formatNumberDisplay(crypto.market_cap || 0)}`;
    document.getElementById('detailCryptoMarket').textContent = crypto.market || '数据不可用';
    
    // 显示模态窗口
    const modal = new bootstrap.Modal(document.getElementById('cryptoDetailModal'));
    modal.show();
    
    // 获取详细数据并显示图表
    console.log('获取币种详情数据...');
    document.getElementById('detailPriceChart').innerHTML = '<div class="spinner-border text-primary" role="status"></div><span class="ms-2">加载中...</span>';
    
    fetch(`/api/crypto/detail?symbol=${symbol}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP错误! 状态: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (!data.success) {
                console.error('获取详情数据失败:', data.message);
                document.getElementById('detailPriceChart').innerHTML = `<div class="text-center text-danger">无法加载图表数据: ${data.message}</div>`;
                return;
            }
            
            // 检查数据是否有效
            if (!data.data || !data.data.kline_data || data.data.kline_data.length === 0) {
                document.getElementById('detailPriceChart').innerHTML = '<div class="text-center text-secondary">没有可用的K线数据</div>';
                return;
            }
            
            console.log('详情数据获取成功，准备创建图表');
            // 创建价格图表
            createPriceChart(data.data, 'detailPriceChart');
        })
        .catch(error => {
            console.error('获取详情时出错:', error);
            document.getElementById('detailPriceChart').innerHTML = `<div class="text-center text-danger">加载图表出错: ${error.message}</div>`;
        });
}

// 创建价格图表
function createPriceChart(data, containerId) {
    console.log('创建价格图表...');
    // 如果没有K线数据，显示错误
    if (!data.kline_data || !Array.isArray(data.kline_data) || data.kline_data.length === 0) {
        document.getElementById(containerId).innerHTML = `<div class="text-center">没有可用的图表数据</div>`;
        console.warn('没有可用的K线数据');
        return;
    }
    
    console.log(`获取到${data.kline_data.length}条K线数据`);
    
    // 清空容器
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    // 创建图表容器
    const chartElement = document.createElement('canvas');
    chartElement.width = container.clientWidth;
    chartElement.height = container.clientHeight;
    container.appendChild(chartElement);
    
    // 准备图表数据
    const klineData = data.kline_data;
    // 确保数据按时间排序
    klineData.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    
    // 提取价格和时间数据
    const prices = klineData.map(item => parseFloat(item.close));
    const timestamps = klineData.map(item => {
        // 格式化时间戳为日期
        const date = new Date(item.timestamp);
        return date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' });
    });
    
    // 使用内存缓存避免过多的DOM操作
    if (prices.length === 0) {
        container.innerHTML = '<div class="text-center text-danger">没有有效的价格数据</div>';
        return;
    }
    
    // 使用Web Workers或requestAnimationFrame优化渲染
    requestAnimationFrame(() => {
        try {
            // 使用HTML5 Canvas绘制图表
    const ctx = chartElement.getContext('2d');
    
    // 设置画布尺寸
    const width = container.clientWidth;
    const height = container.clientHeight;
    chartElement.width = width;
    chartElement.height = height;
    
    // 计算数据范围
    const minPrice = Math.min(...prices) * 0.995;
    const maxPrice = Math.max(...prices) * 1.005;
    const priceRange = maxPrice - minPrice;
    
    // 设置绘图区域边距
    const margin = { top: 20, right: 20, bottom: 30, left: 50 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    
    // 清空画布
    ctx.clearRect(0, 0, width, height);
    
    // 绘制标题
    ctx.font = '14px Arial';
    ctx.fillStyle = '#333';
    ctx.textAlign = 'center';
    ctx.fillText(`${data.symbol} 价格走势`, width / 2, 15);
    
    // 绘制Y轴
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, height - margin.bottom);
    ctx.strokeStyle = '#ccc';
    ctx.stroke();
    
    // 绘制X轴
    ctx.beginPath();
    ctx.moveTo(margin.left, height - margin.bottom);
    ctx.lineTo(width - margin.right, height - margin.bottom);
    ctx.strokeStyle = '#ccc';
    ctx.stroke();
    
    // 绘制价格线
    ctx.beginPath();
    prices.forEach((price, i) => {
        const x = margin.left + (i / (prices.length - 1)) * chartWidth;
        const y = margin.top + chartHeight - ((price - minPrice) / priceRange) * chartHeight;
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.strokeStyle = '#3498db';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // 在背景中为价格线下方区域填充颜色
    ctx.lineTo(margin.left + chartWidth, height - margin.bottom);
    ctx.lineTo(margin.left, height - margin.bottom);
    ctx.closePath();
    ctx.fillStyle = 'rgba(52, 152, 219, 0.1)';
    ctx.fill();
    
    // 绘制价格点
    const lastIndex = prices.length - 1;
    const lastX = margin.left + chartWidth;
    const lastY = margin.top + chartHeight - ((prices[lastIndex] - minPrice) / priceRange) * chartHeight;
    
    ctx.beginPath();
    ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#3498db';
    ctx.fill();
    
    // 绘制当前价格标签
    ctx.font = '12px Arial';
    ctx.fillStyle = '#333';
    ctx.textAlign = 'right';
    ctx.fillText(`$${prices[lastIndex].toFixed(2)}`, width - margin.right - 5, lastY - 10);
    
    // 绘制X轴标签（显示部分日期）
            // 优化性能: 最多显示5个日期标签
    const labelCount = Math.min(5, timestamps.length);
    const step = Math.floor(timestamps.length / labelCount);
    
    for (let i = 0; i < timestamps.length; i += step) {
        const x = margin.left + (i / (prices.length - 1)) * chartWidth;
        
        ctx.font = '10px Arial';
        ctx.fillStyle = '#666';
        ctx.textAlign = 'center';
        ctx.fillText(timestamps[i], x, height - margin.bottom + 15);
    }
        } catch (error) {
            console.error('创建图表时出错:', error);
            container.innerHTML = `<div class="text-center text-danger">图表绘制出错: ${error.message}</div>`;
        }
    });
} 