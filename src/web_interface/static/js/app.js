/**
 * 跟随庄家自动交易系统 - 核心前端脚本
 * 
 * 处理全局UI交互和数据处理功能
 * 
 * @version 4.0.0
 * @date 2025-05-23
 */

// 使用严格模式提高代码质量
'use strict';

// 全局应用对象
const StarApp = {
    /**
     * 初始化应用
     */
    init() {
        console.log('跟随庄家自动交易系统已加载');
        
        // 初始化各组件
        this.initializeAlgorithmParams();
        this.initializeEventListeners();
        this.initializeSystemStatus();
    },

    /**
     * 初始化算法参数设置
     */
    initializeAlgorithmParams() {
        // 获取算法选择和参数容器元素
        const algorithmSelect = document.getElementById('algorithm_select');
        const paramsContainer = document.getElementById('algorithm_params_container');
        const optimizationTargetsSelect = document.getElementById('optimization_target');
        
        // 如果相关元素不存在，直接返回
        if (!algorithmSelect || !paramsContainer || !optimizationTargetsSelect) return;
        
        // 定义算法参数
        const algorithms = {
            'smart_money_detector': {
                name: '庄家行为识别器',
                params: [
                    { id: 'manipulation_score_threshold', name: '操纵评分阈值', default: 70, min: 0, max: 100, step: 1, description: '触发庄家操纵警报的最小分数阈值' },
                    { id: 'volume_threshold', name: '成交量异常阈值', default: 3.0, min: 1.0, max: 10.0, step: 0.1, description: '标记成交量异常的标准差倍数' },
                    { id: 'price_volatility_threshold', name: '价格波动阈值', default: 0.03, min: 0.01, max: 0.1, step: 0.01, description: '价格波动率的阈值' }
                ],
                optimization_targets: ['操纵评分精确度', '盈亏比', '胜率', '夏普比率']
            },
            'volume_analyzer': {
                name: '成交量分析器',
                params: [
                    { id: 'volume_threshold', name: '成交量异常阈值', default: 3.0, min: 1.0, max: 10.0, step: 0.1, description: '标记成交量异常的标准差倍数' },
                    { id: 'volume_window', name: '成交量分析窗口', default: 20, min: 5, max: 60, step: 1, description: '成交量分析的历史数据窗口大小' }
                ],
                optimization_targets: ['成交量识别准确率', '胜率', '盈亏比']
            },
            'price_pattern_detector': {
                name: '价格模式识别器',
                params: [
                    { id: 'price_volatility_threshold', name: '价格波动阈值', default: 0.03, min: 0.01, max: 0.1, step: 0.01, description: '价格波动率的阈值' },
                    { id: 'price_manipulation_window', name: '价格操纵检测窗口', default: 20, min: 5, max: 60, step: 1, description: '价格操纵检测的历史数据窗口大小' },
                    { id: 'breakout_threshold', name: '突破阈值', default: 0.02, min: 0.005, max: 0.05, step: 0.005, description: '标记价格突破的百分比阈值' }
                ],
                optimization_targets: ['模式识别准确率', '胜率', '夏普比率', '最大回撤']
            }
        };
        
        // 绑定算法选择变更事件
        if (algorithmSelect) {
            algorithmSelect.addEventListener('change', () => this.updateParamsForm(algorithmSelect, paramsContainer, optimizationTargetsSelect, algorithms));
            
            // 初始化参数表单
            if (algorithmSelect.value) {
                this.updateParamsForm(algorithmSelect, paramsContainer, optimizationTargetsSelect, algorithms);
            }
        }
        
        // 为已有的优化参数复选框添加事件监听
        document.querySelectorAll('input[name="optimize_params"]').forEach(checkbox => {
            checkbox.addEventListener('change', this.toggleOptimizationRange);
        });
    },
    
    /**
     * 初始化全局事件监听器
     */
    initializeEventListeners() {
        // 添加表单提交验证
        const forms = document.querySelectorAll('form[data-validate="true"]');
        forms.forEach(form => {
            form.addEventListener('submit', this.validateForm);
        });
        
        // 初始化提示工具
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        if (typeof bootstrap !== 'undefined') {
            tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        }
        
        // 初始化回到顶部按钮
        const backToTopBtn = document.getElementById('back-to-top');
        if (backToTopBtn) {
            window.addEventListener('scroll', () => {
                if (window.pageYOffset > 300) {
                    backToTopBtn.classList.add('show');
                } else {
                    backToTopBtn.classList.remove('show');
                }
            });
            
            backToTopBtn.addEventListener('click', (e) => {
                e.preventDefault();
                window.scrollTo({ top: 0, behavior: 'smooth' });
            });
        }
    },
    
    /**
     * 初始化系统状态检查
     */
    initializeSystemStatus() {
        // 如果在仪表盘页面，定期更新系统状态
        const statusElement = document.querySelector('.system-status');
        if (statusElement) {
            this.updateSystemStatus();
            // 每60秒更新一次系统状态
            setInterval(() => this.updateSystemStatus(), 60000);
        }
    },
    
    /**
     * 更新参数表单
     * @param {HTMLElement} algorithmSelect - 算法选择下拉框
     * @param {HTMLElement} paramsContainer - 参数容器
     * @param {HTMLElement} optimizationTargetsSelect - 优化目标选择框
     * @param {Object} algorithms - 算法定义
     */
    updateParamsForm(algorithmSelect, paramsContainer, optimizationTargetsSelect, algorithms) {
        const selectedAlgorithm = algorithmSelect.value;
        const algorithm = algorithms[selectedAlgorithm] || null;
        
        // 清空参数容器
        paramsContainer.innerHTML = '';
        
        // 如果没有选择算法或算法不存在，则返回
        if (!algorithm) return;
        
        // 创建参数表单
        const paramsList = document.createElement('div');
        paramsList.className = 'mt-3';
        
        // 添加算法参数
        algorithm.params.forEach(param => {
            const paramRow = document.createElement('div');
            paramRow.className = 'mb-3 row';
            
            // 参数标签
            const labelCol = document.createElement('div');
            labelCol.className = 'col-sm-4';
            const label = document.createElement('label');
            label.htmlFor = param.id;
            label.className = 'form-label';
            label.textContent = param.name;
            const description = document.createElement('small');
            description.className = 'form-text text-muted d-block';
            description.textContent = param.description;
            labelCol.appendChild(label);
            labelCol.appendChild(description);
            
            // 参数输入
            const inputCol = document.createElement('div');
            inputCol.className = 'col-sm-8';
            const inputGroup = document.createElement('div');
            inputGroup.className = 'input-group';
            
            // 数值输入框
            const input = document.createElement('input');
            input.type = 'number';
            input.className = 'form-control';
            input.id = param.id;
            input.name = `algorithm_params[${param.id}]`;
            input.value = param.default;
            input.min = param.min;
            input.max = param.max;
            input.step = param.step;
            
            // 参数优化复选框
            const optimizeCheckCol = document.createElement('div');
            optimizeCheckCol.className = 'input-group-append';
            const optimizeCheck = document.createElement('div');
            optimizeCheck.className = 'input-group-text';
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.className = 'form-check-input mt-0';
            checkbox.id = `optimize_${param.id}`;
            checkbox.name = `optimize_params`;
            checkbox.value = param.id;
            checkbox.style.marginLeft = '8px';
            // 添加事件监听
            checkbox.addEventListener('change', this.toggleOptimizationRange);
            const checkLabel = document.createElement('label');
            checkLabel.className = 'form-check-label ms-2';
            checkLabel.htmlFor = `optimize_${param.id}`;
            checkLabel.textContent = '优化此参数';
            optimizeCheck.appendChild(checkbox);
            optimizeCheck.appendChild(checkLabel);
            optimizeCheckCol.appendChild(optimizeCheck);
            
            // 添加所有元素
            inputGroup.appendChild(input);
            inputGroup.appendChild(optimizeCheckCol);
            inputCol.appendChild(inputGroup);
            
            // 优化范围设置
            const rangeRow = document.createElement('div');
            rangeRow.className = 'mt-2 optimization-range d-none';
            rangeRow.id = `range_${param.id}`;
            
            const rangeInputs = document.createElement('div');
            rangeInputs.className = 'row g-2';
            
            // 最小值
            const minCol = document.createElement('div');
            minCol.className = 'col-md-4';
            const minGroup = document.createElement('div');
            minGroup.className = 'input-group input-group-sm';
            const minLabel = document.createElement('span');
            minLabel.className = 'input-group-text';
            minLabel.textContent = '最小值';
            const minInput = document.createElement('input');
            minInput.type = 'number';
            minInput.className = 'form-control';
            minInput.name = `optimize_range[${param.id}][min]`;
            minInput.value = param.min;
            minInput.min = param.min;
            minInput.max = param.max;
            minInput.step = param.step;
            minGroup.appendChild(minLabel);
            minGroup.appendChild(minInput);
            minCol.appendChild(minGroup);
            
            // 最大值
            const maxCol = document.createElement('div');
            maxCol.className = 'col-md-4';
            const maxGroup = document.createElement('div');
            maxGroup.className = 'input-group input-group-sm';
            const maxLabel = document.createElement('span');
            maxLabel.className = 'input-group-text';
            maxLabel.textContent = '最大值';
            const maxInput = document.createElement('input');
            maxInput.type = 'number';
            maxInput.className = 'form-control';
            maxInput.name = `optimize_range[${param.id}][max]`;
            maxInput.value = param.max;
            maxInput.min = param.min;
            maxInput.max = param.max;
            maxInput.step = param.step;
            maxGroup.appendChild(maxLabel);
            maxGroup.appendChild(maxInput);
            maxCol.appendChild(maxGroup);
            
            // 步长
            const stepCol = document.createElement('div');
            stepCol.className = 'col-md-4';
            const stepGroup = document.createElement('div');
            stepGroup.className = 'input-group input-group-sm';
            const stepLabel = document.createElement('span');
            stepLabel.className = 'input-group-text';
            stepLabel.textContent = '步长';
            const stepInput = document.createElement('input');
            stepInput.type = 'number';
            stepInput.className = 'form-control';
            stepInput.name = `optimize_range[${param.id}][step]`;
            stepInput.value = param.step;
            stepInput.min = param.step / 10;
            stepInput.max = param.max - param.min;
            stepInput.step = param.step / 10;
            stepGroup.appendChild(stepLabel);
            stepGroup.appendChild(stepInput);
            stepCol.appendChild(stepGroup);
            
            // 添加范围输入
            rangeInputs.appendChild(minCol);
            rangeInputs.appendChild(maxCol);
            rangeInputs.appendChild(stepCol);
            rangeRow.appendChild(rangeInputs);
            
            inputCol.appendChild(rangeRow);
            
            // 添加行到参数列表
            paramRow.appendChild(labelCol);
            paramRow.appendChild(inputCol);
            paramsList.appendChild(paramRow);
        });
        
        // 添加到容器
        paramsContainer.appendChild(paramsList);
        
        // 更新优化目标下拉框
        this.updateOptimizationTargets(optimizationTargetsSelect, algorithm.optimization_targets);
    },
    
    /**
     * 更新优化目标选项
     * @param {HTMLElement} select - 优化目标下拉框
     * @param {Array} targets - 目标列表
     */
    updateOptimizationTargets(select, targets) {
        if (!select || !targets) return;
        
        // 清空现有选项
        select.innerHTML = '';
        
        // 添加新选项
        targets.forEach(target => {
            const option = document.createElement('option');
            option.value = target;
            option.textContent = target;
            select.appendChild(option);
        });
    },
    
    /**
     * 切换优化范围显示
     * @param {Event} event - 复选框变更事件
     */
    toggleOptimizationRange(event) {
        const paramId = event.target.value;
        const rangeElement = document.getElementById(`range_${paramId}`);
        
        if (rangeElement) {
            if (event.target.checked) {
                rangeElement.classList.remove('d-none');
            } else {
                rangeElement.classList.add('d-none');
            }
        }
    },
    
    /**
     * 表单验证
     * @param {Event} event - 表单提交事件
     */
    validateForm(event) {
        // 获取需要验证的字段
        const requiredFields = event.target.querySelectorAll('[required]');
        let isValid = true;
        
        // 检查每个必填字段
        requiredFields.forEach(field => {
            if (!field.value.trim()) {
                isValid = false;
                
                // 添加错误样式
                field.classList.add('is-invalid');
                
                // 检查是否已有错误提示，如果没有则添加
                let errorDiv = field.nextElementSibling;
                if (!errorDiv || !errorDiv.classList.contains('invalid-feedback')) {
                    errorDiv = document.createElement('div');
                    errorDiv.className = 'invalid-feedback';
                    errorDiv.textContent = '此字段是必需的';
                    field.parentNode.insertBefore(errorDiv, field.nextSibling);
                }
            } else {
                // 移除错误样式
                field.classList.remove('is-invalid');
                
                // 移除错误提示
                const errorDiv = field.nextElementSibling;
                if (errorDiv && errorDiv.classList.contains('invalid-feedback')) {
                    errorDiv.remove();
                }
            }
        });
        
        // 如果表单无效，阻止提交
        if (!isValid) {
            event.preventDefault();
            
            // 滚动到第一个错误字段
            const firstInvalidField = event.target.querySelector('.is-invalid');
            if (firstInvalidField) {
                firstInvalidField.focus();
                firstInvalidField.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }
    },
    
    /**
     * 更新系统状态信息
     */
    updateSystemStatus() {
        const statusElement = document.querySelector('.system-status');
        if (!statusElement) return;
        
        // 请求系统状态
        fetch('/api/system_status')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP错误! 状态: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                // 更新状态指示器
                for (const [component, status] of Object.entries(data)) {
                    const componentElement = document.querySelector(`.status-${component}`);
                    if (componentElement) {
                        // 更新状态文本
                        componentElement.textContent = status;
                        
                        // 更新状态类
                        componentElement.className = 'badge';
                        if (status.includes('正常')) {
                            componentElement.classList.add('bg-success');
                        } else if (status.includes('警告')) {
                            componentElement.classList.add('bg-warning');
                        } else if (status.includes('错误')) {
                            componentElement.classList.add('bg-danger');
                        } else {
                            componentElement.classList.add('bg-info');
                        }
                    }
                }
            })
            .catch(error => {
                console.error('获取系统状态失败:', error);
                statusElement.innerHTML = '<div class="alert alert-danger">获取系统状态失败</div>';
            });
    },
    
    /**
     * 显示通知提示
     * @param {string} message - 消息内容
     * @param {string} type - 提示类型 (success, warning, danger, info)
     * @param {number} duration - 显示时长(毫秒)
     */
    showNotification(message, type = 'info', duration = 3000) {
        const alertContainer = document.getElementById('alert-container');
        
        // 如果容器不存在，创建一个
        if (!alertContainer) {
            const newContainer = document.createElement('div');
            newContainer.id = 'alert-container';
            newContainer.className = 'position-fixed top-0 end-0 p-3';
            newContainer.style.zIndex = '1050';
            document.body.appendChild(newContainer);
        }
        
        // 创建提示元素
        const alertElement = document.createElement('div');
        alertElement.className = `alert alert-${type} alert-dismissible fade show`;
        alertElement.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="关闭"></button>
        `;
        
        // 添加到容器
        document.getElementById('alert-container').appendChild(alertElement);
        
        // 指定时间后自动关闭
        setTimeout(() => {
            alertElement.classList.remove('show');
            setTimeout(() => alertElement.remove(), 200);
        }, duration);
    }
};

// 当DOM加载完成后初始化应用
document.addEventListener('DOMContentLoaded', () => StarApp.init());
