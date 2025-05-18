@echo off
echo ===================================================
echo       跟随庄家自动交易系统 (Smart Money Follow System)
echo ===================================================
echo 正在启动系统...

rem 检查Python是否已安装
python --version > nul 2>&1
if errorlevel 1 (
    echo 错误: 未检测到Python! 请先安装Python 3.8或更高版本。
    pause
    exit /b
)

rem 检查依赖包是否已安装
echo 检查依赖包...
pip install -r requirements_simple.txt > nul 2>&1

rem 启动主程序
echo 正在启动交易系统...
python src/main.py %*

pause 