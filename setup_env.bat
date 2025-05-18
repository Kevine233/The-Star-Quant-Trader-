@echo off
echo ===================================================
echo       跟随庄家自动交易系统环境配置工具
echo ===================================================
echo.

REM 检查管理员权限
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 请以管理员身份运行此脚本!
    echo 右键点击脚本，选择"以管理员身份运行"
    pause
    exit /b 1
)

echo 正在检查Python安装...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python未安装。请安装Python 3.8或更高版本。
    echo 访问 https://www.python.org/downloads/ 下载安装程序。
    pause
    exit /b 1
)

echo 正在确保Python版本兼容...
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 检测到Python版本过低。需要Python 3.8或更高版本。
    echo 当前安装的版本是:
    python --version
    pause
    exit /b 1
)

echo 正在检查和创建虚拟环境...
if not exist ".venv" (
    echo 正在创建虚拟环境...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo 错误: 创建虚拟环境失败。
        pause
        exit /b 1
    )
    echo 虚拟环境创建成功。
) else (
    echo 检测到现有虚拟环境。
)

echo 正在激活虚拟环境...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo 错误: 激活虚拟环境失败。
    pause
    exit /b 1
)

echo 正在更新pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo 警告: 更新pip失败，但将继续安装依赖项。
)

echo 正在安装基本依赖...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo 警告: 安装某些依赖项失败。将继续尝试安装TA-Lib。
)

echo.
echo 正在检查TA-Lib...
python -c "import talib" >nul 2>&1
if %errorlevel% neq 0 (
    echo TA-Lib未安装。正在尝试安装...
    
    REM 检查Python架构和版本以确定正确的wheel文件
    python -c "import platform; import sys; print(f'{platform.architecture()[0]}_py{sys.version_info.major}{sys.version_info.minor}')" > tmp.txt
    set /p PYCONFIG=<tmp.txt
    del tmp.txt
    
    echo 检测到的Python配置: %PYCONFIG%
    
    REM 为不同的Python版本和架构设置不同的下载URL
    if "%PYCONFIG%"=="64bit_py38" (
        set TALIB_URL=https://download.lfd.uci.edu/pythonlibs/archived/TA_Lib-0.4.24-cp38-cp38-win_amd64.whl
    ) else if "%PYCONFIG%"=="64bit_py39" (
        set TALIB_URL=https://download.lfd.uci.edu/pythonlibs/archived/TA_Lib-0.4.24-cp39-cp39-win_amd64.whl
    ) else if "%PYCONFIG%"=="64bit_py310" (
        set TALIB_URL=https://download.lfd.uci.edu/pythonlibs/archived/TA_Lib-0.4.24-cp310-cp310-win_amd64.whl
    ) else if "%PYCONFIG%"=="64bit_py311" (
        set TALIB_URL=https://download.lfd.uci.edu/pythonlibs/archived/TA_Lib-0.4.24-cp311-cp311-win_amd64.whl
    ) else if "%PYCONFIG%"=="64bit_py312" (
        set TALIB_URL=https://download.lfd.uci.edu/pythonlibs/archived/TA_Lib-0.4.24-cp312-cp312-win_amd64.whl
    ) else (
        echo 未找到适合您Python版本(%PYCONFIG%)的TA-Lib预编译文件。
        echo 请访问以下网址手动下载TA-Lib：
        echo https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
        goto talib_skip
    )
    
    echo 正在下载TA-Lib预编译文件...
    echo URL: %TALIB_URL%
    
    REM 检查curl是否可用
    curl --version >nul 2>&1
    if %errorlevel% equ 0 (
        curl -L %TALIB_URL% -o talib_wheel.whl
    ) else (
        REM 如果curl不可用，尝试使用powershell
        powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%TALIB_URL%', 'talib_wheel.whl')"
    )
    
    if not exist "talib_wheel.whl" (
        echo 下载TA-Lib失败。
        echo 请访问以下网址手动下载TA-Lib：
        echo https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
        goto talib_skip
    )
    
    echo 正在安装TA-Lib...
    pip install talib_wheel.whl
    if %errorlevel% neq 0 (
        echo TA-Lib安装失败。
        echo 请访问以下网址手动下载TA-Lib：
        echo https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
    ) else (
        echo TA-Lib安装成功。
        del talib_wheel.whl
    )
) else (
    echo TA-Lib已安装。
)

:talib_skip

echo 创建必要的目录...
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "data\stock" mkdir data\stock
if not exist "data\crypto" mkdir data\crypto
if not exist "config" mkdir config
if not exist "backtest_results" mkdir backtest_results

echo.
echo ===================================================
echo               环境配置完成
echo ===================================================
echo.
echo 您可以使用以下命令启动系统:
echo    run.bat
echo.
echo 或者直接运行:
echo    python src/main.py
echo.

pause 