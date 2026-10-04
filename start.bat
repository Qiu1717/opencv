@echo off
chcp 65001 > nul
title 智能校园人脸识别门禁系统

echo ============================================
echo    智能校园人脸识别门禁系统启动器
echo ============================================
echo.

cd /d "%~dp0"

echo [1/3] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo [2/3] 检查后端依赖...
if not exist "backend\venv\Scripts\activate.bat" (
    echo 创建虚拟环境...
    cd backend
    python -m venv venv
    cd ..
)

echo [3/3] 安装依赖...
call backend\venv\Scripts\activate.bat
pip install -r backend\requirements.txt -q

echo.
echo ============================================
echo 所有依赖已安装完成！
echo ============================================
echo.
echo 启动后端服务器...
start "后端服务器" cmd /k "call backend\venv\Scripts\activate.bat && cd backend && python app.py"

echo.
echo 等待服务器启动...
timeout /t 5 /nobreak > nul

echo.
echo 正在打开浏览器...
start http://localhost:5000

echo.
echo ============================================
echo 系统已启动！
echo 后端地址: http://localhost:5000
echo 按任意键关闭此窗口...
echo ============================================

pause > nul
