@echo off
REM 灵龙2.0 控制台 启动脚本
REM 用 pythonw.exe(无控制台)启动,如果崩了会自动用 python.exe 重启并保留错误日志

cd /d %~dp0

if exist "panel.log" del panel.log

".venv\Scripts\pythonw.exe" "sim\panel.py" > "panel.log" 2>&1
if errorlevel 1 (
    echo Panel exited with error. Falling back to python.exe for visible traceback.
    echo.
    ".venv\Scripts\python.exe" "sim\panel.py"
    pause
)
