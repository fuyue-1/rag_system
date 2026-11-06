@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title 智蜂专家 - 蜜源环境融合评估
set "PROJ=%~dp0"
set "APPDIR=%PROJ%teammate_modules\bp_ds_fusion"
set "APPFILE=fusion_app.py"

set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "HF_ENDPOINT=https://hf-mirror.com"

echo ============================================================
echo    智蜂专家 · BP神经网络 + 改进D-S 融合评估 - 一键启动
echo ============================================================
echo.
echo  [算法]   3 个 BP 神经网络证据生成器 + 改进 D-S 证据理论融合
echo  [训练]   启动时自动训练，约 10 秒，请耐心等待
echo  [端口]   5002 - 与主系统 5000 / 溯源 5001 互不影响
echo.
echo  评估页面: http://127.0.0.1:5002
echo  按 Ctrl+C 停止服务
echo.
cd /d "%APPDIR%"
if errorlevel 1 (
    echo [错误] 无法进入目录: %APPDIR%
    pause
    exit /b 1
)
if not exist "%APPFILE%" (
    echo [错误] 目录 %CD% 下找不到 %APPFILE%
    pause
    exit /b 1
)
call :find_python
if not defined PY_EXE goto :no_python

"%PY_EXE%" %PY_ARGS% -c "import flask, numpy" >nul 2>&1
if errorlevel 1 (
    echo [错误] 当前 Python 缺少依赖 flask / numpy，请先执行:
    echo     "%PY_EXE%" %PY_ARGS% -m pip install flask numpy
    pause
    exit /b 1
)

echo  [解释器] %PY_EXE%
echo  正在启动并训练 BP 网络...
echo.
"%PY_EXE%" %PY_ARGS% "%APPFILE%"
echo.
echo  [提示] 服务已退出，请把上面的报错信息截图反馈。
pause
exit /b 0

:no_python
echo [错误] 没有找到可用的 Python 3，请安装 Python 3.10 以上版本，
echo        安装时勾选 Add Python to PATH，或手动创建 .venv 虚拟环境。
pause
exit /b 1

:find_python
set "PY_EXE="
set "PY_ARGS="
if exist "%PROJ%.venv\Scripts\python.exe" set "PY_EXE=%PROJ%.venv\Scripts\python.exe"
if not defined PY_EXE if exist "%USERPROFILE%\.workbuddy\binaries\python\versions\3.13.12\python.exe" set "PY_EXE=%USERPROFILE%\.workbuddy\binaries\python\versions\3.13.12\python.exe"
if not defined PY_EXE (
    for /f "delims=" %%I in ('where python.exe 2^>nul') do (
        if not defined PY_EXE set "PY_EXE=%%I"
    )
)
if not defined PY_EXE (
    for /f "delims=" %%I in ('where py.exe 2^>nul') do (
        if not defined PY_EXE set "PY_EXE=%%I"
    )
    if defined PY_EXE set "PY_ARGS=-3"
)
exit /b 0
