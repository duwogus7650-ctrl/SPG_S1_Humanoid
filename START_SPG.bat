@echo off
rem ============================================================
rem  SPG S1 (Unitree G1) walking sim - offline launcher (Windows)
rem  First run installs deps (internet once); then fully offline.
rem  ASCII-only on purpose: Korean text in .bat breaks on cp949 consoles.
rem ============================================================
cd /d "%~dp0"
title SPG S1 HUMANOID

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found. Install Python 3.10-3.14 from https://www.python.org
  echo         ^(check "Add Python to PATH" during install^), then retry.
  pause
  exit /b 1
)

if not exist ".spg_setup_done" (
  echo [First-time setup] Installing dependencies ^(internet required, once^)...
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Dependency install failed. Check your internet and retry.
    pause
    exit /b 1
  )
  echo done> ".spg_setup_done"
  echo [Setup done] Next runs skip installation.
)

:menu
echo.
echo ==================== SPG S1 HUMANOID ====================
echo   1^) Reference walk      ^(target gait, instant^)
echo   2^) TRAIN               ^(watch PPO learn from scratch^)
echo   3^) Play trained policy ^(.\checkpoints^)
echo   4^) Quit
echo ========================================================
echo   Keys: Q/E rotate . ESC quit   ^(train/play: R reset^)
echo.
set "ch="
set /p "ch=Enter number then Enter: "
if "%ch%"=="1" ( python run.py reference & goto menu )
if "%ch%"=="2" ( python run.py train     & goto menu )
if "%ch%"=="3" ( python run.py play      & goto menu )
if "%ch%"=="4" ( exit /b 0 )
echo Invalid choice. Enter 1-4.
goto menu
