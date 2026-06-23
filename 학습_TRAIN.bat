@echo off
rem ============================================================
rem  SPG S1 - LOCAL TRAINING (PPO). Double-click -> training window.
rem  (run.bat shows only the reference gait. Use THIS for training.)
rem  ASCII-only on purpose: Korean text in .bat breaks on cp949 consoles.
rem ============================================================
cd /d "%~dp0"
title SPG S1 - Local Training (PPO)

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found. Install Python from https://www.python.org then retry.
  pause
  exit /b 1
)

echo ================================================
echo   SPG S1 - LOCAL TRAINING ^(PPO^)
echo   A 4-panel "PPO" training window will open:
echo   robot / live neural net / learning curve / stats.
echo   It starts clumsy and improves as it learns.
echo   Keys: Q/E rotate . R reset . ESC quit  ^(no autosave^)
echo ================================================
echo.
python run.py train
echo.
echo [Training ended] Press any key to close.
pause >nul
