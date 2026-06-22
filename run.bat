@echo off
rem SPG S1 (Unitree G1) 시뮬레이션 더블클릭 실행 (Windows)
rem 처음 한 번만:  pip install -r requirements.txt
chcp 65001 >nul
cd /d "%~dp0"
echo ================================================
echo  SPG S1 humanoid walking simulation
echo  ( python run.py %* )
echo ================================================
python run.py %*
echo.
echo [종료됨] 창을 닫으려면 아무 키나 누르세요.
pause >nul
