@echo off
rem SPG S1 (Unitree G1) reference viewer - double-click (Windows).
rem ASCII-only on purpose: Korean text in .bat breaks on cp949 consoles.
rem First time only:  pip install -r requirements.txt
rem For TRAINING use the TRAIN launcher, or START_SPG.bat option 2.
cd /d "%~dp0"
echo ================================================
echo  SPG S1 humanoid - reference walk (target gait)
echo  ( python run.py %* )
echo  TRAIN: run "python run.py train" or START_SPG.bat ^(2^)
echo ================================================
python run.py %*
echo.
echo [Closed] Press any key to exit.
pause >nul
