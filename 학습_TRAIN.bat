@echo off
rem ============================================================
rem  SPG S1 로컬 학습(PPO) — 더블클릭하면 바로 "학습 화면"으로.
rem  (run.bat은 레퍼런스(목표 걸음새)만 뜹니다. 학습은 이 파일을 쓰세요.)
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"
title SPG S1 - 로컬 학습(PPO)

where python >nul 2>nul
if errorlevel 1 (
  echo [오류] Python 을 찾을 수 없습니다. https://www.python.org 에서 설치 후 다시 실행하세요.
  pause
  exit /b 1
)

echo ================================================
echo   SPG S1 로컬 학습 (PPO 강화학습) 시작
echo   - "PPO 강화학습" 4분할 화면이 뜹니다(로봇/신경망/학습곡선/패널).
echo   - 처음엔 어설프게 움직이다 점점 걸음이 좋아집니다.
echo   - 조작: Q/E 회전 . R 리셋 . ESC 종료   (자동저장 없음)
echo ================================================
echo.
python run.py train
echo.
echo [학습 종료] 아무 키나 누르면 창을 닫습니다.
pause >nul
