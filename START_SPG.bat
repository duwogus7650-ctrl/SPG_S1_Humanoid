@echo off
rem ============================================================
rem  SPG S1 (Unitree G1) 휴머노이드 보행 시뮬레이션 — 오프라인 실행기
rem  더블클릭 → (최초 1회 의존성 자동 설치) → 메뉴에서 모드 선택.
rem  ※ 최초 1회만 인터넷 필요(파이썬 패키지 + G1 모델 다운로드). 이후 오프라인 동작.
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"
title SPG S1 HUMANOID

rem --- Python 확인 ---
where python >nul 2>nul
if errorlevel 1 (
  echo [오류] Python 을 찾을 수 없습니다.
  echo        https://www.python.org 에서 Python 3.10~3.14 설치 후 다시 실행하세요.
  echo        (설치 시 "Add Python to PATH" 체크)
  pause
  exit /b 1
)

rem --- 최초 1회 의존성 설치(인터넷 필요). .spg_setup_done 마커로 이후 생략 ---
if not exist ".spg_setup_done" (
  echo [최초 설정] 파이썬 의존성 설치 중... 인터넷이 필요합니다 ^(1회만^).
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [오류] 의존성 설치 실패. 인터넷 연결을 확인하고 다시 실행하세요.
    pause
    exit /b 1
  )
  echo done> ".spg_setup_done"
  echo [설정 완료] 다음 실행부터는 설치를 건너뜁니다.
)

:menu
echo.
echo ==================== SPG S1 HUMANOID ====================
echo   1^) 자연 보행 레퍼런스 보기   (즉시·체크포인트 불필요)
echo   2^) 학습하며 보기             (빈 정책부터 PPO 학습·관찰)
echo   3^) 학습된 정책 재생          (.\checkpoints 의 정책)
echo   4^) 종료
echo ========================================================
echo   조작: Q/E 카메라 회전 · ESC 종료  (학습/재생은 R 리셋)
echo.
set "ch="
set /p "ch=번호 입력 후 Enter: "
if "%ch%"=="1" ( python run.py reference & goto menu )
if "%ch%"=="2" ( python run.py train     & goto menu )
if "%ch%"=="3" ( python run.py play      & goto menu )
if "%ch%"=="4" ( exit /b 0 )
echo [안내] 1~4 중에서 입력하세요.
goto menu
