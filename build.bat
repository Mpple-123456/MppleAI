@echo off
chcp 65001 > nul
title MppleAI Builder

echo ========================================
echo    MppleAI - Build NumPy Executable
echo ========================================
echo.

echo [*] Installing dependencies...
pip install numpy pyinstaller -q

echo.
echo [*] Building executable...
echo.

pyinstaller app.spec --clean

if exist "dist\MppleAI\MppleAI.exe" (
    echo.
    echo [OK] Build successful!
    echo [*] Executable: dist\MppleAI\MppleAI.exe
    echo [*] Copy chat_model.pkl to run with trained model
) else (
    echo.
    echo [ERROR] Build failed!
)

pause
