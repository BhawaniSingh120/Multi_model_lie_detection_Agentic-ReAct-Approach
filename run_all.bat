@echo off
set PYTHON=.venv\Scripts\python.exe

echo ==========================================
echo Step 1: Preprocessing (may take time)
echo ==========================================
%PYTHON% -m src.preprocessor

if %ERRORLEVEL% NEQ 0 (
    echo Preprocessing failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ==========================================
echo Step 2: Training (5-Fold CV)
echo ==========================================
%PYTHON% -m src.train

if %ERRORLEVEL% NEQ 0 (
    echo Training failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ==========================================
echo Step 3: Training Final Model (All Data)
echo ==========================================
%PYTHON% -m src.train_final

if %ERRORLEVEL% NEQ 0 (
    echo Final Training failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ==========================================
echo Step 4: Interpretation
echo ==========================================
%PYTHON% -m src.interpret

echo.
echo Done.
pause



.venv\Scripts\streamlit run app.py  


C:\Users\Hp\Desktop\git\lie_detection\.venv\Scripts\Activate.ps1