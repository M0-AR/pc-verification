@echo off
echo ============================================
echo PC Verification Suite for AI Research
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    exit /b 1
)

:: Install dependencies if needed
echo Checking dependencies...
pip install -q -r requirements.txt

:: Run verification
echo.
echo Starting full PC verification...
echo This will take approximately 10-15 minutes
echo.

python pc_verify.py --duration 60 --output verification_report.json

echo.
echo ============================================
echo Verification complete!
echo Report saved to: verification_report.json
echo ============================================

pause
