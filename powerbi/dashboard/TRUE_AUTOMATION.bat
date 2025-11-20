@echo off
chcp 65001 >nul
echo ========================================
echo 🚀 POWER BI TRUE AUTOMATION
echo ========================================
echo.

echo Step 1: Opening Power BI Desktop...
start "" "C:\Program Files\Microsoft Power BI Desktop\bin\PBIDesktop.exe"

timeout /t 3 >nul

echo Step 2: Opening your data folder...
explorer "D:\bank-churn-prediction\powerbi\data"

timeout /t 2 >nul

echo Step 3: Launching Instant Dashboard Creator...
powershell -ExecutionPolicy Bypass -WindowStyle Hidden -File "D:\bank-churn-prediction\powerbi\dashboard\instant_dashboard.ps1"

echo.
echo ✅ AUTOMATION COMPLETE!
echo.
echo 📋 Check the PowerShell window for instant dashboard creation guide!
echo.
pause
