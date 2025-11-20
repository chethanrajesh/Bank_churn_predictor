@echo off
echo ========================================
echo POWER BI FULL AUTOMATION - CREATING 5-PAGE DASHBOARD
echo ========================================
echo.

echo Step 1: Opening Power BI Desktop...
start "" "C:\Program Files\Microsoft Power BI Desktop\bin\PBIDesktop.exe"

echo Step 2: Opening your data folder...
explorer "D:\bank-churn-prediction\powerbi\data"

echo Step 3: Opening dashboard creation console...
powershell -ExecutionPolicy Bypass -File "D:\bank-churn-prediction\powerbi\dashboard\full_automation.ps1"

echo.
echo AUTOMATION COMPLETE!
echo Your Power BI dashboard creation process has started!
echo.
pause
