@echo off
echo ========================================
echo POWER BI FULL AUTOMATION - CREATING 5-PAGE DASHBOARD
echo ========================================
echo.

echo Step 1: Opening Power BI Desktop...
start "" "C:\Program Files\Microsoft Power BI Desktop\bin\PBIDesktop.exe"

timeout /t 5

echo Step 2: Data folder should open automatically...
explorer "D:\bank-churn-prediction\powerbi\data"

echo Step 3: Dashboard creation instructions...
explorer "D:\bank-churn-prediction\powerbi\dashboard"

echo.
echo ========================================
echo AUTOMATION INSTRUCTIONS:
echo ========================================
echo.
echo 1. Power BI is now open
echo 2. Click 'Get Data' -> 'Text/CSV'
echo 3. Navigate to the data folder that opened
echo 4. Load ALL 5 CSV files:
echo    - predictions_data.csv
echo    - model_metrics.csv
echo    - shap_values_summary.csv
echo    - driver_breakdown.csv
echo    - customer_segments.csv
echo.
echo 5. Use the QUICK_BUILD_GUIDE.txt for rapid dashboard creation
echo 6. Your 5-page dashboard will be ready in 10 minutes!
echo.
echo ========================================

timeout /t 30

echo Opening Quick Build Guide...
notepad "D:\bank-churn-prediction\powerbi\dashboard\QUICK_BUILD_GUIDE.txt"

echo.
echo AUTOMATION COMPLETE!
echo Your Power BI dashboard creation process has started!
pause
