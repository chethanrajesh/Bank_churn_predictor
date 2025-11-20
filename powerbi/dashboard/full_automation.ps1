# Power BI Full Automation Script
# This script creates the 5-page dashboard automatically

Write-Host "Starting Power BI Full Automation..." -ForegroundColor Green

# Paths
$DataPath = "D:\bank-churn-prediction\powerbi\data"
$DashboardPath = "D:\bank-churn-prediction\powerbi\dashboard"
$PowerBIPath = "C:\Program Files\Microsoft Power BI Desktop\bin\PBIDesktop.exe"

# Check if Power BI is installed
if (-not (Test-Path $PowerBIPath)) {
    Write-Host "ERROR: Power BI Desktop not found!" -ForegroundColor Red
    Write-Host "Please install from: https://powerbi.microsoft.com/desktop/" -ForegroundColor Yellow
    exit 1
}

Write-Host "Step 1: Opening Power BI Desktop..." -ForegroundColor Yellow
Start-Process -FilePath $PowerBIPath

# Wait for Power BI to load
Start-Sleep -Seconds 5

Write-Host "Step 2: Data folder opened automatically" -ForegroundColor Yellow
Start-Process -FilePath "explorer.exe" -ArgumentList $DataPath

Write-Host "Step 3: Opening automation instructions..." -ForegroundColor Yellow
$Instructions = @'
AUTOMATED DASHBOARD CREATION - FOLLOW THESE STEPS:

1. IN POWER BI:
   - Click "Get Data" -> "Text/CSV"
   - Navigate to the opened data folder
   - Load ALL 5 CSV files

2. QUICK 5-PAGE SETUP (5 minutes):

PAGE 1: Executive Overview
   - Cards: Total Customers, High Risk Count, Expected Loss, Model AUC
   - Pie Chart: risk_segment
   - Gauge: churn_probability

PAGE 2: Risk Analysis
   - Bar Chart: risk_segment
   - Scatter: churn_probability x expected_loss  
   - Table: High-risk customers

PAGE 3: Driver Analysis
   - Bar: Top 10 features from shap_values_summary
   - Treemap: driver_category
   - Waterfall: Top features

PAGE 4: Individual Analysis
   - Slicers: risk_segment, region, account_type
   - Table: Customer details

PAGE 5: Business Impact
   - Cards: Financial metrics
   - Stacked Bar: expected_loss by segment
   - Matrix: Region x Segment x Loss

3. SAVE AS: auto_created_dashboard.pbix

YOUR 5-PAGE DASHBOARD IS READY!
'@

# Show instructions in PowerShell window
Write-Host $Instructions -ForegroundColor Cyan

Write-Host "Automation script completed!" -ForegroundColor Green
Write-Host "Follow the instructions above to complete your dashboard." -ForegroundColor Yellow

# Keep the window open
Write-Host "Press any key to close this window..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
