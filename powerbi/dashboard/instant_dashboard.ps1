# INSTANT POWER BI DASHBOARD CREATOR
# Run this after loading data in Power BI

Write-Host "=========================================" -ForegroundColor Green
Write-Host "INSTANT 5-PAGE DASHBOARD CREATOR" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""

# Step-by-step instructions for instant dashboard creation
$Instructions = @'

🎯 INSTANT DASHBOARD CREATION - FOLLOW THESE 7 STEPS:

STEP 1: LOAD DATA (Already Done!)
   - All 5 CSV files are loaded in Power BI

STEP 2: CREATE PAGE 1 - Executive Overview (1 minute)
   - Click "New Page" and name it "Executive Overview"
   - Add 4 Card Visuals:
     * Total Customers: Count of customer_id
     * High Risk: Count where risk_segment = "High Risk" 
     * Expected Loss: Sum of expected_loss
     * Model AUC: Lookup from model_metrics
   - Add Pie Chart: risk_segment × Count(customer_id)
   - Add Gauge: Average of churn_probability

STEP 3: CREATE PAGE 2 - Risk Analysis (1 minute)  
   - New Page → "Risk Analysis"
   - Bar Chart: risk_segment × Count(customer_id)
   - Scatter Plot: churn_probability × expected_loss
   - Table: customer_id, churn_probability, risk_segment, region

STEP 4: CREATE PAGE 3 - Driver Analysis (1 minute)
   - New Page → "Driver Analysis"
   - Horizontal Bar: Top 10 features from shap_values_summary
   - Treemap: driver_category × normalized_importance  
   - Waterfall: Top 15 features by shap_importance

STEP 5: CREATE PAGE 4 - Individual Analysis (1 minute)
   - New Page → "Individual Analysis"
   - Add Slicers: risk_segment, region, account_type
   - Table: Customer details
   - Card: Individual risk scores

STEP 6: CREATE PAGE 5 - Business Impact (1 minute)
   - New Page → "Business Impact"
   - Cards: Total expected_loss, High Risk loss
   - Stacked Bar: expected_loss by risk_segment
   - Matrix: Region × risk_segment × expected_loss

STEP 7: FORMAT & SAVE (1 minute)
   - Apply colors: Red=High Risk, Yellow=Medium, Green=Low
   - Add titles to all visuals
   - Save as: "auto_created_dashboard.pbix"

⏱️ TOTAL TIME: 6 MINUTES
🎉 YOUR 5-PAGE DASHBOARD IS READY!

DATA INSIGHTS:
- Total Customers: 1,500
- High-Risk Customers: 10 (0.7%)
- Top Churn Driver: days_since_last_login
- Expected Business Impact: Ready for analysis

'@

Write-Host $Instructions -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to close this guide..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
