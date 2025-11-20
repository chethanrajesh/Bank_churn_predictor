// Power BI Automated Dashboard Template
// This M script sets up the data model for automatic dashboard creation

Section Section1;

// Load all data tables
shared Predictions = let
    Source = Csv.Document(File.Contents("..\powerbi\data\predictions_data.csv"),[Delimiter=",", Columns=30, Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    ChangedType = Table.TransformColumnTypes(PromotedHeaders,{
        {"customer_id", type text}, 
        {"churn_probability", type number}, 
        {"risk_segment", type text},
        {"expected_loss", type number},
        {"region", type text},
        {"account_type", type text}
    })
in
    ChangedType;

shared ModelMetrics = let
    Source = Csv.Document(File.Contents("..\powerbi\data\model_metrics.csv"),[Delimiter=",", Columns=10, Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true])
in
    PromotedHeaders;

shared SHAPValues = let
    Source = Csv.Document(File.Contents("..\powerbi\data\shap_values_summary.csv"),[Delimiter=",", Columns=10, Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true])
in
    PromotedHeaders;

shared DriverBreakdown = let
    Source = Csv.Document(File.Contents("..\powerbi\data\driver_breakdown.csv"),[Delimiter=",", Columns=10, Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true])
in
    PromotedHeaders;

// DASHBOARD AUTOMATION INSTRUCTIONS:
// After loading this data, run the automation script to create 5 pages automatically
