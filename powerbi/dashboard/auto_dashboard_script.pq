// Power BI Automated Dashboard Script
// This creates a 5-page churn prediction dashboard automatically

Section Section1;
    shared Predictions = let
        Source = Csv.Document(File.Contents("..\powerbi\data\predictions_data.csv"),[Delimiter=",", Columns=30, Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
        PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
        ChangedType = Table.TransformColumnTypes(PromotedHeaders,{{"customer_id", type text}, {"churn_probability", type number}, {"risk_segment", type text}})
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
