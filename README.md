<<<<<<< HEAD
# 🏦 Bank Churn Prediction System

A comprehensive **Machine Learning system** to predict customer churn for banking institutions using **52 features** across **10 key churn drivers**.  
The system automatically selects the **best-performing model** from **XGBoost**, **LightGBM**, **Random Forest**, and **Logistic Regression**, delivering explainable, data-driven retention insights.

---

## 🎥 Complete Workflow Demo

> ▶️ [Watch Full System Workflow on YouTube](https://youtu.be/bN494w5L6e8?si=rxQyoc8KwHhbzC9V) 
> 

---

## 🚀 Quick Start

### **Prerequisites**
- Python **3.8+**
- Minimum **8GB RAM** (16GB recommended for 100k dataset)
- At least **2GB free disk space**

---

### **Installation & Setup**

```bash
# Clone repository and setup environment
git clone https://github.com/DPriyangkush/Customer-Churn-Prediction-System-with-Explainable-AI.git
cd bank-churn-prediction

# Create virtual environment
python -m venv churn_env
source churn_env/bin/activate      # For Linux/Mac
churn_env\Scripts\activate         # For Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🔄 Complete Execution Pipeline

Run these commands sequentially in your terminal:

```bash
# 1️⃣ Generate synthetic data (100,000 customers)
python src/synthetic_data_generator.py

# 2️⃣ Process and clean data
python src/data_processing.py

# 3️⃣ Feature engineering and transformation
python src/feature_engineering.py

# 4️⃣ Train models and select best performer
python src/model_training.py

# 5️⃣ Generate SHAP explanations
python src/shap_explanations.py

# 6️⃣ If predictions.csv not generated
python src/model_evaluation.py

# 7️⃣ Regenerate SHAP explanations with final predictions
python src/shap_explanations.py
```

---

## 📊 Visual Analysis Pipeline (Jupyter)

```bash
# 1. Data exploration and statistics
jupyter notebook notebooks/01_data_exploration.ipynb

# 2. Feature engineering analysis
jupyter notebook notebooks/02_feature_engineering.ipynb

# 3. Model training and SHAP analysis
jupyter notebook notebooks/03_model_training_shap.ipynb

# 4. Streamlit dashboard export (automatically launches)
jupyter notebook notebooks/04_power_bi_export.ipynb
```

---

## 🧠 System Architecture

### Model Selection Process

The system automatically trains and evaluates 4 machine learning models:

| Model | Description |
|-------|-------------|
| **XGBoost** | Extreme Gradient Boosting |
| **LightGBM** | Light Gradient Boosting Machine |
| **Random Forest** | Ensemble Decision Trees |
| **Logistic Regression** | Linear baseline model |

**Selection Criteria:** Based on highest AUC-ROC, Precision, Recall, and F1-Score

### Data Flow Pipeline

```
Synthetic Data Generation (100k) 
    → Data Processing & Cleaning 
    → Feature Engineering (52 features) 
    → Model Training & Selection 
    → SHAP Explanations 
    → Streamlit Dashboard
```

---

## 📈 Model Performance Expectations

| Metric | XGBoost | LightGBM | Random Forest | Logistic Regression |
|--------|---------|----------|---------------|---------------------|
| **AUC-ROC** | 0.88+ | 0.87+ | 0.85+ | 0.78+ |
| **Precision** | 0.78+ | 0.77+ | 0.75+ | 0.70+ |
| **Recall** | 0.72+ | 0.71+ | 0.69+ | 0.65+ |
| **F1-Score** | 0.75+ | 0.74+ | 0.72+ | 0.67+ |

---

## 🎯 10 Key Churn Drivers

| Driver | Key Features | Impact |
|--------|--------------|--------|
| **Low Engagement** | days_since_last_login, activity_score | 24.3% |
| **High Fees** | overdraft_frequency, avg_monthly_fees | 14.1% |
| **Poor Service** | num_complaints, satisfaction_score | 12.1% |
| **Low Product Holding** | has_credit_card, num_products | 17.5% |
| **Short Tenure** | tenure_months, is_new_customer | 4.5% |
| **Balance Fluctuations** | balance_volatility, salary_consistency | 8.2% |
| **Demographics** | age, income, region | 3.8% |
| **Credit Issues** | credit_score, debt_to_income | 13.1% |
| **Personalization** | offer_relevance_score, promotion_response_rate | 0.6% |
| **Contract Events** | renewal_flag, account_closure_requests | 1.8% |

---

## ✅ What You Get

### Immediate Outputs (30–45 minutes total)

- 🧾 100,000 synthetic customers with realistic banking behavior
- 🧠 4 trained models with automatic best selection
- 📊 SHAP explanations for feature importance
- 💻 Interactive Streamlit dashboard for visualization
- 📈 Detailed evaluation metrics (AUC, F1, Recall, Precision)

### Key Deliverables

| File / Folder | Description |
|---------------|-------------|
| `models/production/churn_model_*.pkl` | Best performing model |
| `models/explainability/` | SHAP explanations & visuals |
| `data/processed/predictions.csv` | Final churn predictions |
| `ui/app.py` | Streamlit dashboard |

---

## ⚙️ Key Features

- ✅ 52 engineered features across 10 churn drivers
- ✅ Multi-model ensemble with automatic best model selection
- ✅ Explainable AI (SHAP) to identify churn factors
- ✅ Risk segmentation (Low/Medium/High)
- ✅ Fairness & bias detection
- ✅ Retention ROI estimation
- ✅ Interactive Streamlit dashboard for business teams

---

## 🗂️ Project Structure

```
bank-churn-prediction/
├── data/
│   ├── raw/                     # Original synthetic data
│   └── processed/               # Cleaned & engineered data
├── src/
│   ├── synthetic_data_generator.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── shap_explanations.py
├── models/
│   ├── production/
│   ├── explainability/
│   └── preprocessing/
├── ui/
│   └── app.py                   # Streamlit dashboard
├── powerbi/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training_shap.ipynb
│   └── 04_power_bi_export.ipynb
└── requirements.txt
```

---

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `predictions.csv` not generated | Run `python src/model_evaluation.py` then `python src/shap_explanations.py` |
| Memory issues (100k dataset) | Close other apps or use Google Colab |
| SHAP errors | Ensure model training completed successfully |
| Streamlit dashboard not launching | Run `streamlit run ui/app.py` |

---

## 🕐 Execution Time Estimates (100k Dataset)

| Step | Estimated Time |
|------|----------------|
| Data Generation | 2–3 min |
| Data Processing | 1–2 min |
| Feature Engineering | 3–5 min |
| Model Training | 5–8 min |
| SHAP Explanations | 3–5 min |
| **Total Pipeline** | **15–25 min** |

---

## 📞 Support

- 📚 **Documentation:** See `docs/` folder
- 🐞 **Issues:** [GitHub Issues Page](https://github.com/your-repo/issues)
- ✉️ **Email:** dpriyangkush004@gmail.com

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🌟 Star This Project

If you find this project helpful, please consider giving it a ⭐ on GitHub!

---

**Made with ❤️ for Data Science & Banking Analytics**

---

## 📝 Note About Power BI References

> **Don't get confused seeing `powerbi/` folders or "export to Power BI" files in the project structure.**  
> 
> These files will work seamlessly with the current system, even though **this system doesn't actually use Power BI**. 
> 
> **Why are they there?**  
> Initially, the plan was to automate the complete process from data visualization to exporting to Power BI and displaying all visuals in Power BI. However, I later realized that achieving complete automation with Power BI is not possible with the free tier.
> 
> **Solution:** I switched to **Streamlit** for the interactive dashboard, which provides:

> - ✅ Completely free and open-source
> - ✅ Full automation capabilities
> - ✅ Easy deployment and sharing
> - ✅ Real-time interactivity
> 
> The Power BI-related files remain in the repository for reference and can be safely ignored. The Streamlit dashboard provides all the visualization functionality you need!
=======
# bank_churn_predictor
It give churn prediction od different bank users
>>>>>>> e69522b04f1031de2309a7c68c57f9a5c9d7a3c3
