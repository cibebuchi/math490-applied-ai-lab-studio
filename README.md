# MATH 490 Applied AI Lab Studio

A Streamlit teaching app for undergraduate labs in data science and machine learning.

Domains supported:
- Behavior / Society
- Finance
- Environment
- Health
- Forecasting
- Computer Vision

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

Download Kaggle CSV/image datasets and upload them inside the app. The app also includes synthetic demo datasets so the class can run without downloads.

Recommended Kaggle datasets:
- Students Performance in Exams: https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
- Student Performance Data Set: https://www.kaggle.com/datasets/larsen0966/student-performance-data-set
- Credit Card Fraud Detection: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Loan Approval Classification: https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data
- S&P 500 Stocks: https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks
- World Stock Prices: https://www.kaggle.com/datasets/nelgiriyewithana/world-stock-prices-daily-updating
- Air Quality Time Series UCI: https://www.kaggle.com/datasets/aayushkandpal/air-quality-time-series-data-uci
- Heart Disease: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
- PlantVillage: https://www.kaggle.com/datasets/mohitsingh1804/plantvillage

## Class structure

Each page includes:
- Lab goal
- What students should change
- What students should interpret
- Short assignment prompts


## Version 5 update

- Expanded Model Evaluation & Comparison page.
- Students can now choose models to compare, evaluation metrics, training data size, random seed, and optional cross-validation.
- Added model comparison tables, best-model highlight, overfitting-gap visualization, observed-vs-predicted plot, and confusion matrix for the best classifier.
- Replaced technical metric labels with human-readable names in the app interface.
- Added footer copyright notice.
