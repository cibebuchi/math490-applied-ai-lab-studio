from __future__ import annotations

import io
import json
import hashlib
import math
import re
import warnings
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image
from scipy import stats

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_recall_curve, precision_score, r2_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    KFold, LeaveOneOut, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold,
    TimeSeriesSplit, cross_val_score, train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")
st.set_page_config(page_title="MATH 490 Applied AI Lab Studio", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.25rem;padding-bottom:3rem;max-width:1450px}
h1{font-size:2.05rem!important;margin-bottom:.25rem!important} h2{font-size:1.5rem!important}
[data-testid="stMetricValue"]{font-size:1.55rem}.stButton>button{border-radius:9px;font-weight:600}
.student-card{border:1px solid rgba(49,51,63,.18);border-radius:14px;padding:1rem;margin:.35rem 0;background:rgba(240,244,248,.48)}
.path-card{border-left:5px solid #6c63ff;border-radius:8px;padding:.9rem 1rem;background:rgba(108,99,255,.07);margin-bottom:.7rem}
.lesson-card{border:1px solid rgba(49,51,63,.17);border-radius:16px;padding:1.1rem 1.2rem;margin:.6rem 0;background:rgba(248,250,252,.82)}
.lesson-number{display:inline-block;min-width:2rem;height:2rem;line-height:2rem;text-align:center;border-radius:50%;background:#6c63ff;color:white;font-weight:700;margin-right:.55rem}
.simple-note{border-left:5px solid #2b7de9;border-radius:9px;padding:.85rem 1rem;background:rgba(43,125,233,.08);margin:.55rem 0}
.success-note{border-left:5px solid #2f9e44;border-radius:9px;padding:.85rem 1rem;background:rgba(47,158,68,.08);margin:.55rem 0}
.progress-pill{display:inline-block;border-radius:999px;padding:.25rem .65rem;margin:.12rem;background:rgba(108,99,255,.11);font-size:.9rem}
.home-card{border:1px solid rgba(49,51,63,.16);border-radius:18px;padding:1.25rem;min-height:175px;background:#fff;box-shadow:0 4px 16px rgba(15,23,42,.05);margin:.4rem 0}
.home-card h3{margin:.15rem 0 .4rem 0!important;font-size:1.25rem!important}
.tiny-label{font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;color:#5b6475;font-weight:700}
.question-card{border-radius:16px;padding:1.05rem 1.15rem;background:linear-gradient(135deg,rgba(94,92,230,.11),rgba(43,125,233,.07));border:1px solid rgba(94,92,230,.2);margin:.65rem 0}
.warning-card{border-left:5px solid #e67700;border-radius:9px;padding:.85rem 1rem;background:rgba(230,119,0,.08);margin:.55rem 0}
.library-card{border:1px solid rgba(49,51,63,.16);border-radius:18px;padding:1.2rem 1.3rem;background:linear-gradient(135deg,rgba(94,92,230,.08),rgba(43,125,233,.04));margin:.75rem 0}
.library-card h3,.model-card h3{margin:.28rem 0 .5rem 0!important;font-size:1.45rem!important}
.library-card p,.model-card p{font-size:1.02rem;line-height:1.55;margin-bottom:.2rem}
.model-card{border:1px solid rgba(49,51,63,.16);border-radius:18px;padding:1.2rem 1.3rem;background:linear-gradient(135deg,rgba(47,158,68,.08),rgba(43,125,233,.04));margin:.75rem 0}
.course-map-card{border:1px solid rgba(49,51,63,.14);border-radius:14px;padding:.85rem 1rem;margin:.45rem 0;background:rgba(248,250,252,.78);line-height:1.45}
</style>
""", unsafe_allow_html=True)

NAVIGATION = {
    "Start Here": ["Home and Quick Start", "Data and Research Questions"],
    "Foundations": ["Probability and Uncertainty", "Visualization and Descriptive Statistics", "Relationships and Association"],
    "Supervised Regression": ["Simple and Multiple Regression", "Machine Learning for Regression"],
    "Supervised Classification": ["Logistic Regression", "Machine Learning for Classification"],
    "Reliability": ["Predictor Selection", "Model Explanations", "Model Evaluation and Comparison", "Cross-Validation and Model Selection", "Bootstrap and Uncertainty"],
    "Applications": ["Time Series Forecasting", "Computer Vision"],
    "Communicate": ["Three-Slide Mini-Report Builder"],
}

WEEKLY_LABS = {
    "Week 1": {
        "title": "App Orientation and Probability Foundations",
        "tool": "Probability and Uncertainty",
        "learn": "Learn the weekly app rhythm while using a simple probability simulation to compare theoretical and experimental probability.",
        "key_idea": "Probability describes how likely an outcome is. A simulation repeats a random experiment so we can compare the long-run observed proportion with the probability we expected.",
        "terms": {
            "Experiment": "A repeatable chance process, such as tossing a coin.",
            "Outcome": "One possible result of the experiment.",
            "Sample space": "The complete set of possible outcomes.",
            "Theoretical probability": "The probability expected from the rules of the experiment.",
            "Experimental probability": "The observed proportion after repeating the experiment.",
        },
        "steps": ["Follow the app's Learn → Practise → Reflect rhythm", "Choose a probability and number of trials", "Run the simulation", "Compare theoretical and experimental probability", "Explain what changes when the number of trials increases"],
        "assignment": "Use the probability simulator, compare theoretical and experimental probability, and explain what happens as the number of trials increases.",
        "required": ["Experiment and sample space", "Theoretical probability", "Experimental probability", "Number of trials", "One plain-language conclusion"],
        "mistake": "A probability of 0.50 does not guarantee that exactly half of a small number of trials will produce the outcome.",
    },
    "Week 2": {
        "title": "Data Introduction and Research Questions",
        "tool": "Data and Research Questions",
        "learn": "Choose a clear target, useful predictors, and one answerable research question.",
        "key_idea": "A good data-science question names the outcome and the information that may help explain or predict it.",
        "terms": {
            "Research question": "The exact question the analysis will answer.",
            "Target": "The outcome we want to explain or predict.",
            "Candidate predictor": "A variable that may contain useful information about the target.",
            "Leakage": "Using information that already contains the answer or would not be available when making a prediction.",
        },
        "steps": ["Inspect the dataset", "Choose the target", "Choose candidate predictors", "Write one clear research question"],
        "assignment": "Prepare a three-slide mini-report describing the assigned dataset, one target, possible predictors, and at least one data-science question.",
        "required": ["Research question", "Target", "Candidate predictors", "Why the question matters"],
        "mistake": "Do not use a predictor that directly reveals the target.",
    },
    "Week 3": {
        "title": "Descriptive Statistics and Visualization",
        "tool": "Visualization and Descriptive Statistics",
        "learn": "Use simple statistics and clear plots to describe the target and important predictors.",
        "key_idea": "Before fitting a model, look at the data. A plot can reveal the center, spread, unusual values, and group differences.",
        "terms": {
            "Center": "A typical value, often described by the mean or median.",
            "Spread": "How widely values differ from one another.",
            "Outlier": "A value that is far from most other values.",
            "Distribution": "The overall pattern of values in a variable.",
        },
        "steps": ["Summarize the target", "Create one target plot", "Create one relationship or group plot", "Write two observations"],
        "assignment": "Prepare a three-slide mini-report using statistics and visualizations to address the research question.",
        "required": ["At least one summary statistic", "At least two suitable plots", "Two patterns noticed", "One limitation"],
        "mistake": "Do not describe a pattern without checking the axis labels and units.",
    },
    "Week 4": {
        "title": "Correlation and Association",
        "tool": "Relationships and Association",
        "learn": "Choose the correct association method and explain the strength and direction of a relationship.",
        "key_idea": "Association tells us whether variables move together or differ across groups. It does not prove that one variable causes another.",
        "terms": {
            "Positive association": "Larger values of one variable tend to occur with larger values of another.",
            "Negative association": "Larger values of one variable tend to occur with smaller values of another.",
            "Partial correlation": "The relationship between two numerical variables after accounting for selected controls.",
            "Eta": "The strength of association between a categorical variable and a numerical variable.",
        },
        "steps": ["Choose two variables", "Choose the matching association method", "Add controls when appropriate", "Explain strength, direction, and limits"],
        "assignment": "Prepare a three-slide mini-report examining associations connected to the research question using suitable correlation or group-association methods.",
        "required": ["Variables compared", "Method used", "Association value", "Plain-language interpretation"],
        "mistake": "Correlation and association do not establish causation.",
    },
    "Week 5": {
        "title": "Simple Linear Regression",
        "tool": "Simple and Multiple Regression",
        "learn": "Fit one straight-line model and explain its intercept, slope, fit, errors, and limitation.",
        "key_idea": "Simple linear regression uses one numerical predictor to estimate a numerical target with a straight line.",
        "terms": {
            "Intercept": "The predicted target when the predictor equals zero.",
            "Slope": "The predicted change in the target for a one-unit increase in the predictor.",
            "Residual": "Observed value minus predicted value.",
            "R-squared": "How much of the target variation is explained by the fitted model.",
        },
        "steps": ["Choose one numerical predictor", "Inspect the scatterplot", "Fit the model", "Interpret slope, intercept, fit, and residuals"],
        "assignment": "Prepare a three-slide mini-report explaining the slope, intercept, model fit, and one limitation in relation to the research question.",
        "required": ["Model equation", "Slope", "Intercept", "MAE or RMSE", "R-squared", "One limitation"],
        "mistake": "A slope describes an association in the fitted data; it does not prove causation.",
    },
    "Week 6": {
        "title": "Multiple Linear Regression and Assumptions",
        "tool": "Simple and Multiple Regression",
        "learn": "Add predictors, compare simple and multiple regression, and inspect residual patterns.",
        "key_idea": "Multiple regression estimates the target using several predictors at the same time. Each coefficient is interpreted while the other predictors are held constant.",
        "terms": {
            "Multiple regression": "A linear model with two or more predictors.",
            "Coefficient": "The model's estimated effect for one predictor while holding the others constant.",
            "Overlapping information": "Predictors may contain similar information, making individual coefficients less stable.",
            "Residual pattern": "A visible pattern in residuals can show that a straight-line model misses important structure.",
        },
        "steps": ["Use the Week 5 target", "Add approved predictors", "Fit multiple regression", "Compare held-out error with the simple model"],
        "assignment": "Compare simple and multiple regression performance and assess the added value of the multiple model.",
        "required": ["Predictors added", "Coefficient directions", "Held-out error", "Comparison with simple regression", "One assumption or limitation"],
        "mistake": "Adding predictors does not automatically make a model better on unseen data.",
    },
    "Week 7": {
        "title": "Machine Learning Regression Model Comparison",
        "tool": "Machine Learning for Regression",
        "learn": "Compare a flexible regression model with simpler baselines using held-out data.",
        "key_idea": "A more complex model is useful only when it predicts new observations better, not merely when it fits the training data better.",
        "terms": {
            "Held-out data": "Rows not used to fit the model.",
            "MAE": "The average absolute prediction error in the target's units.",
            "RMSE": "An error measure that gives extra weight to larger mistakes.",
            "Overfitting": "Learning the training data too closely and performing poorly on new data.",
        },
        "steps": ["Keep the same target and predictors", "Choose a machine-learning model", "Compare with linear and mean baselines", "Explain whether complexity helped"],
        "assignment": "Prepare a three-slide mini-report comparing simple and multiple linear regression with a complex machine-learning regression model.",
        "required": ["Models compared", "Same evaluation split", "MAE or RMSE", "Best model", "Reason for recommendation"],
        "mistake": "Do not compare models that were evaluated on different rows.",
    },
    "Week 8": {
        "title": "Midterm Preparation Lab",
        "tool": None,
        "learn": "Review the main ideas from data questions, visualization, association, and regression.",
        "key_idea": "The goal is to explain what each output means, not to memorize buttons.",
        "terms": {
            "Target": "The outcome being studied.",
            "Association": "A relationship between variables without a causal claim.",
            "Prediction error": "The difference between observed and predicted values.",
            "Generalization": "How well a method works on new observations.",
        },
        "steps": ["Review key terms", "Interpret sample outputs", "Practice choosing methods", "Explain one model result in plain language"],
        "assignment": "Complete the instructor's review questions and interpretation exercises.",
        "required": ["Method-choice practice", "Metric interpretation", "One regression interpretation", "One limitation statement"],
        "mistake": "Do not focus only on formulas; explain the meaning of each result.",
    },
    "Week 9": {
        "title": "Logistic Regression for Classification",
        "tool": "Logistic Regression",
        "learn": "Predict a two-class outcome using probabilities, a threshold, and a confusion matrix.",
        "key_idea": "Logistic regression first estimates a probability. A threshold then turns that probability into a class prediction.",
        "terms": {
            "Predicted probability": "The model's estimated chance of the positive class.",
            "Threshold": "The cutoff used to turn a probability into a class.",
            "False positive": "The model predicts positive when the observed class is negative.",
            "False negative": "The model predicts negative when the observed class is positive.",
        },
        "steps": ["Choose a binary target", "Fit logistic regression", "Move the threshold", "Explain the confusion matrix and important error"],
        "assignment": "Prepare a three-slide mini-report explaining the classification target, predicted probabilities, confusion matrix, and one limitation.",
        "required": ["Positive class", "Threshold", "Confusion matrix", "Accuracy or F1-score", "Important error type"],
        "mistake": "A threshold of 0.50 is common, but it is not automatically best for every decision.",
    },
    "Week 10": {
        "title": "Machine Learning Classification Model Comparison",
        "tool": "Machine Learning for Classification",
        "learn": "Compare logistic regression with machine-learning classifiers using suitable held-out metrics.",
        "key_idea": "Different classification metrics answer different questions. Accuracy alone can be misleading when classes are unequal.",
        "terms": {
            "Accuracy": "The fraction of all predictions that are correct.",
            "Precision": "Among predicted positives, the fraction that are truly positive.",
            "Recall": "Among observed positives, the fraction correctly found.",
            "F1-score": "A balance between precision and recall.",
        },
        "steps": ["Keep the same classification question", "Choose a machine-learning classifier", "Compare held-out metrics", "Recommend a model and explain why"],
        "assignment": "Prepare a three-slide mini-report comparing logistic regression with one or more machine-learning classifiers.",
        "required": ["Models compared", "Primary metric", "Confusion matrix", "Best model", "Interpretability trade-off"],
        "mistake": "Do not choose a model only because it has the highest training accuracy.",
    },
    "Week 11": {
        "title": "Model Evaluation and Comparison",
        "tool": "Model Evaluation and Comparison",
        "learn": "Choose a metric that matches the task and compare models fairly.",
        "key_idea": "The best model depends on the decision goal. A model can be best for prediction while another is easier to explain.",
        "terms": {
            "Primary metric": "The main score used to rank models.",
            "Fair comparison": "Every model uses the same target, predictors, split, and observations.",
            "Prediction model": "The model chosen mainly for held-out performance.",
            "Interpretation model": "A simpler model chosen because its behavior is easier to explain.",
        },
        "steps": ["Choose regression or classification", "Select models", "Choose the primary metric", "Recommend one model for prediction and one for explanation"],
        "assignment": "Evaluate regression or classification outputs and recommend the best-performing model in relation to the research question.",
        "required": ["Task type", "Primary metric", "Fair comparison table", "Prediction recommendation", "Interpretation recommendation"],
        "mistake": "Do not mix metrics whose direction differs; lower error is better, while higher accuracy or F1 is better.",
    },
    "Week 12": {
        "title": "Cross-Validation",
        "tool": "Cross-Validation and Model Selection",
        "learn": "Measure how model performance changes across different validation splits.",
        "key_idea": "One train-test split can be lucky or unlucky. Cross-validation repeats the evaluation across several parts of the data.",
        "terms": {
            "Fold": "One part of the data used for validation while the other parts are used for training.",
            "Cross-validation mean": "Average performance across folds.",
            "Variability": "How much the score changes from one fold to another.",
            "Stable model": "A model whose performance and ranking remain reasonably similar across resamples.",
        },
        "steps": ["Run a single validation split", "Run k-fold cross-validation", "Compare average and variability", "Check whether model ranking stays stable"],
        "assignment": "Compare validation-split and k-fold results, explain why one split may be unstable, and assess model-ranking stability.",
        "required": ["Validation strategy", "Mean score", "Score variability", "Stability statement", "Recommended model"],
        "mistake": "Do not use shuffled cross-validation for time-series forecasting.",
    },
    "Week 13": {
        "title": "Bootstrap and Uncertainty",
        "tool": "Bootstrap and Uncertainty",
        "learn": "Use resampling with replacement to estimate uncertainty around a chosen result.",
        "key_idea": "Bootstrap resampling creates many new samples from the observed data to show how much a statistic, coefficient, metric, or prediction may vary.",
        "terms": {
            "Bootstrap sample": "A sample drawn with replacement from the observed rows.",
            "Standard error": "The typical amount the estimate changes across bootstrap samples.",
            "Confidence interval": "A range summarizing uncertainty around the estimate.",
            "Bias": "The difference between the bootstrap average and the original estimate.",
        },
        "steps": ["Choose a result connected to the question", "Run bootstrap samples", "Inspect the distribution", "Interpret the estimate and interval"],
        "assignment": "Prepare a three-slide mini-report estimating and interpreting uncertainty for a statistic, coefficient, metric, or prediction.",
        "required": ["Quantity bootstrapped", "Original estimate", "Standard error", "Confidence interval", "Interpretation"],
        "mistake": "A confidence interval is not the range containing 95% of individual observations.",
    },
    "Week 14": {
        "title": "Time Series Forecasting",
        "tool": "Time Series Forecasting",
        "learn": "Create past-value predictors, preserve time order, and compare a forecasting model with simple baselines.",
        "key_idea": "Forecasting predicts a later value using information that would have been available earlier. Future information must never enter the training predictors.",
        "terms": {
            "Lag": "A past value used as a predictor.",
            "Forecast horizon": "How far ahead the model predicts.",
            "Naïve forecast": "A simple forecast, such as using the latest observed value.",
            "Time-ordered split": "Training on earlier observations and testing on later observations.",
        },
        "steps": ["Choose the time column and target", "Choose lags or an LSTM lookback", "Add only predictors available at forecast time", "Compare the model with naïve baselines"],
        "assignment": "Prepare a three-slide mini-report comparing a simple forecast with a machine-learning or LSTM forecast using time-ordered evaluation.",
        "required": ["Forecast question", "Horizon", "Inputs or lags", "Baseline comparison", "Best forecast model"],
        "mistake": "Never randomly split time-series data or use future observations as predictors.",
    },
    "Week 15": {
        "title": "Introductory Computer Vision",
        "tool": "Computer Vision",
        "learn": "Understand images as pixel data, use pretrained recognition, and evaluate a small image classifier on unseen images.",
        "key_idea": "A computer sees an image as numbers arranged in rows, columns, and color channels. A classifier learns patterns that help separate labeled image classes.",
        "terms": {
            "Pixel": "A tiny image element represented by numerical color values.",
            "Class label": "The category assigned to an image.",
            "Confidence": "The model's predicted probability or strength for a class.",
            "Validation image": "An unseen image used to test whether the model generalizes.",
        },
        "steps": ["Inspect image dimensions", "Run pretrained recognition or load labeled folders", "Train a classifier", "Evaluate unseen-image accuracy and errors"],
        "assignment": "Prepare a three-slide mini-report documenting the image task, model, unseen-image accuracy, errors, and limitation.",
        "required": ["Image classes", "Training method", "Validation result", "Confusion matrix", "One limitation"],
        "mistake": "High training accuracy does not prove that the model recognizes new images well.",
    },
    'Week 16': {'title': 'Final Review: Classification, Validation, Uncertainty, Forecasting, and Computer Vision',
 'tool': None,
 'learn': 'Integrate the second half of the course by choosing suitable methods, interpreting outputs, diagnosing '
          'errors, and explaining responsible AI decisions.',
 'key_idea': 'A strong data scientist can connect the target, data structure, model, evaluation design, uncertainty, '
             'and limitations into one defensible analysis.',
 'terms': {'Classification threshold': 'The cutoff that turns a predicted probability into a class decision.',
           'Validation': 'Testing model choices on data not used to fit those choices.',
           'Uncertainty': 'The amount an estimate or prediction may vary.',
           'Forecast horizon': 'How far into the future a forecast is made.',
           'Generalization': 'Performance on genuinely unseen, representative data.'},
 'steps': ['Review classification models and metrics',
           'Review validation, tuning, regularization, and uncertainty',
           'Review forecasting and time-series leakage',
           'Review neural networks and computer vision',
           'Complete the 50-question final review'],
 'assignment': 'Complete the comprehensive final review and prepare a three-slide correction summary of the concepts '
               'that required the most revision.',
 'required': ['Classification and metric interpretation',
              'Validation and tuning decisions',
              'Uncertainty interpretation',
              'Forecasting and computer-vision concepts',
              'Three corrected explanations'],
 'mistake': 'Do not select a method or metric without first identifying the target, data structure, and real decision '
            'being supported.'},
}

@st.cache_data(show_spinner=False)
def demo_students(n=500, seed=42):
    rng=np.random.default_rng(seed); gender=rng.choice(["female","male"],n); prep=rng.choice(["none","completed"],n,p=[.65,.35])
    lunch=rng.choice(["standard","free/reduced"],n,p=[.65,.35]); parent=rng.choice(["high school","some college","bachelor","master"],n)
    study=rng.normal(5,2,n).clip(0,12); attend=rng.normal(86,9,n).clip(50,100); sleep=rng.normal(7,1.1,n).clip(3.5,10)
    base=25+3.4*study+.13*attend+.8*sleep+(prep=="completed")*6+(lunch=="standard")*3
    math_score=(base+rng.normal(0,10,n)).clip(0,100); reading=(base+(gender=="female")*4+rng.normal(0,9,n)).clip(0,100)
    writing=(base+(gender=="female")*5+rng.normal(0,9,n)).clip(0,100)
    df=pd.DataFrame({"gender":gender,"lunch":lunch,"parent_education":parent,"test_preparation":prep,"study_hours":study.round(2),
        "attendance_percent":attend.round(1),"sleep_hours":sleep.round(2),"math_score":math_score.round(1),"reading_score":reading.round(1),"writing_score":writing.round(1)})
    df["passed_math"]=np.where(df.math_score>=60,"passed","not passed"); return df

@st.cache_data(show_spinner=False)
def demo_health(n=500, seed=7):
    rng=np.random.default_rng(seed); age=rng.integers(29,78,n); chol=rng.normal(220,45,n).clip(120,390)
    maxhr=rng.normal(185-.72*age,15,n).clip(70,205); bp=rng.normal(112+.28*age,15,n).clip(85,200)
    pain=rng.choice(["typical","atypical","non-anginal","asymptomatic"],n); angina=rng.choice(["yes","no"],n,p=[.27,.73]); activity=rng.gamma(2.5,35,n).clip(0,400)
    logit=-7.8+.045*age+.008*chol+.022*bp-.022*maxhr+.85*(angina=="yes")+.55*(pain=="asymptomatic")-.003*activity
    disease=np.where(rng.binomial(1,1/(1+np.exp(-logit)))==1,"disease","no disease")
    return pd.DataFrame({"age":age,"cholesterol":chol.round(1),"max_heart_rate":maxhr.round(1),"resting_blood_pressure":bp.round(1),
        "chest_pain_type":pain,"exercise_angina":angina,"weekly_activity_minutes":activity.round(1),"heart_disease":disease})

@st.cache_data(show_spinner=False)
def demo_finance(n=900, seed=11):
    rng=np.random.default_rng(seed); income=rng.lognormal(10.8,.45,n)/1000; loan=rng.normal(130,55,n).clip(15,400)
    credit=rng.normal(680,75,n).clip(350,850); dti=rng.beta(2.2,6,n).clip(.02,.9); years=rng.integers(0,25,n); home=rng.choice(["rent","mortgage","own"],n,p=[.45,.42,.13])
    logit=-1.9+.012*(credit-650)+.015*income-.01*loan-3*dti+.035*years+.25*(home=="own")
    approved=np.where(rng.binomial(1,1/(1+np.exp(-logit)))==1,"approved","not approved")
    return pd.DataFrame({"income_thousand":income.round(1),"loan_amount_thousand":loan.round(1),"credit_score":credit.round(),"debt_to_income":dti.round(3),
        "employment_years":years,"home_ownership":home,"loan_approved":approved})

@st.cache_data(show_spinner=False)
def demo_air_quality(n=1000, seed=20):
    rng=np.random.default_rng(seed); date=pd.date_range("2024-01-01",periods=n,freq="h"); hour=date.hour; day=np.arange(n)/24
    temp=18+8*np.sin(2*np.pi*np.arange(n)/24)+3*np.sin(2*np.pi*day/30)+rng.normal(0,2,n); hum=68-.9*temp+rng.normal(0,7,n)
    traffic=45+26*((hour>=7)&(hour<=9))+22*((hour>=16)&(hour<=19))+rng.normal(0,7,n); wind=rng.gamma(2,1.5,n).clip(.1,15)
    pm=8+.15*traffic+.16*hum-.12*temp-.55*wind+rng.normal(0,3,n)
    return pd.DataFrame({"datetime":date,"hour":hour,"temperature":temp.round(2),"humidity":hum.round(2),"traffic_index":traffic.round(2),"wind_speed":wind.round(2),"pm25":pm.clip(1).round(2)})

@st.cache_data(show_spinner=False)
def demo_stock(n=650, seed=30):
    rng=np.random.default_rng(seed); date=pd.bdate_range("2023-01-02",periods=n); raw=rng.normal(.00045,.011,n)
    momentum=pd.Series(raw).rolling(5,min_periods=1).mean().to_numpy(); returns=.00025+.4*momentum+rng.normal(0,.012,n); close=100*np.exp(np.cumsum(returns))
    volume=rng.normal(2_000_000,280_000,n).clip(500_000); vol=pd.Series(returns).rolling(10,min_periods=2).std().fillna(0)
    return pd.DataFrame({"date":date,"close":close.round(2),"volume":volume.round(),"daily_return":returns.round(5),"rolling_volatility":vol.round(5)})

DEMO_DATASETS={"Student Performance":demo_students,"Health and Heart Disease":demo_health,"Finance and Loan Approval":demo_finance,
               "Environment and Air Quality":demo_air_quality,"Finance and Stock Prices":demo_stock}


def humanize(value):
    text=re.sub(r"^(num|cat)__","",str(value)).replace("_"," ").strip(); return text[:1].upper()+text[1:] if text else text

def axis_style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.tick_params(labelsize=10)

def show_fig(fig, filename, key):
    fig.tight_layout(); st.pyplot(fig,clear_figure=False,use_container_width=True); buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=300,bbox_inches="tight"); buf.seek(0)
    st.download_button("Download this figure",buf,filename,"image/png",key=key); plt.close(fig)

def guide(learn, tries, notices, explains, mistake=None):
    with st.expander("🧭 Activity guide", expanded=True):
        st.markdown(f"**What you will learn:** {learn}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Try this**")
            for item in tries:
                st.markdown(f"- {item}")
        with c2:
            st.markdown("**What to notice**")
            for item in notices:
                st.markdown(f"- {item}")
        with c3:
            st.markdown("**Explain your result**")
            for item in explains:
                st.markdown(f"- {item}")
        if mistake:
            st.warning(f"Common mistake to avoid: {mistake}")


def before_you_run(concepts, note=None, expanded=True, title="📘 Before you run: key terms and metrics"):
    """Show concise definitions before students choose settings or run an analysis."""
    with st.expander(title, expanded=expanded):
        columns = st.columns(2)
        for index, (term, explanation) in enumerate(concepts):
            with columns[index % 2]:
                st.markdown(f"**{term}**  \n{explanation}")
        if note:
            st.info(note)


def describe_time_spacing(values):
    """Return a student-friendly description of the typical spacing between ordered rows."""
    series = pd.Series(pd.to_datetime(values, errors="coerce")).dropna().sort_values().drop_duplicates()
    if len(series) < 2:
        return "The app could not determine the time spacing from this column."
    delta = series.diff().dropna().median()
    seconds = float(delta.total_seconds())
    if seconds <= 0:
        return "The app could not determine a positive time spacing from this column."
    units = [
        (86400 * 365.25, "year"),
        (86400 * 30.4375, "month"),
        (86400 * 7, "week"),
        (86400, "day"),
        (3600, "hour"),
        (60, "minute"),
        (1, "second"),
    ]
    for divisor, label in units:
        value = seconds / divisor
        if value >= 0.95:
            rounded = round(value, 2)
            shown = int(rounded) if float(rounded).is_integer() else rounded
            plural = label if shown == 1 else label + "s"
            return f"The typical spacing between rows is about **{shown} {plural}**."
    return "The rows are separated by less than one second."


def forecast_comparison_notes(result):
    """Explain the model-versus-baseline forecasting table in ordinary language."""
    if not isinstance(result, dict):
        return []
    table = result.get("comparison")
    if not isinstance(table, pd.DataFrame) or table.empty or "Root mean squared error" not in table:
        return []
    clean = table.dropna(subset=["Root mean squared error"]).copy()
    if clean.empty:
        return []
    clean = clean.sort_values("Root mean squared error", ascending=True)
    best = clean.iloc[0]
    model_name = result.get("model_name", "Selected model")
    notes = [
        f"The lowest root mean squared error is **{float(best['Root mean squared error']):.3f}**, achieved by **{best['Model']}**. Lower error is better.",
    ]
    selected = clean[clean["Model"] == model_name]
    baselines = clean[clean["Model"] != model_name]
    if not selected.empty and not baselines.empty:
        model_rmse = float(selected.iloc[0]["Root mean squared error"])
        best_baseline = baselines.iloc[0]
        baseline_rmse = float(best_baseline["Root mean squared error"] )
        difference = baseline_rmse - model_rmse
        if difference > 0:
            notes.append(
                f"The selected model beats the strongest simple baseline, **{best_baseline['Model']}**, by **{difference:.3f} root mean squared error units** on the final chronological test period."
            )
        elif difference < 0:
            notes.append(
                f"The strongest simple baseline, **{best_baseline['Model']}**, beats the selected model by **{abs(difference):.3f} root mean squared error units**. The more complex model did not add predictive value in this run."
            )
        else:
            notes.append("The selected model and the strongest simple baseline have the same root mean squared error in this run.")
    horizon = result.get("horizon")
    season = result.get("season")
    if horizon is not None:
        notes.append(f"The backtest predicts **{horizon} future row{'s' if horizon != 1 else ''} ahead** while preserving time order.")
    if season is not None:
        notes.append(f"The seasonal naïve comparison uses the target observed **{season} rows earlier**, representing one assumed repeating cycle.")
    return notes


def ensure_project_state():
    if "course_project" not in st.session_state:
        st.session_state.course_project = {
            "student_name": "",
            "main_question": "",
            "continuous_target": "",
            "classification_target": "",
            "candidate_predictors": [],
            "current_plans": {},
            "assignment_plans": {},
            "practice_plans": {},
            "today_progress": {},
            "practical_progress": {},
            "wrap_up_progress": {},
            "wrap_up_attempts": {},
            "exam_practice": {},
            "weeks": {},
        }
    project = st.session_state.course_project
    for key, default in {
        "student_name": "",
        "main_question": "",
        "continuous_target": "",
        "classification_target": "",
        "candidate_predictors": [],
        "current_plans": {},
        "assignment_plans": {},
        "practice_plans": {},
        "today_progress": {},
        "practical_progress": {},
        "wrap_up_progress": {},
        "wrap_up_attempts": {},
        "exam_practice": {},
        "weeks": {},
    }.items():
        project.setdefault(key, default)
    if "lab_briefs" not in st.session_state:
        st.session_state.lab_briefs = {}
    initialize_data_workspaces()
    return project

def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (pd.Timestamp,)): return value.isoformat()
    return value


def project_json():
    return json.dumps(json_safe(ensure_project_state()), indent=2)


def default_target_for_week(week, df):
    """Return a type-compatible fallback without preferring any named variable."""
    numeric = num_cols(df)
    classes = class_targets(df)
    if week in ["Week 5", "Week 6", "Week 7", "Week 14"]:
        return numeric[0] if numeric else ""
    if week in ["Week 9", "Week 10"]:
        return classes[0] if classes else ""
    return str(df.columns[0]) if len(df.columns) else ""

def default_predictors_for_target(df, target, maximum=5):
    """Choose only generic, valid columns; never assume a subject-specific name."""
    options = [c for c in df.columns if c != target]
    useful = []
    for col in options:
        lowered = str(col).lower()
        if lowered in {"id", "row_id", "index"} or lowered.endswith("_id"):
            continue
        useful.append(col)
    useful += [c for c in options if c not in useful]
    return useful[: min(maximum, len(useful))]

def default_brief(week, df):
    lab = WEEKLY_LABS[week]
    return {
        "week": week,
        "research_question": "",
        "targets": [],
        "predictors": [],
        "choice_mode": "Use the instructor's exact variables",
        "class_example": "",
        "instructions": lab["assignment"],
        "required_outputs": lab["required"],
        "duration": 60,
        "dataset_name": active_dataset_name(),
        "dataset_signature": dataframe_signature(df),
    }

def get_lab_brief(week, df):
    """Return the saved brief plus compatibility information for the active data."""
    ensure_project_state()
    if week not in st.session_state.lab_briefs:
        st.session_state.lab_briefs[week] = default_brief(week, df)
    brief = dict(st.session_state.lab_briefs[week])
    columns = set(df.columns)
    brief["missing_targets"] = [c for c in brief.get("targets", []) if c not in columns]
    brief["missing_predictors"] = [c for c in brief.get("predictors", []) if c not in columns]
    brief["dataset_matches"] = brief.get("dataset_signature") in {None, "", dataframe_signature(df)}
    return brief

def suitable_targets(week, df):
    if week in ["Week 5", "Week 6", "Week 7", "Week 14"]:
        return num_cols(df)
    if week in ["Week 9", "Week 10"]:
        return class_targets(df)
    return list(df.columns)


MODEL_SIMPLE_NOTES = {
    "Linear Regression": "Draws the best-fitting straight line or flat surface through the data.",
    "Ridge Regression": "A linear model that gently shrinks coefficients to reduce unstable predictions.",
    "Lasso Regression": "A linear model that can shrink some coefficients all the way to zero.",
    "Elastic Net": "Combines Ridge and Lasso ideas.",
    "Huber Regression": "A linear model designed to be less affected by unusual target values.",
    "Decision Tree": "Learns a sequence of simple yes-or-no decisions.",
    "Random Forest": "Combines many decision trees and averages their answers.",
    "Gradient Boosting": "Builds small trees one after another, with each new tree trying to correct earlier mistakes.",
    "Support Vector Regression": "Fits a prediction rule that can use straight or curved relationships.",
    "Support Vector Machine": "Finds a boundary that separates classes, possibly using a curved relationship.",
    "K-Nearest Neighbors": "Looks at the most similar training rows and uses their outcomes.",
    "Feedforward Neural Network (FFNN)": "Passes a fixed row of predictors through connected layers to make a prediction.",
    "Logistic Regression": "Estimates the probability of a class and uses a threshold to choose the final class.",
    "Linear Discriminant Analysis": "Finds linear combinations of predictors that separate classes.",
    "Quadratic Discriminant Analysis": "Allows each class to have a different curved separation pattern.",
    "Long Short-Term Memory Recurrent Neural Network (LSTM-RNN)": "Reads an ordered sequence of past values and learns which earlier time steps matter for a later forecast.",
    "Small Convolutional Neural Network (CNN)": "Keeps the image as a two-dimensional grid and learns local visual patterns such as edges, textures, and shapes before choosing a class.",
}


def model_simple_note(name):
    return MODEL_SIMPLE_NOTES.get(name, "This method learns a pattern from training data and is checked on unseen data.")


def simple_result_notes(result):
    """Create plain-language, result-specific statements for guided learning.

    The function accepts ordinary fitted-model results as well as resampling,
    bootstrap, and model-comparison results. It is used in Practical Studio and
    My Notebook so students see the same interpretation support after a run.
    """
    if not isinstance(result, dict):
        return []

    notes = []
    result_type = result.get("result_type", "")

    # Cross-validation and other resampling experiments.
    if result_type == "cross_validation" or (
        "scores" in result and "strategy" in result and "metric" in result
    ):
        scores = np.asarray(result.get("scores", []), dtype=float)
        scores = scores[np.isfinite(scores)]
        if not len(scores):
            return []
        metric = result.get("metric", "performance metric")
        strategy = result.get("strategy", "resampling")
        mean = float(scores.mean())
        sd = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
        direction = "Higher values are better for this metric." if result.get("higher_is_better", True) else "Lower values are better for this metric."
        if len(scores) == 1:
            notes.append(
                f"The **{metric}** is **{mean:.3f}** on the single held-out split. {direction}"
            )
            notes.append(
                "Only one split was used, so this run cannot show how much performance changes across different splits."
            )
        else:
            notes.append(
                f"Across **{len(scores)} resamples**, the mean **{metric}** is **{mean:.3f}**. {direction}"
            )
            scale = max(abs(mean), 1e-12)
            relative_sd = abs(sd) / scale
            if relative_sd <= 0.05:
                stability = "very stable across the resamples"
            elif relative_sd <= 0.15:
                stability = "reasonably stable, although the split still matters"
            else:
                stability = "quite variable across resamples, so performance depends strongly on the split"
            notes.append(
                f"The cross-validation standard deviation is **{sd:.3f}**. This means the scores were **{stability}**."
            )
            notes.append(
                f"The lowest and highest resample scores were **{scores.min():.3f}** and **{scores.max():.3f}**. This range shows the observed spread, not a confidence interval."
            )
        notes.append(
            f"The validation strategy was **{strategy}**. Use the average score to describe typical performance and the variation to describe stability."
        )
        return notes

    # Bootstrap results for a statistic, slope, model metric, or prediction.
    if result_type == "bootstrap" or (
        "values" in result and "original" in result and "label" in result
    ):
        values = np.asarray(result.get("values", []), dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return []
        original = float(result.get("original", np.mean(values)))
        label = result.get("label", "quantity")
        bias = float(values.mean() - original)
        standard_error = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        lo, hi = np.percentile(values, [2.5, 97.5])
        width = float(hi - lo)
        notes.append(
            f"The original **{label}** estimate is **{original:.4f}**. This is the value calculated from the observed sample before resampling."
        )
        if standard_error > 0:
            bias_ratio = abs(bias) / standard_error
            if bias_ratio <= 0.25:
                bias_reading = "small compared with the bootstrap standard error"
            elif bias_ratio <= 1:
                bias_reading = "noticeable but not larger than one bootstrap standard error"
            else:
                bias_reading = "large compared with the bootstrap standard error"
        else:
            bias_reading = "not meaningfully assessable because the resamples showed no variation"
        direction = "above" if bias > 0 else "below" if bias < 0 else "equal to"
        notes.append(
            f"The estimated bias is **{bias:.4f}**. The average bootstrap estimate is {abs(bias):.4f} {direction} the original estimate, so the observed bias is **{bias_reading}**."
        )
        notes.append(
            f"The bootstrap standard error is **{standard_error:.4f}**. It estimates how much the {label.lower()} would typically change if comparable samples were repeatedly drawn."
        )
        notes.append(
            f"The percentile 95% interval is **[{lo:.4f}, {hi:.4f}]**, with width **{width:.4f}**. It is an uncertainty interval for the estimated quantity, not a range expected to contain 95% of individual observations."
        )
        if result.get("quantity_kind") == "slope":
            if lo > 0:
                notes.append("The entire interval is above zero, so the bootstrap results consistently support a positive slope in these resamples.")
            elif hi < 0:
                notes.append("The entire interval is below zero, so the bootstrap results consistently support a negative slope in these resamples.")
            else:
                notes.append("The interval includes zero, so the resampled slopes do not consistently rule out a near-zero relationship.")
        return notes

    # Model-comparison table.
    if isinstance(result.get("table"), pd.DataFrame) and result.get("best") and result.get("metric"):
        table = result["table"]
        best = result["best"]
        metric = result["metric"]
        match = table.loc[table.get("Model", pd.Series(dtype=str)).astype(str) == str(best)]
        if len(match) and "Test performance" in match.columns:
            value = match.iloc[0]["Test performance"]
            if pd.notna(value):
                notes.append(
                    f"The best held-out model by **{metric}** is **{best}**, with test performance **{float(value):.3f}**."
                )
        notes.append(
            "The recommendation is based on held-out performance using the same target, predictors, split, and metric for every model."
        )
        if "Cross-validation standard deviation" in table.columns:
            row = match.iloc[0] if len(match) else None
            if row is not None and pd.notna(row.get("Cross-validation standard deviation")):
                notes.append(
                    f"Its cross-validation standard deviation is **{float(row['Cross-validation standard deviation']):.3f}**; smaller variation means the result is more stable across folds."
                )
        return notes

    problem = result.get("problem")
    metrics = result.get("metrics", {})
    model_name = result.get("model_name")
    if model_name:
        notes.append(f"**How the model works:** {model_simple_note(model_name)}")
    if problem == "regression":
        if model_name == "Linear Regression" and result.get("context") == "linear regression":
            try:
                coefficients = linear_coefficient_table(result)
                slopes = coefficients[coefficients["Term"] != "Intercept"]
                if len(slopes):
                    row = slopes.iloc[0]
                    notes.append(f"The first slope is **{row['Estimate']:.3f}**. {row['Meaning']}")
            except Exception:
                pass
        mae = metrics.get("Mean absolute error")
        rmse = metrics.get("Root mean squared error")
        r2 = metrics.get("R-squared")
        if mae is not None:
            notes.append(f"The mean absolute error is **{mae:.3f}**. On average, predictions miss the observed target by about {mae:.3f} target units.")
        if rmse is not None:
            notes.append(f"The root mean squared error is **{rmse:.3f}**. It gives extra weight to larger prediction mistakes.")
        if r2 is not None:
            if r2 < 0:
                notes.append(f"R-squared is **{r2:.3f}**. On the held-out rows, this model performed worse than a simple mean prediction.")
            elif r2 < .25:
                notes.append(f"R-squared is **{r2:.3f}**. The model explains only a small part of the held-out variation.")
            elif r2 < .60:
                notes.append(f"R-squared is **{r2:.3f}**. The model explains a meaningful but incomplete part of the held-out variation.")
            else:
                notes.append(f"R-squared is **{r2:.3f}**. The model explains a large share of the held-out variation, but the remaining errors still matter.")
        comparison = result.get("comparison")
        if isinstance(comparison, pd.DataFrame) and "Root mean squared error" in comparison.columns and len(comparison) > 1:
            ranked = comparison.sort_values("Root mean squared error")
            best = ranked.iloc[0]
            notes.append(f"The lowest forecast RMSE is **{best['Root mean squared error']:.3f}**, produced by **{best['Model']}**.")
    elif problem == "classification":
        for metric in ["Accuracy", "Precision", "Recall", "F1-score"]:
            if metric in metrics:
                notes.append(f"{metric} is **{metrics[metric]:.3f}**. " + ({
                    "Accuracy": "This is the fraction of all held-out predictions that are correct.",
                    "Precision": "Among predicted positives, this is the fraction that are truly positive.",
                    "Recall": "Among observed positives, this is the fraction the model correctly finds.",
                    "F1-score": "This balances precision and recall.",
                }[metric]))
    return notes

def week_result(week):
    mapping = {
        "Week 5": "linear_result",
        "Week 6": "linear_result",
        "Week 7": "mlr_result",
        "Week 9": "log_result",
        "Week 10": "mlc_result",
        "Week 11": "latest_model_result",
        "Week 12": "cv_result",
        "Week 13": "bootstrap",
        "Week 14": "forecast_result",
        "Week 15": "image_result",
    }
    key = mapping.get(week)
    return st.session_state.get(key) if key else None


def _tool_plan_stamp_key(week, context, scope=None):
    scope = scope or st.session_state.get("active_dataset_scope", "class")
    return "_tool_plan_seed_" + re.sub(r"[^A-Za-z0-9_]+", "_", f"{scope}_{context}_{week}")


def _tool_plan_anchor_keys(week):
    """Widget keys that remain visible while a week's main tool is open.

    Streamlit removes widget state when a widget is not rendered. These stable keys
    let the app detect that a student has returned to the analysis step and reseed the
    saved plan once, while still preserving any choices made during the active session.
    """
    return {
        "Week 2": ["question_target"],
        "Week 3": ["chart"],
        "Week 4": ["assoc_x", "assoc_y"],
        "Week 5": ["lin_target", "lin_mode"],
        "Week 6": ["lin_target", "lin_mode"],
        "Week 7": ["mlr_target"],
        "Week 9": ["log_target"],
        "Week 10": ["mlc_target"],
        "Week 11": ["cmp_problem", "cmp_target"],
        "Week 12": ["cv_problem", "cv_target"],
        "Week 13": ["boot_mode"],
        "Week 14": ["fc_target"],
    }.get(week, [])


def apply_plan_to_tool(week, plan, context="guided", force=False):
    """Load a saved plan as the starting values for the analytical controls.

    The plan is seeded when the analysis page is first opened, when the saved plan or
    dataset changes, or when the student returns after visiting another assignment
    step. While the tool remains open, the student's own target, predictor, chart, and
    model changes are preserved and are never forced back on each Streamlit rerun.
    """
    target = plan.get("target")
    predictors = [c for c in plan.get("predictors", []) if c != target]
    df = get_df()
    scope = st.session_state.get("active_dataset_scope", "class")
    signature_payload = {
        "week": week,
        "scope": scope,
        "context": context,
        "target": target,
        "predictors": predictors,
        "dataset_signature": plan.get("dataset_signature") or dataframe_signature(df),
    }
    signature = hashlib.sha256(
        json.dumps(json_safe(signature_payload), sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    stamp_key = _tool_plan_stamp_key(week, context, scope)
    anchors = _tool_plan_anchor_keys(week)
    controls_are_open = bool(anchors) and all(key in st.session_state for key in anchors)
    if not force and st.session_state.get(stamp_key) == signature and (controls_are_open or not anchors):
        return False

    def set_if_valid(key, value, options=None):
        if value is None:
            return
        if options is not None and value not in options:
            return
        st.session_state[key] = value

    if week == "Week 2":
        set_if_valid("question_target", target, list(df.columns))
        st.session_state["question_pred"] = [c for c in predictors if c in df.columns]
    elif week == "Week 3":
        if target in num_cols(df) and predictors:
            first = predictors[0]
            if first in num_cols(df):
                st.session_state["chart"] = "Scatter plot"
                st.session_state["scatter_x"] = first
                st.session_state["scatter_y"] = target
            elif first in cat_cols(df):
                st.session_state["chart"] = "Boxplot"
                st.session_state["box_y"] = target
                st.session_state["box_g"] = first
            else:
                st.session_state["chart"] = "Histogram"
                st.session_state["hist_x"] = target
        elif target in num_cols(df):
            st.session_state["chart"] = "Histogram"
            st.session_state["hist_x"] = target
    elif week == "Week 4":
        if target and predictors:
            set_if_valid("assoc_x", predictors[0], list(df.columns))
            set_if_valid("assoc_y", target, [c for c in df.columns if c != predictors[0]])
    elif week == "Week 5":
        st.session_state["lin_mode"] = "Simple regression"
        set_if_valid("lin_target", target, num_cols(df))
        numerical = [c for c in predictors if c in num_cols(df)]
        if numerical:
            st.session_state["lin_single"] = numerical[0]
    elif week == "Week 6":
        st.session_state["lin_mode"] = "Multiple regression"
        set_if_valid("lin_target", target, num_cols(df))
        st.session_state["lin_features"] = [c for c in predictors if c in df.columns]
    elif week == "Week 7":
        set_if_valid("mlr_target", target, num_cols(df))
        st.session_state["mlr_features"] = [c for c in predictors if c in df.columns]
    elif week == "Week 9":
        set_if_valid("log_target", target, class_targets(df))
        st.session_state["log_features"] = [c for c in predictors if c in df.columns]
    elif week == "Week 10":
        set_if_valid("mlc_target", target, class_targets(df))
        st.session_state["mlc_features"] = [c for c in predictors if c in df.columns]
    elif week == "Week 11":
        if target in num_cols(df):
            st.session_state["cmp_problem"] = "Regression"
        else:
            st.session_state["cmp_problem"] = "Classification"
        set_if_valid("cmp_target", target, list(df.columns))
        st.session_state["cmp_features"] = [c for c in predictors if c in df.columns]
    elif week == "Week 12":
        if target in num_cols(df):
            st.session_state["cv_problem"] = "Regression"
        else:
            st.session_state["cv_problem"] = "Classification"
        set_if_valid("cv_target", target, list(df.columns))
        st.session_state["cv_features"] = [c for c in predictors if c in df.columns]
    elif week == "Week 13":
        set_if_valid("boot_var", target, num_cols(df))
        if target in num_cols(df):
            st.session_state["boot_train_problem"] = "Regression"
            set_if_valid("boot_train_target", target, num_cols(df))
        elif target in class_targets(df):
            st.session_state["boot_train_problem"] = "Classification"
            set_if_valid("boot_train_target", target, class_targets(df))
        st.session_state["boot_train_features"] = [c for c in predictors if c in df.columns and c != target]
    elif week == "Week 14":
        set_if_valid("fc_target", target, num_cols(df))
        datelike = [
            c for c in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[c])
            or "date" in c.lower()
            or "time" in c.lower()
        ]
        if datelike:
            st.session_state["fc_date"] = datelike[0]
        st.session_state["fc_exog_tabular"] = [c for c in predictors if c in df.columns and c != target]
        st.session_state["fc_exog_lstm"] = [c for c in predictors if c in num_cols(df) and c != target]

    st.session_state[stamp_key] = signature
    return True

def dataframe_signature(df):
    if not isinstance(df, pd.DataFrame):
        return ""
    schema = "|".join(f"{c}:{df[c].dtype}" for c in df.columns)
    return hashlib.sha256(f"{len(df)}|{schema}".encode("utf-8")).hexdigest()[:16]


def initialize_data_workspaces():
    if "class_df" not in st.session_state:
        st.session_state.class_df = demo_students()
        st.session_state.class_dataset_name = "Student Performance"
    if "notebook_df" not in st.session_state:
        st.session_state.notebook_df = demo_students()
        st.session_state.notebook_dataset_name = "Student Performance"
    if "studio_df" not in st.session_state:
        st.session_state.studio_df = demo_students()
        st.session_state.studio_dataset_name = "Student Performance"
    st.session_state.setdefault("active_dataset_scope", "class")
    st.session_state.setdefault("scope_analysis_cache", {})
    scope = st.session_state.active_dataset_scope
    st.session_state.df = st.session_state[f"{scope}_df"].copy()
    st.session_state.dataset_name = st.session_state[f"{scope}_dataset_name"]


def _analysis_keys_in_session():
    keys = set()
    for key in st.session_state.keys():
        if key in ANALYSIS_RESULT_KEYS or key.endswith("_result"):
            keys.add(key)
        if key.endswith(("_perm_table", "_shap_table", "_explanation_signature")):
            keys.add(key)
    return keys


def activate_dataset_scope(scope):
    initialize_data_workspaces()
    if scope not in {"class", "notebook", "studio"}:
        raise ValueError(f"Unknown dataset scope: {scope}")
    current = st.session_state.get("active_dataset_scope")
    if current != scope:
        cache = st.session_state.setdefault("scope_analysis_cache", {})
        if current:
            cache[current] = {key: st.session_state[key] for key in _analysis_keys_in_session()}
        clear_analysis_results()
        for key, value in cache.get(scope, {}).items():
            st.session_state[key] = value
        st.session_state.active_dataset_scope = scope
    st.session_state.df = st.session_state[f"{scope}_df"].copy()
    st.session_state.dataset_name = st.session_state[f"{scope}_dataset_name"]


def set_scope_dataset(scope, df, name):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("The selected dataset is empty.")
    initialize_data_workspaces()
    st.session_state[f"{scope}_df"] = df.copy()
    st.session_state[f"{scope}_dataset_name"] = str(name)
    st.session_state.setdefault("scope_analysis_cache", {}).pop(scope, None)
    if st.session_state.get("active_dataset_scope") == scope:
        clear_analysis_results()
        st.session_state.df = df.copy()
        st.session_state.dataset_name = str(name)
    project = ensure_project_state() if "course_project" in st.session_state else None
    if project is not None:
        if scope == "class":
            project["practice_plans"] = {}
            project["today_progress"] = {}
            project["practical_progress"] = {}
            project["wrap_up_progress"] = {}
            project["wrap_up_attempts"] = {}
        elif scope == "notebook":
            project["assignment_plans"] = {}


def scope_dataset(scope):
    initialize_data_workspaces()
    return (
        st.session_state[f"{scope}_df"].copy(),
        st.session_state[f"{scope}_dataset_name"],
    )


def active_dataset_name():
    initialize_data_workspaces()
    return st.session_state.get("dataset_name", "Dataset")


def get_df():
    initialize_data_workspaces()
    scope = st.session_state.get("active_dataset_scope", "class")
    st.session_state.df = st.session_state[f"{scope}_df"].copy()
    st.session_state.dataset_name = st.session_state[f"{scope}_dataset_name"]
    return st.session_state.df.copy()

def num_cols(df): return df.select_dtypes(include=np.number).columns.tolist()
def cat_cols(df): return df.select_dtypes(exclude=np.number).columns.tolist()
def class_targets(df,max_unique=20):
    return [c for c in df.columns if (not pd.api.types.is_numeric_dtype(df[c]) and df[c].dropna().nunique()>=2) or (pd.api.types.is_numeric_dtype(df[c]) and 2<=df[c].dropna().nunique()<=max_unique)]

def onehot(drop=None):
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=drop)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, drop=drop)


def preprocessor(df, features, scale_numeric=True, drop_first=False):
    nums = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cats = [c for c in features if c not in nums]
    transformers = []
    if nums:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scale", StandardScaler()))
        transformers.append(("num", Pipeline(numeric_steps), nums))
    if cats:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot(drop="first" if drop_first else None)),
        ]), cats))
    return ColumnTransformer(transformers, remainder="drop")


def split_xy(df, target, features):
    clean = df[list(features) + [target]].dropna(subset=[target]).copy()
    return clean[list(features)], clean[target]


def feature_names(pipe):
    try:
        return pipe.named_steps["prep"].get_feature_names_out().tolist()
    except Exception:
        return []


def original_name(transformed, originals):
    clean = re.sub(r"^(num|cat)__", "", str(transformed))
    for name in sorted(originals, key=len, reverse=True):
        if clean == name or clean.startswith(name + "_"):
            return name
    return clean


def reg_metrics(y, p):
    return {
        "Mean absolute error": mean_absolute_error(y, p),
        "Root mean squared error": math.sqrt(mean_squared_error(y, p)),
        "R-squared": r2_score(y, p),
    }


def clf_metrics(y, p, proba=None):
    labels = np.unique(y)
    avg = "binary" if len(labels) == 2 else "weighted"
    out = {
        "Accuracy": accuracy_score(y, p),
        "Balanced accuracy": balanced_accuracy_score(y, p),
        "Precision": precision_score(y, p, average=avg, zero_division=0),
        "Recall": recall_score(y, p, average=avg, zero_division=0),
        "F1-score": f1_score(y, p, average=avg, zero_division=0),
    }
    if proba is not None and len(labels) == 2:
        out["ROC area under the curve"] = roc_auc_score(y, proba[:, 1])
    return out


def tidy_frame(frame, decimals=3):
    out = pd.DataFrame(frame).copy()
    numeric = out.select_dtypes(include=np.number).columns
    if len(numeric):
        out[numeric] = out[numeric].round(decimals)
    return out


def metric_table(metrics):
    return tidy_frame(pd.DataFrame({"Metric": metrics.keys(), "Value": metrics.values()}), 3)


ANALYSIS_RESULT_KEYS = {
    "linear_result", "mlr_result", "log_result", "mlc_result", "selection_result",
    "comparison", "cv_result", "bootstrap", "forecast_result", "latest_model_result",
}


def clear_explanation_outputs():
    suffixes = ("_perm_table", "_shap_table", "_perm", "_shap", "_explanation_signature")
    for key in list(st.session_state.keys()):
        if key.endswith(suffixes):
            del st.session_state[key]


def clear_analysis_results():
    for key in list(st.session_state.keys()):
        if key in ANALYSIS_RESULT_KEYS or key.endswith("_result"):
            del st.session_state[key]
    clear_explanation_outputs()


def result_signature(result):
    return repr((
        result.get("dataset_name"),
        result.get("target"),
        result.get("model_name"),
        tuple(result.get("features", [])),
        result.get("context"),
        result.get("parameters", {}),
    ))


def save_result(key, result):
    clear_explanation_outputs()
    st.session_state[key] = result
    st.session_state.latest_model_result = result

REG_MODELS=["Linear Regression","Ridge Regression","Lasso Regression","Elastic Net","Huber Regression","Decision Tree","Random Forest","Gradient Boosting","Support Vector Regression","K-Nearest Neighbors","Feedforward Neural Network (FFNN)"]
CLF_MODELS=["Logistic Regression","Linear Discriminant Analysis","Quadratic Discriminant Analysis","K-Nearest Neighbors","Decision Tree","Random Forest","Gradient Boosting","Support Vector Machine","Feedforward Neural Network (FFNN)"]

def params_ui(name,problem,prefix,expanded=False):
    p={}
    with st.expander(f"⚙️ Parameters for {name}",expanded=expanded):
        if name=="Linear Regression": st.caption("No main complexity parameter.")
        elif name in ["Ridge Regression","Lasso Regression"]: p["alpha"]=st.number_input("Regularization strength (alpha)",.0001,1000.,1. if name.startswith("Ridge") else .1,step=.1,key=prefix+"a")
        elif name=="Elastic Net": p["alpha"]=st.number_input("Alpha",.0001,1000.,.1,step=.1,key=prefix+"a"); p["l1_ratio"]=st.slider("L1 ratio",0.,1.,.5,.05,key=prefix+"l1")
        elif name=="Huber Regression": p["epsilon"]=st.slider("Outlier sensitivity",1.01,3.,1.35,.05,key=prefix+"eps"); p["alpha"]=st.number_input("Alpha",0.,10.,.0001,format="%.4f",key=prefix+"a")
        elif name=="Decision Tree":
            d=st.slider("Maximum depth; 0 is unlimited",0,30,5,key=prefix+"d"); p["max_depth"]=None if d==0 else d; p["min_leaf"]=st.slider("Minimum observations per leaf",1,30,2,key=prefix+"leaf")
        elif name=="Random Forest":
            p["trees"]=st.slider("Number of trees",50,600,200,50,key=prefix+"trees"); d=st.slider("Maximum depth; 0 is unlimited",0,30,0,key=prefix+"d"); p["max_depth"]=None if d==0 else d
            p["max_features"]=st.selectbox("Predictors per split",["sqrt","log2",1.0],key=prefix+"mf"); p["min_leaf"]=st.slider("Minimum observations per leaf",1,30,1,key=prefix+"leaf")
        elif name=="Gradient Boosting":
            p["trees"]=st.slider("Boosting stages",50,600,150,25,key=prefix+"trees"); p["learning_rate"]=st.slider("Learning rate",.01,.3,.05,.01,key=prefix+"lr"); p["max_depth"]=st.slider("Maximum depth",1,12,3,key=prefix+"d")
        elif name in ["Support Vector Regression","Support Vector Machine"]:
            p["C"]=st.number_input("Penalty C",.01,1000.,1.,step=.1,key=prefix+"C"); p["kernel"]=st.selectbox("Kernel",["rbf","linear","poly"],key=prefix+"ker"); p["gamma"]=st.selectbox("Gamma",["scale","auto"],key=prefix+"gam")
            if problem=="regression": p["epsilon"]=st.number_input("Epsilon width",0.,10.,.1,step=.05,key=prefix+"eps")
            else: p["class_weight"]=None if st.selectbox("Class weighting",["None","balanced"],key=prefix+"cw")=="None" else "balanced"
        elif name=="K-Nearest Neighbors": p["neighbors"]=st.slider("Number of neighbors",1,50,5,key=prefix+"n"); p["weights"]=st.selectbox("Weights",["uniform","distance"],key=prefix+"w")
        elif name=="Feedforward Neural Network (FFNN)":
            layers = st.radio("Hidden layers", [1, 2], horizontal=True, key=prefix+"layers")
            hidden_1 = st.slider("Neurons in first hidden layer", 4, 256, 32, 4, key=prefix+"h1")
            hidden_2 = st.slider("Neurons in second hidden layer", 4, 256, 16, 4, key=prefix+"h2") if layers == 2 else None
            p["hidden_layers"] = (hidden_1,) if layers == 1 else (hidden_1, hidden_2)
            p["activation"] = st.selectbox("Activation function", ["relu", "tanh", "logistic"], key=prefix+"act")
            p["alpha"] = st.number_input("Regularization alpha", .00001, 10., .0001, format="%.5f", key=prefix+"a")
            p["learning_rate_init"] = st.number_input("Initial learning rate", .00001, 1., .001, format="%.5f", key=prefix+"nnlr")
            p["iterations"] = st.slider("Maximum iterations", 100, 2000, 500, 100, key=prefix+"it")
            st.caption("An FFNN passes a fixed predictor row forward through hidden layers. It has no recurrent memory.")
        elif name=="Logistic Regression": p["C"]=st.number_input("Inverse regularization C",.001,1000.,1.,step=.1,key=prefix+"C"); p["penalty"]=st.selectbox("Penalty",["l2","l1"],key=prefix+"pen"); p["class_weight"]=None if st.selectbox("Class weighting",["None","balanced"],key=prefix+"cw")=="None" else "balanced"
        elif name=="Linear Discriminant Analysis": p["solver"]=st.selectbox("Solver",["svd","lsqr"],key=prefix+"sol"); p["shrinkage"]="auto" if p["solver"]=="lsqr" and st.checkbox("Use automatic shrinkage",key=prefix+"sh") else None
        elif name=="Quadratic Discriminant Analysis": p["reg_param"]=st.slider("Covariance regularization",0.,1.,0.,.05,key=prefix+"reg")
        st.caption("Only parameters used by the selected model are shown.")
    return p

def reg_model(name,p,seed=42):
    if name=="Linear Regression": return LinearRegression()
    if name=="Ridge Regression": return Ridge(alpha=p.get("alpha",1.))
    if name=="Lasso Regression": return Lasso(alpha=p.get("alpha",.1),max_iter=20000)
    if name=="Elastic Net": return ElasticNet(alpha=p.get("alpha",.1),l1_ratio=p.get("l1_ratio",.5),max_iter=20000)
    if name=="Huber Regression": return HuberRegressor(epsilon=p.get("epsilon",1.35),alpha=p.get("alpha",.0001),max_iter=1000)
    if name=="Decision Tree": return DecisionTreeRegressor(max_depth=p.get("max_depth"),min_samples_leaf=p.get("min_leaf",2),random_state=seed)
    if name=="Random Forest": return RandomForestRegressor(n_estimators=p.get("trees",200),max_depth=p.get("max_depth"),max_features=p.get("max_features","sqrt"),min_samples_leaf=p.get("min_leaf",1),random_state=seed,n_jobs=-1)
    if name=="Gradient Boosting": return GradientBoostingRegressor(n_estimators=p.get("trees",150),learning_rate=p.get("learning_rate",.05),max_depth=p.get("max_depth",3),random_state=seed)
    if name=="Support Vector Regression": return SVR(C=p.get("C",1.),kernel=p.get("kernel","rbf"),gamma=p.get("gamma","scale"),epsilon=p.get("epsilon",.1))
    if name=="K-Nearest Neighbors": return KNeighborsRegressor(n_neighbors=p.get("neighbors",5),weights=p.get("weights","uniform"))
    if name=="Feedforward Neural Network (FFNN)": return MLPRegressor(hidden_layer_sizes=p.get("hidden_layers",(32,)),activation=p.get("activation","relu"),alpha=p.get("alpha",.0001),learning_rate_init=p.get("learning_rate_init",.001),max_iter=p.get("iterations",500),early_stopping=True,random_state=seed)
    return LinearRegression()

def clf_model(name,p,seed=42):
    if name=="Logistic Regression":
        pen=p.get("penalty","l2"); return LogisticRegression(C=p.get("C",1.),penalty=pen,solver="liblinear" if pen=="l1" else "lbfgs",class_weight=p.get("class_weight"),max_iter=1500,random_state=seed)
    if name=="Linear Discriminant Analysis": return LinearDiscriminantAnalysis(solver=p.get("solver","svd"),shrinkage=p.get("shrinkage"))
    if name=="Quadratic Discriminant Analysis": return QuadraticDiscriminantAnalysis(reg_param=p.get("reg_param",0.))
    if name=="K-Nearest Neighbors": return KNeighborsClassifier(n_neighbors=p.get("neighbors",5),weights=p.get("weights","uniform"))
    if name=="Decision Tree": return DecisionTreeClassifier(max_depth=p.get("max_depth"),min_samples_leaf=p.get("min_leaf",2),random_state=seed)
    if name=="Random Forest": return RandomForestClassifier(n_estimators=p.get("trees",200),max_depth=p.get("max_depth"),max_features=p.get("max_features","sqrt"),min_samples_leaf=p.get("min_leaf",1),random_state=seed,n_jobs=-1)
    if name=="Gradient Boosting": return GradientBoostingClassifier(n_estimators=p.get("trees",150),learning_rate=p.get("learning_rate",.05),max_depth=p.get("max_depth",3),random_state=seed)
    if name=="Support Vector Machine": return SVC(C=p.get("C",1.),kernel=p.get("kernel","rbf"),gamma=p.get("gamma","scale"),class_weight=p.get("class_weight"),probability=True,random_state=seed)
    if name=="Feedforward Neural Network (FFNN)": return MLPClassifier(hidden_layer_sizes=p.get("hidden_layers",(32,)),activation=p.get("activation","relu"),alpha=p.get("alpha",.0001),learning_rate_init=p.get("learning_rate_init",.001),max_iter=p.get("iterations",500),early_stopping=True,random_state=seed)
    return LogisticRegression(max_iter=1500)

def build_pipe(X, features, name, problem, p, seed=42):
    unscaled_models = {"Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"}
    drop_first = name in {"Linear Regression", "Logistic Regression"}
    prep = preprocessor(X, features, scale_numeric=name not in unscaled_models, drop_first=drop_first)
    model = reg_model(name, p, seed) if problem == "regression" else clf_model(name, p, seed)
    return Pipeline([("prep", prep), ("model", model)])

# -----------------------------------------------------------------------------
# Explanations, sidebar, and foundational pages
# -----------------------------------------------------------------------------
def prepare_explanation_state(result, prefix):
    signature_key = prefix + "_explanation_signature"
    signature = result_signature(result)
    if st.session_state.get(signature_key) != signature:
        for key in [prefix + "_perm_table", prefix + "_shap_table", prefix + "_perm", prefix + "_shap"]:
            st.session_state.pop(key, None)
        st.session_state[signature_key] = signature


def shap_summary(explanation, transformed_names, original_features):
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, -1]
    mapping = [original_name(name, original_features) for name in transformed_names]
    rows = []
    for feature in original_features:
        indexes = [i for i, mapped in enumerate(mapping) if mapped == feature]
        if not indexes:
            continue
        contributions = values[:, indexes].sum(axis=1)
        mean_abs = float(np.mean(np.abs(contributions)))
        mean_signed = float(np.mean(contributions))
        positive_share = float(np.mean(contributions > 0))
        if positive_share >= 0.60:
            direction = "usually raises prediction"
        elif positive_share <= 0.40:
            direction = "usually lowers prediction"
        else:
            direction = "mixed direction"
        rows.append({
            "Predictor": humanize(feature),
            "Mean absolute SHAP contribution": mean_abs,
            "Average signed contribution": mean_signed,
            "Direction across observations": direction,
        })
    return tidy_frame(pd.DataFrame(rows).sort_values("Mean absolute SHAP contribution", ascending=False), 3)


def explanation_panel(result, prefix):
    st.subheader("How the model used the predictors")
    prepare_explanation_state(result, prefix)
    tabs = st.tabs(["Permutation importance", "SHAP summary"])

    with tabs[0]:
        metric_wording = (
            "increase in root mean squared error"
            if result["problem"] == "regression"
            else "decrease in weighted F1-score"
        )
        st.info(
            "Permutation importance shuffles one predictor at a time in the held-out data. "
            f"A larger {metric_wording} means the trained model relied more strongly on that predictor."
        )
        repeats = st.slider("Permutation repetitions", 3, 30, 10, key=prefix + "_rep")
        if st.button("Calculate permutation importance", key=prefix + "_perm_btn"):
            try:
                scoring = "neg_root_mean_squared_error" if result["problem"] == "regression" else "f1_weighted"
                values = permutation_importance(
                    result["pipeline"],
                    result["X_test"],
                    result["y_test"],
                    n_repeats=repeats,
                    random_state=42,
                    scoring=scoring,
                )
                table = pd.DataFrame({
                    "Predictor": [humanize(v) for v in result["features"]],
                    "Mean importance": values.importances_mean,
                    "Variation across shuffles": values.importances_std,
                }).sort_values("Mean importance", ascending=False)
                st.session_state[prefix + "_perm_table"] = tidy_frame(table, 3)
            except Exception as exc:
                st.error(f"Permutation importance failed: {exc}")
        table = st.session_state.get(prefix + "_perm_table")
        if isinstance(table, pd.DataFrame):
            st.dataframe(tidy_frame(table,3), use_container_width=True, hide_index=True)
            if len(table) == 1:
                value = float(table.iloc[0]["Mean importance"])
                st.metric("Permutation importance", f"{value:.3f}")
                st.caption("There is only one predictor, so a comparison bar chart would add no useful information.")
            else:
                top = table.head(10).sort_values("Mean importance")
                fig, ax = plt.subplots(figsize=(8, max(3.2, 0.48 * len(top) + 1.2)))
                ax.barh(top["Predictor"], top["Mean importance"], xerr=top["Variation across shuffles"])
                ax.set_xlabel(metric_wording.capitalize())
                ax.set_ylabel("")
                axis_style(ax)
                show_fig(fig, prefix + "_permutation.png", prefix + "_perm_dl")
                st.caption(
                    "Longer bars indicate greater reliance by the fitted model. Error bars show how much the result "
                    "changed across repeated shuffles. Correlated predictors may share importance."
                )

    with tabs[1]:
        st.info(
            "SHAP estimates how much each predictor contributes to the model's predictions across the held-out observations. "
            "The summary below reports the average contribution size and whether contributions usually raise or lower predictions."
        )
        if not SHAP_AVAILABLE:
            st.info("Add shap to requirements.txt to activate this optional section.")
        else:
            n = st.slider("Maximum held-out observations", 20, 150, 60, 10, key=prefix + "_shap_n")
            if st.button("Calculate SHAP summary", key=prefix + "_shap_btn"):
                try:
                    pipe = result["pipeline"]
                    prep = pipe.named_steps["prep"]
                    model = pipe.named_steps["model"]
                    train = np.asarray(prep.transform(result["X_train"]))
                    test = np.asarray(prep.transform(result["X_test"].iloc[:n]))
                    names = feature_names(pipe)
                    background = train[: min(100, len(train))]
                    if hasattr(model, "coef_") or hasattr(model, "feature_importances_"):
                        explainer = shap.Explainer(model, background, feature_names=names)
                    else:
                        prediction_function = (
                            (lambda values: model.predict_proba(values)[:, 1])
                            if result["problem"] == "classification"
                            and result.get("n_classes", 2) == 2
                            and hasattr(model, "predict_proba")
                            else model.predict
                        )
                        explainer = shap.Explainer(prediction_function, background, feature_names=names)
                    explanation = explainer(test)
                    st.session_state[prefix + "_shap_table"] = shap_summary(
                        explanation, names, result["features"]
                    )
                except Exception as exc:
                    st.error(f"SHAP failed: {exc}")
            table = st.session_state.get(prefix + "_shap_table")
            if isinstance(table, pd.DataFrame):
                st.dataframe(tidy_frame(table,3), use_container_width=True, hide_index=True)
                if len(table) == 1:
                    value = float(table.iloc[0]["Mean absolute SHAP contribution"])
                    st.metric("Mean absolute SHAP contribution", f"{value:.3f}")
                    st.caption("With one predictor, the table is clearer than a one-bar chart.")
                else:
                    top = table.head(10).sort_values("Mean absolute SHAP contribution")
                    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.48 * len(top) + 1.2)))
                    ax.barh(top["Predictor"], top["Mean absolute SHAP contribution"])
                    ax.set_xlabel("Mean absolute SHAP contribution")
                    ax.set_ylabel("")
                    axis_style(ax)
                    show_fig(fig, prefix + "_shap_summary.png", prefix + "_shap_dl")
                    st.caption(
                        "Longer bars mean larger average contributions to predictions. The direction column in the table "
                        "shows whether contributions usually raise or lower predictions. SHAP explains the model, not causation."
                    )

def render_dataset_sidebar():
    activate_dataset_scope("studio")
    st.sidebar.subheader("Full Studio dataset")
    source = st.sidebar.radio("Data source", ["Demo dataset", "Upload CSV or Excel"], key="studio_source")
    if source == "Demo dataset":
        name = st.sidebar.selectbox("Demo", list(DEMO_DATASETS), key="studio_demo_name")
        if st.sidebar.button("Load demo", use_container_width=True, key="studio_load_demo"):
            set_scope_dataset("studio", DEMO_DATASETS[name](), name)
            st.sidebar.success("Loaded")
            st.rerun()
    else:
        uploaded = st.sidebar.file_uploader("Upload file", type=["csv", "xlsx", "xls"], key="studio_upload")
        if uploaded is not None and st.sidebar.button("Load uploaded data", use_container_width=True, key="studio_load_upload"):
            try:
                frame = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
                set_scope_dataset("studio", frame, uploaded.name)
                st.sidebar.success("Loaded")
                st.rerun()
            except Exception as exc:
                st.sidebar.error(str(exc))
    frame, name = scope_dataset("studio")
    st.sidebar.caption(f"Current: {name}")
    st.sidebar.caption(f"{len(frame):,} rows · {frame.shape[1]} columns")

def project_sidebar():
    project = ensure_project_state()
    with st.sidebar.expander("📘 Assignment notebook", expanded=False):
        name = st.text_input("Student name", value=project.get("student_name", ""), key="project_student_name")
        project["student_name"] = name
        completed = len(project.get("weeks", {}))
        st.caption(f"Saved assignment records: {completed}")
        st.download_button(
            "Download readable notebook",
            complete_notebook_markdown(),
            "MATH490_complete_notebook.md",
            "text/markdown",
            use_container_width=True,
            key="project_markdown_download",
        )
        st.download_button(
            "Download continuation backup",
            project_json(),
            "MATH490_notebook_backup.json",
            "application/json",
            use_container_width=True,
            key="project_download",
        )
        upload = st.file_uploader("Continue from a notebook backup", type=["json"], key="project_upload")
        if upload is not None and st.button("Import notebook backup", use_container_width=True, key="project_import"):
            try:
                loaded = json.loads(upload.getvalue().decode("utf-8"))
                if not isinstance(loaded, dict):
                    raise ValueError("The notebook file is not valid.")
                st.session_state.course_project = loaded
                ensure_project_state()
                st.success("Notebook imported.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not import the notebook: {exc}")

def sidebar():
    st.sidebar.title("MATH 490 Lab Studio")
    st.sidebar.caption("Learn → Practise → Reflect → Apply")
    st.sidebar.divider()
    space = st.sidebar.radio(
        "Open",
        ["Today's Lab", "Practical Studio", "Wrap-Up", "My Notebook", "Learning Library", "Exam Practice", "Full Studio", "Instructor Setup"],
        key="v65_space",
        help="Today's Lab teaches. Practical Studio is the hands-on class. Wrap-Up ends class. My Notebook is the independent assignment. Learning Library explains terms and models. Exam Practice provides private revision.",
    )
    if space in {"Today's Lab", "Practical Studio", "Wrap-Up", "My Notebook", "Instructor Setup"}:
        week = st.sidebar.selectbox("Week", list(WEEKLY_LABS), key="v65_week")
        if space in {"Today's Lab", "Practical Studio", "Wrap-Up", "Instructor Setup"}:
            activate_dataset_scope("class")
            frame, name = scope_dataset("class")
            st.sidebar.caption(f"Class data: {name}")
            st.sidebar.caption(f"{len(frame):,} rows · {frame.shape[1]} columns")
        else:
            activate_dataset_scope("notebook")
            frame, name = scope_dataset("notebook")
            st.sidebar.caption(f"Assignment data: {name}")
            st.sidebar.caption(f"{len(frame):,} rows · {frame.shape[1]} columns")
            project_sidebar()
        return {"space": space, "week": week}
    if space in {"Learning Library", "Exam Practice"}:
        if space == "Learning Library":
            st.sidebar.caption("Search terms, models, parameters, and the course map.")
        else:
            project = ensure_project_state()
            completed = sum(int(item.get("attempts", 0)) for item in project.get("exam_practice", {}).values())
            st.sidebar.caption(f"Completed practice attempts: {completed}")
        return {"space": space}
    render_dataset_sidebar()
    section = st.sidebar.radio("Learning area", list(NAVIGATION), key="section")
    page = st.sidebar.radio("Activity", NAVIGATION[section], key="page_" + section)
    return {"space": space, "page": page}

def page_home():
    df=get_df(); st.title("🧠 MATH 490 Applied AI Lab Studio"); st.markdown("### Learn the mathematics, build the model, test it, explain it, and communicate the result.")
    st.markdown('<div class="path-card"><strong>Recommended learning path</strong><br>Question → Data → Probability → Association → Prediction → Predictor selection → Evaluation → Explanation → Communication</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3); c1.metric("Current observations",f"{len(df):,}"); c2.metric("Current variables",df.shape[1]); c3.metric("Missing cells",int(df.isna().sum().sum()))
    cols=st.columns(3); cards=[("1. Choose data","Load a course demo or upload your own table."),("2. Choose an activity","Follow the curriculum sequence or open the current week's method."),("3. Learn and prepare","Use the Learning Library, Exam Practice, and saved results to explain your work.")]
    for col,(title,text) in zip(cols,cards):
        with col: st.markdown(f'<div class="student-card"><strong>{title}</strong><br>{text}</div>',unsafe_allow_html=True)
    st.markdown("### What students can do")
    c1,c2=st.columns(2)
    with c1: st.markdown("- Explore and visualize data\n- Simulate probability and Bayes' theorem\n- Calculate Pearson, Spearman, Kendall, partial, eta, and Cramér's V\n- Fit regression and classification models")
    with c2: st.markdown("- Select predictors without test leakage\n- Use cross-validation and bootstrap\n- Search the glossary and study model parameters\n- Practise exam questions with targeted feedback\n- Backtest forecasts and build mini-reports")

def page_data():
    df=get_df(); st.title("Data and Research Questions")
    guide("Identify rows, variables, data quality, a target, and suitable predictors.",["Inspect a preview","Review types and missing values","Build a research question"],["Numerical versus categorical variables","Possible leakage","Whether the target suggests regression or classification"],["State the question","Name the target","Name at least three predictors"],"Do not use a predictor that directly contains the answer.")
    n=st.slider("Rows to preview",5,min(100,max(5,len(df))),min(12,max(5,len(df))),key="preview"); st.dataframe(df.head(n),use_container_width=True)
    c1,c2,c3,c4=st.columns(4); c1.metric("Rows",len(df)); c2.metric("Columns",df.shape[1]); c3.metric("Numerical",len(num_cols(df))); c4.metric("Categorical",len(cat_cols(df)))
    inventory=pd.DataFrame({"Variable":df.columns,"Type":["Numerical" if pd.api.types.is_numeric_dtype(df[c]) else "Categorical or text" for c in df],"Missing":df.isna().sum().values,"Unique":[df[c].nunique(dropna=True) for c in df]})
    st.dataframe(inventory,use_container_width=True); tabs=st.tabs(["Numerical summary","Categorical summary","Question builder"])
    with tabs[0]:
        nums = num_cols(df)
        if nums:
            numerical_summary = df[nums].describe().T.assign(
                median=df[nums].median(),
                skewness=df[nums].skew(),
            )
            st.dataframe(tidy_frame(numerical_summary, 3), use_container_width=True)
        else:
            st.info("No numerical variables.")
    with tabs[1]:
        rows = []
        for c in cat_cols(df):
            vc = df[c].value_counts(dropna=True)
            rows.append({
                "Variable": c,
                "Categories": df[c].nunique(dropna=True),
                "Most common": vc.index[0] if len(vc) else "NA",
                "Count": int(vc.iloc[0]) if len(vc) else 0,
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No categorical variables.")
    with tabs[2]:
        target=st.selectbox("Possible target",df.columns,key="question_target"); predictors=st.multiselect("Possible predictors",[c for c in df if c!=target],default=[c for c in df if c!=target][:3],key="question_pred")
        kind="regression" if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique()>20 else "classification"; st.info(f"This target most naturally suggests **{kind}**.")
        if predictors: st.success(f"How well can {', '.join(map(humanize,predictors))} explain or predict {humanize(target)}?")

def page_probability():
    st.title("Probability and Uncertainty")
    guide(
        "Use random experiments, probability rules, conditional probability, Bayes' theorem, expected value, and variance.",
        ["Run a coin or die experiment", "Change event probabilities", "Update a belief after one observation"],
        ["Observed proportions vary from theory", "Conditional probability changes the sample space", "Evidence can raise or lower a prior belief"],
        ["Name the sample space and event", "Apply one probability rule", "Explain the prior and updated belief"],
        "Probability describes uncertainty; it does not guarantee what will happen in one trial.",
    )
    before_you_run([
        ("Random experiment", "A process with an uncertain outcome, such as tossing a coin."),
        ("Sample space", "The complete set of possible outcomes."),
        ("Event", "One outcome or a group of outcomes that we care about."),
        ("Theoretical probability", "The probability expected from a mathematical model."),
        ("Experimental probability", "The observed proportion after repeated trials."),
        ("Conditional probability", "The probability of an event after restricting attention to cases where another event occurred."),
        ("Prior and posterior", "The prior is the belief before evidence; the posterior is the updated belief after evidence."),
        ("Expected value and variance", "Expected value is the long-run average; variance measures how widely outcomes vary around it."),
    ])

    t1, t2, t3, t4 = st.tabs([
        "Random experiments and events",
        "Probability rules",
        "Bayes: update a belief",
        "Random variables and uncertainty",
    ])

    with t1:
        st.info(
            "A random experiment has uncertain outcomes. The sample space contains every possible outcome, "
            "and an event is a selected group of outcomes."
        )
        experiment = st.radio("Experiment", ["Coin toss", "Six-sided die"], horizontal=True, key="prob_exp")
        trials = st.slider("Number of trials", 10, 10000, 500, 10, key="prob_trials")
        seed = st.number_input("Random seed", 0, 9999, 42, key="prob_seed")

        if experiment == "Coin toss":
            probability_heads = st.slider("Probability of heads", 0.0, 1.0, 0.5, 0.01, key="p_heads")
            st.markdown("**Sample space:** {Heads, Tails}  \n**Event:** Heads")
            if st.button("Run coin experiment", type="primary", key="run_coin"):
                rng = np.random.default_rng(seed)
                outcomes = rng.binomial(1, probability_heads, trials)
                running = np.cumsum(outcomes) / np.arange(1, trials + 1)
                c1, c2 = st.columns(2)
                c1.metric("Theoretical probability", f"{probability_heads:.3f}")
                c2.metric("Observed proportion", f"{outcomes.mean():.3f}")
                fig, ax = plt.subplots(figsize=(9, 4.5))
                ax.plot(np.arange(1, trials + 1), running, label="Observed running proportion")
                ax.axhline(probability_heads, linestyle="--", label="Theoretical probability")
                ax.set_xlabel("Number of trials")
                ax.set_ylabel("Proportion of heads")
                ax.legend(frameon=False)
                axis_style(ax)
                show_fig(fig, "coin_probability.png", "coin_dl")
                st.caption("The line may fluctuate at first, but it usually moves closer to the theoretical probability as the number of trials grows.")
        else:
            event_faces = st.multiselect("Event: selected die faces", [1, 2, 3, 4, 5, 6], default=[1, 2], key="die_event")
            st.markdown("**Sample space:** {1, 2, 3, 4, 5, 6}")
            st.markdown(f"**Event:** {{{', '.join(map(str, event_faces))}}}" if event_faces else "**Event:** empty event")
            theoretical = len(event_faces) / 6
            if st.button("Run die experiment", type="primary", key="run_die"):
                rng = np.random.default_rng(seed)
                outcomes = rng.integers(1, 7, trials)
                event = np.isin(outcomes, event_faces)
                running = np.cumsum(event) / np.arange(1, trials + 1)
                c1, c2 = st.columns(2)
                c1.metric("Theoretical probability", f"{theoretical:.3f}")
                c2.metric("Observed proportion", f"{event.mean():.3f}")
                fig, ax = plt.subplots(figsize=(9, 4.5))
                ax.plot(np.arange(1, trials + 1), running, label="Observed running proportion")
                ax.axhline(theoretical, linestyle="--", label="Theoretical probability")
                ax.set_xlabel("Number of trials")
                ax.set_ylabel("Proportion in the selected event")
                ax.legend(frameon=False)
                axis_style(ax)
                show_fig(fig, "die_probability.png", "die_dl")

    with t2:
        st.info(
            "Use the sliders to describe two events A and B. Their overlap must be mathematically possible. "
            "The app then applies complement, union, intersection, and conditional-probability rules."
        )
        c1, c2 = st.columns(2)
        with c1:
            p_a = st.slider("P(A)", 0.0, 1.0, 0.60, 0.01, key="prob_a")
        with c2:
            p_b = st.slider("P(B)", 0.0, 1.0, 0.50, 0.01, key="prob_b")
        lower = max(0.0, p_a + p_b - 1.0)
        upper = min(p_a, p_b)
        if np.isclose(lower, upper):
            p_intersection = float(lower)
            st.metric("P(A and B)", f"{p_intersection:.2f}")
            st.caption("The overlap is fixed by the selected values of P(A) and P(B).")
        else:
            current = float(st.session_state.get("prob_intersection", min(0.30, upper)))
            if current < lower or current > upper:
                st.session_state.prob_intersection = round((lower + upper) / 2, 2)
            p_intersection = st.slider(
                "P(A and B)",
                float(round(lower, 2)),
                float(round(upper, 2)),
                float(st.session_state.get("prob_intersection", round((lower + upper) / 2, 2))),
                0.01,
                key="prob_intersection",
            )
        p_union = p_a + p_b - p_intersection
        p_not_a = 1 - p_a
        p_a_given_b = p_intersection / p_b if p_b > 0 else np.nan
        p_b_given_a = p_intersection / p_a if p_a > 0 else np.nan
        rule_table = pd.DataFrame({
            "Quantity": ["P(not A)", "P(A or B)", "P(A given B)", "P(B given A)"],
            "Value": [p_not_a, p_union, p_a_given_b, p_b_given_a],
            "Rule": [
                "1 - P(A)",
                "P(A) + P(B) - P(A and B)",
                "P(A and B) / P(B)",
                "P(A and B) / P(A)",
            ],
        })
        st.dataframe(rule_table, use_container_width=True, hide_index=True)
        st.caption("Conditional probability restricts attention to the trials in the conditioning event.")

    with t3:
        st.info(
            "Bayes' theorem begins with a prior belief. After an observation is made, the belief is updated "
            "according to how likely that observation would be under competing explanations."
        )
        hypothesis = st.text_input("Hypothesis", "It will rain tomorrow", key="bayes_hypothesis")
        evidence = st.text_input("Observation", "Dark clouds are observed", key="bayes_evidence")
        c1, c2, c3 = st.columns(3)
        with c1:
            prior = st.slider("Prior belief P(H)", 0.01, 0.99, 0.30, 0.01, key="bayes_prior")
        with c2:
            likelihood_h = st.slider("P(observation | H)", 0.01, 0.99, 0.80, 0.01, key="bayes_like_h")
        with c3:
            likelihood_not_h = st.slider("P(observation | not H)", 0.01, 0.99, 0.25, 0.01, key="bayes_like_not_h")
        evidence_probability = likelihood_h * prior + likelihood_not_h * (1 - prior)
        posterior = likelihood_h * prior / evidence_probability
        c1, c2, c3 = st.columns(3)
        c1.metric("Belief before observation", f"{prior:.1%}")
        c2.metric("Belief after observation", f"{posterior:.1%}")
        c3.metric("Change in belief", f"{posterior - prior:+.1%}")
        st.latex(r"P(H\mid E)=\frac{P(E\mid H)P(H)}{P(E\mid H)P(H)+P(E\mid H^c)P(H^c)}")
        direction = "increased" if posterior > prior else "decreased" if posterior < prior else "did not change"
        st.success(
            f"After observing **{evidence}**, the belief that **{hypothesis}** {direction} "
            f"from {prior:.1%} to {posterior:.1%}."
        )
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.bar(["Before observation", "After observation"], [prior, posterior])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability of the hypothesis")
        axis_style(ax)
        show_fig(fig, "bayes_belief_update.png", "bayes_dl")
        st.caption("The chart compares the prior probability with the posterior probability after one observation.")

    with t4:
        st.info(
            "A discrete random variable assigns a numerical value to each possible outcome. Expected value is "
            "the probability-weighted average, while variance measures spread around that expected value."
        )
        values_text = st.text_input("Possible values, comma-separated", "0,1,2,3", key="rv_values")
        probabilities_text = st.text_input("Probabilities, comma-separated", "0.10,0.20,0.40,0.30", key="rv_probs")
        try:
            values = np.array([float(x.strip()) for x in values_text.split(",") if x.strip()])
            probabilities = np.array([float(x.strip()) for x in probabilities_text.split(",") if x.strip()])
            if len(values) == 0 or len(values) != len(probabilities):
                raise ValueError("Enter the same non-zero number of values and probabilities.")
            if np.any(probabilities < 0):
                raise ValueError("Probabilities cannot be negative.")
            if not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
                raise ValueError(f"The probabilities must add to 1. They currently add to {probabilities.sum():.4f}.")
            expected = float(np.sum(values * probabilities))
            variance = float(np.sum(((values - expected) ** 2) * probabilities))
            standard_deviation = math.sqrt(variance)
            c1, c2, c3 = st.columns(3)
            c1.metric("Expected value", f"{expected:.4f}")
            c2.metric("Variance", f"{variance:.4f}")
            c3.metric("Standard deviation", f"{standard_deviation:.4f}")
            st.dataframe(pd.DataFrame({"Value": values, "Probability": probabilities}), use_container_width=True, hide_index=True)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(values.astype(str), probabilities)
            ax.axvline(0, alpha=0)
            ax.set_xlabel("Possible value")
            ax.set_ylabel("Probability")
            axis_style(ax)
            show_fig(fig, "random_variable_distribution.png", "rv_dl")
            st.caption("The bar heights show the probability distribution. Greater spread around the expected value produces greater variance.")
        except ValueError as exc:
            st.warning(str(exc))

def _clear_invalid_widget_value(key, options):
    """Remove a stale selectbox value when the active dataset or option list changes."""
    current = st.session_state.get(key)
    if current is not None and current not in options:
        st.session_state.pop(key, None)


def page_visualization():
    df = get_df()
    nums = num_cols(df)
    cats = cat_cols(df)
    st.title("Visualization and Descriptive Statistics")
    guide(
        "Choose a plot that matches the variables.",
        ["Choose a chart type", "Select the required variables", "Select Show visualization"],
        ["Center and spread", "Outliers", "Group differences and nonlinear patterns"],
        ["Describe one pattern", "State one limitation"],
        "A pattern may be driven by outliers or a third variable.",
    )
    before_you_run([
        ("Histogram", "Shows the distribution of one numerical variable using value intervals called bins."),
        ("Boxplot", "Summarizes the median, middle 50%, spread, and potential outliers."),
        ("Scatter plot", "Shows how two numerical variables move together."),
        ("Bar chart", "Compares counts or a numerical summary such as a mean across categories."),
        ("Mean and median", "The mean is the arithmetic average; the median is the middle ordered value."),
        ("Outlier", "An unusually distant observation that may strongly affect a mean, correlation, or model."),
    ])

    chart_options = [
        "Histogram",
        "Boxplot",
        "Scatter plot",
        "Bar chart",
        "Correlation heatmap",
        "Missing-data chart",
    ]
    _clear_invalid_widget_value("chart", chart_options)
    chart = st.selectbox(
        "Chart type",
        chart_options,
        index=None,
        placeholder="Select a chart type",
        key="chart",
    )
    if not chart:
        st.info("Select a chart type. The app will not draw a figure until you choose the required settings and select **Show visualization**.")
        return

    st.caption("Choose the required settings, then select **Show visualization**. After changing a setting, select the button again to redraw the figure.")

    if chart == "Histogram":
        if not nums:
            st.warning("A numerical variable is required.")
            return
        _clear_invalid_widget_value("hist_x", nums)
        x = st.selectbox(
            "Numerical variable",
            nums,
            index=None,
            placeholder="Select a numerical variable",
            format_func=humanize,
            key="hist_x",
        )
        bins = st.slider("Bins", 5, 80, 25, key="bins")
        if st.button("Show visualization", type="primary", key="show_histogram"):
            if not x:
                st.warning("Select a numerical variable before drawing the histogram.")
                return
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df[x].dropna(), bins=bins, kde=True, ax=ax)
            ax.axvline(df[x].mean(), linestyle="--", label="Mean")
            ax.axvline(df[x].median(), linestyle=":", label="Median")
            ax.legend(frameon=False)
            ax.set_xlabel(humanize(x))
            ax.set_ylabel("Number of observations")
            axis_style(ax)
            show_fig(fig, "histogram.png", "hist_dl")

    elif chart == "Boxplot":
        if not nums:
            st.warning("A numerical variable is required.")
            return
        _clear_invalid_widget_value("box_y", nums)
        y = st.selectbox(
            "Numerical variable",
            nums,
            index=None,
            placeholder="Select a numerical variable",
            format_func=humanize,
            key="box_y",
        )
        group_options = ["None"] + cats
        _clear_invalid_widget_value("box_g", group_options)
        group = st.selectbox(
            "Optional grouping variable",
            group_options,
            index=0 if "box_g" not in st.session_state else None,
            format_func=lambda value: value if value == "None" else humanize(value),
            key="box_g",
        )
        if st.button("Show visualization", type="primary", key="show_boxplot"):
            if not y:
                st.warning("Select a numerical variable before drawing the boxplot.")
                return
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df, y=y, x=None if group == "None" else group, ax=ax)
            ax.set_ylabel(humanize(y))
            if group != "None":
                ax.set_xlabel(humanize(group))
            ax.tick_params(axis="x", rotation=30)
            axis_style(ax)
            show_fig(fig, "boxplot.png", "box_dl")

    elif chart == "Scatter plot":
        if len(nums) < 2:
            st.warning("Two numerical variables are required.")
            return
        _clear_invalid_widget_value("scatter_x", nums)
        x = st.selectbox(
            "Horizontal variable",
            nums,
            index=None,
            placeholder="Select the horizontal numerical variable",
            format_func=humanize,
            key="scatter_x",
        )
        y_options = [c for c in nums if c != x]
        _clear_invalid_widget_value("scatter_y", y_options)
        y = st.selectbox(
            "Vertical variable",
            y_options,
            index=None,
            placeholder="Select the vertical numerical variable",
            format_func=humanize,
            key="scatter_y",
        )
        group_options = ["None"] + cats
        _clear_invalid_widget_value("scatter_g", group_options)
        group = st.selectbox(
            "Optional color group",
            group_options,
            index=0 if "scatter_g" not in st.session_state else None,
            format_func=lambda value: value if value == "None" else humanize(value),
            key="scatter_g",
        )
        if st.button("Show visualization", type="primary", key="show_scatter"):
            if not x or not y:
                st.warning("Select both the horizontal and vertical numerical variables before drawing the scatter plot.")
                return
            fig, ax = plt.subplots(figsize=(8, 5))
            if group == "None":
                sns.regplot(data=df, x=x, y=y, scatter_kws={"alpha": .65}, ax=ax)
            else:
                sns.scatterplot(data=df, x=x, y=y, hue=group, alpha=.7, ax=ax)
            ax.set_xlabel(humanize(x))
            ax.set_ylabel(humanize(y))
            axis_style(ax)
            show_fig(fig, "scatter.png", "scatter_dl")

    elif chart == "Bar chart":
        if not cats:
            st.warning("A categorical variable is required.")
            return
        _clear_invalid_widget_value("bar_x", cats)
        x = st.selectbox(
            "Category",
            cats,
            index=None,
            placeholder="Select a categorical variable",
            format_func=humanize,
            key="bar_x",
        )
        mode_options = ["Count", "Mean", "Median"]
        _clear_invalid_widget_value("bar_mode", mode_options)
        mode = st.selectbox(
            "What should the bar height represent?",
            mode_options,
            index=None,
            placeholder="Select count, mean, or median",
            key="bar_mode",
        )
        y = None
        if mode in {"Mean", "Median"}:
            if not nums:
                st.warning("A numerical variable is required for mean or median bars.")
                return
            _clear_invalid_widget_value("bar_y", nums)
            y = st.selectbox(
                "Numerical variable summarized on the y-axis",
                nums,
                index=None,
                placeholder="Select a numerical variable",
                format_func=humanize,
                key="bar_y",
            )
        if st.button("Show visualization", type="primary", key="show_bar_chart"):
            if not x or not mode:
                st.warning("Select the category and what the bar height should represent.")
                return
            if mode in {"Mean", "Median"} and not y:
                st.warning("Select the numerical variable to summarize on the y-axis.")
                return
            if mode == "Count":
                plot = df[x].astype(str).value_counts().rename_axis(x).reset_index(name="bar_height")
                y_label = "Number of observations"
            else:
                plot = df.groupby(x, dropna=False)[y].agg(mode.lower()).reset_index(name="bar_height")
                y_label = f"{mode} {humanize(y).lower()}"
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=plot, x=x, y="bar_height", ax=ax)
            ax.set_xlabel(humanize(x))
            ax.set_ylabel(y_label)
            ax.tick_params(axis="x", rotation=30)
            axis_style(ax)
            show_fig(fig, "bar.png", "bar_dl")

    elif chart == "Correlation heatmap":
        selected = st.multiselect(
            "Numerical variables",
            nums,
            default=[],
            format_func=humanize,
            key="heat_vars",
            placeholder="Select at least two numerical variables",
        )
        method = st.selectbox("Correlation method", ["Pearson", "Spearman", "Kendall"], key="heat_method")
        if st.button("Show visualization", type="primary", key="show_heatmap"):
            if len(selected) < 2:
                st.warning("Select at least two numerical variables.")
                return
            corr = df[selected].corr(method=method.lower())
            fig, ax = plt.subplots(figsize=(9, 7))
            sns.heatmap(corr, annot=True, fmt=".2f", center=0, cmap="coolwarm", ax=ax)
            ax.set_xticklabels(map(humanize, corr.columns), rotation=35, ha="right")
            ax.set_yticklabels(map(humanize, corr.index), rotation=0)
            show_fig(fig, "heatmap.png", "heat_dl")

    else:
        st.write("This chart shows the number of missing values in every column.")
        if st.button("Show visualization", type="primary", key="show_missing_chart"):
            missing = df.isna().sum().sort_values()
            fig, ax = plt.subplots(figsize=(8, max(4, len(missing) * .32)))
            ax.barh(missing.index.map(humanize), missing.values)
            ax.set_xlabel("Number of missing values")
            ax.set_ylabel("Variable")
            axis_style(ax)
            show_fig(fig, "missing.png", "missing_dl")

def partial_corr(df,x,y,controls):
    sub=df[[x,y]+list(controls)].dropna(); C=np.column_stack([np.ones(len(sub)),sub[list(controls)].to_numpy()]); rx=sub[x]-C@np.linalg.lstsq(C,sub[x],rcond=None)[0]; ry=sub[y]-C@np.linalg.lstsq(C,sub[y],rcond=None)[0]; r,p=stats.pearsonr(rx,ry); return r,p,len(sub)
def eta(categories,values):
    z=pd.DataFrame({"c":categories,"v":values}).dropna(); grand=z.v.mean(); between=sum(len(g)*(g.v.mean()-grand)**2 for _,g in z.groupby("c")); total=((z.v-grand)**2).sum(); e2=between/total if total else 0; return math.sqrt(max(e2,0)),e2
def cramer(table):
    chi=stats.chi2_contingency(table)[0]; n=table.to_numpy().sum(); return math.sqrt((chi/n)/max(min(table.shape[0]-1,table.shape[1]-1),1))
def page_association():
    df=get_df(); st.title("Relationships and Association"); guide("Match association methods to variable types.",["Choose two variables","Add controls","Compare group-based and numeric relationships"],["Strength and direction","Adjusted versus unadjusted association","Group differences"],["Name the method","Interpret magnitude","Avoid causal claims"],"Eta and Cramér's V have no positive or negative sign.")
    before_you_run([
        ("Pearson correlation", "Measures the strength and direction of a linear relationship between two numerical variables."),
        ("Spearman and Kendall", "Rank-based measures that can detect monotonic relationships and are less dependent on linearity."),
        ("Partial correlation", "Measures the relationship between two numerical variables after controlling for selected variables."),
        ("Eta", "Measures the strength of association between a categorical grouping variable and a numerical variable; it has no sign."),
        ("Cramér's V", "Measures association between two categorical variables; values closer to 1 indicate stronger association."),
        ("P-value", "Assesses how surprising the observed association would be under a no-association assumption; it does not measure effect size or prove causation."),
    ])
    x=st.selectbox("First variable",df.columns,key="assoc_x"); y=st.selectbox("Second variable",[c for c in df if c!=x],key="assoc_y"); low=st.checkbox("Treat numerical variables with 10 or fewer values as categorical",True,key="lowcat")
    kind=lambda c:"numeric" if pd.api.types.is_numeric_dtype(df[c]) and not(low and df[c].nunique()<=10) else "categorical"; kx,ky=kind(x),kind(y); st.caption(f"Detected: {kx} versus {ky}")
    if kx==ky=="numeric":
        method=st.selectbox("Method",["Pearson","Spearman","Kendall"],key="assoc_method"); sub=df[[x,y]].dropna(); r,p=(stats.pearsonr(sub[x],sub[y]) if method=="Pearson" else stats.spearmanr(sub[x],sub[y]) if method=="Spearman" else stats.kendalltau(sub[x],sub[y])); c1,c2,c3=st.columns(3); c1.metric(method,f"{r:.3f}"); c2.metric("p-value",f"{p:.4f}"); c3.metric("Complete rows",len(sub))
        controls=st.multiselect("Numerical controls",[c for c in num_cols(df) if c not in [x,y]],key="controls")
        if controls:
            pr,pp,n=partial_corr(df,x,y,controls); st.write(f"**Partial Pearson correlation:** {pr:.3f} · **p-value:** {pp:.4f} · **rows:** {n}")
        fig,ax=plt.subplots(figsize=(8,5)); sns.regplot(data=sub,x=x,y=y,scatter_kws={"alpha":.65},ax=ax); axis_style(ax); show_fig(fig,"numeric_association.png","assoc_num_dl")
    elif {kx,ky}=={"numeric","categorical"}:
        cat=x if kx=="categorical" else y; val=y if ky=="numeric" else x; clean=df[[cat,val]].dropna(); e,e2=eta(clean[cat],clean[val]); c1,c2,c3=st.columns(3); c1.metric("Eta",f"{e:.3f}"); c2.metric("Eta squared",f"{e2:.3f}"); c3.metric("Rows",len(clean)); st.info(f"Approximately {e2:.1%} of observed variation is associated with group differences.")
        st.dataframe(tidy_frame(clean.groupby(cat)[val].agg(["count","mean","median","std"]).reset_index(),3),use_container_width=True,hide_index=True); groups=[g[val].to_numpy() for _,g in clean.groupby(cat) if len(g)>=2]
        if len(groups)>=2:
            test=st.selectbox("Group test",["One-way ANOVA","Kruskal-Wallis"],key="group_test"); stat,p=stats.f_oneway(*groups) if test.startswith("One") else stats.kruskal(*groups); st.write(f"**Statistic:** {stat:.3f} · **p-value:** {p:.4f}")
        fig,ax=plt.subplots(figsize=(8,5)); sns.boxplot(data=clean,x=cat,y=val,ax=ax); ax.tick_params(axis="x",rotation=30); axis_style(ax); show_fig(fig,"eta.png","eta_dl")
    else:
        table=pd.crosstab(df[x].astype(str),df[y].astype(str)); chi,p,dof,_=stats.chi2_contingency(table); v=cramer(table); c1,c2,c3=st.columns(3); c1.metric("Cramér's V",f"{v:.3f}"); c2.metric("p-value",f"{p:.4f}"); c3.metric("Degrees of freedom",dof); st.dataframe(table,use_container_width=True)
        prop=table.div(table.sum(axis=1),axis=0); fig,ax=plt.subplots(figsize=(9,5)); prop.plot(kind="bar",stacked=True,ax=ax); ax.legend(frameon=False,bbox_to_anchor=(1.02,1)); ax.tick_params(axis="x",rotation=30); axis_style(ax); show_fig(fig,"categorical_association.png","cramer_dl")

# -----------------------------------------------------------------------------
# Prediction pages
# -----------------------------------------------------------------------------
def regression_plots(result,prefix,qq=False):
    y=np.asarray(result["y_test"]); p=np.asarray(result["pred"]); residual=y-p; c1,c2=st.columns(2)
    with c1:
        fig,ax=plt.subplots(figsize=(6.5,5)); ax.scatter(y,p,alpha=.65); lo=min(y.min(),p.min()); hi=max(y.max(),p.max()); ax.plot([lo,hi],[lo,hi],linestyle="--"); ax.set_xlabel("Observed"); ax.set_ylabel("Predicted"); axis_style(ax); show_fig(fig,prefix+"_observed_predicted.png",prefix+"_op_dl")
    with c2:
        fig,ax=plt.subplots(figsize=(6.5,5)); ax.scatter(p,residual,alpha=.65); ax.axhline(0,linestyle="--"); ax.set_xlabel("Predicted"); ax.set_ylabel("Residual"); axis_style(ax); show_fig(fig,prefix+"_residuals.png",prefix+"_res_dl")
    if qq:
        fig,ax=plt.subplots(figsize=(7,5)); stats.probplot(residual,dist="norm",plot=ax); ax.set_title("Normal quantile plot of residuals"); axis_style(ax); show_fig(fig,prefix+"_qq.png",prefix+"_qq_dl")

def linear_coefficient_table(result):
    pipe = result["pipeline"]
    model = pipe.named_steps["model"]
    names = feature_names(pipe)
    coefficients = np.asarray(model.coef_).ravel()
    rows = [{
        "Term": "Intercept",
        "Estimate": float(np.asarray(model.intercept_).ravel()[0]),
        "Meaning": "Predicted target when numerical predictors are 0 and categorical predictors are at their reference levels.",
    }]
    for name, value in zip(names, coefficients):
        original = original_name(name, result["features"])
        if original in result["X_train"] and not pd.api.types.is_numeric_dtype(result["X_train"][original]):
            meaning = "Difference from the omitted reference category, holding other predictors fixed."
        else:
            meaning = "Expected target change for a one-unit predictor increase, holding other predictors fixed."
        rows.append({"Term": humanize(name), "Estimate": float(value), "Meaning": meaning})
    return tidy_frame(pd.DataFrame(rows), 3)


def logistic_coefficient_table(result):
    pipe = result["pipeline"]
    model = pipe.named_steps["model"]
    names = feature_names(pipe)
    coefficients = np.asarray(model.coef_).ravel()
    rows = []
    for name, value in zip(names, coefficients):
        rows.append({
            "Predictor": humanize(name),
            "Log-odds coefficient": float(value),
            "Odds ratio": float(np.exp(value)),
            "Direction": "increases odds" if value > 0 else "decreases odds" if value < 0 else "no change",
        })
    return tidy_frame(pd.DataFrame(rows), 3)


def page_linear_regression():
    df = get_df()
    nums = num_cols(df)
    st.title("Simple and Multiple Regression")
    guide(
        "Fit a straight-line model and interpret the intercept, slopes, prediction error, and residuals.",
        ["Choose simple or multiple regression", "Change predictors", "Change the held-out proportion"],
        ["Intercept and slope direction", "Training versus test error", "Residual patterns"],
        ["Interpret a slope", "Report model fit", "State one limitation"],
        "R-squared does not establish causation.",
    )
    before_you_run([
        ("Numerical target", "The continuous outcome the regression model predicts."),
        ("Predictor", "A variable used to explain or predict the target."),
        ("Intercept", "The predicted target when all numerical predictors equal zero and categorical predictors are at their reference levels."),
        ("Slope", "The expected change in the predicted target for a one-unit predictor increase, holding other predictors fixed."),
        ("Residual", "Observed target minus predicted target for one row."),
        ("Mean absolute error", "The average absolute prediction mistake. Lower is better."),
        ("Root mean squared error", "An error measure that gives extra weight to large mistakes. Lower is better."),
        ("R-squared", "The proportion of held-out target variation explained by the model. It does not establish causation."),
    ])
    if not nums:
        return st.warning("A numerical target is required.")
    target = st.selectbox("Continuous target", nums, key="lin_target")
    mode = st.radio("Regression type", ["Simple regression", "Multiple regression"], horizontal=True, key="lin_mode")
    if mode == "Simple regression":
        predictor_options = [c for c in nums if c != target]
        if not predictor_options:
            return st.warning("Simple linear regression requires another numerical variable.")
        features = [st.selectbox("Numerical predictor", predictor_options, key="lin_single")]
    else:
        options = [c for c in df if c != target]
        features = st.multiselect("Predictors", options, default=options[: min(5, len(options))], key="lin_features")
    test = st.slider("Test proportion", 0.1, 0.5, 0.25, 0.05, key="lin_test")
    seed = st.number_input("Random seed", 0, 9999, 42, key="lin_seed")
    if st.button("Fit linear regression", type="primary", key="lin_fit"):
        if not features:
            return st.warning("Select a predictor.")
        X, y = split_xy(df, target, features)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test, random_state=seed)
        pipe = build_pipe(X, features, "Linear Regression", "regression", {}, seed)
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        trainpred = pipe.predict(Xtr)
        save_result("linear_result", {
            "problem": "regression", "context": "linear regression", "model_name": "Linear Regression",
            "parameters": {}, "pipeline": pipe, "features": features, "target": target,
            "X_train": Xtr, "X_test": Xte, "y_train": ytr, "y_test": yte,
            "pred": pred, "train_pred": trainpred, "metrics": reg_metrics(yte, pred),
            "dataset_name": st.session_state.dataset_name,
        })
    result = st.session_state.get("linear_result")
    if (
        isinstance(result, dict)
        and result.get("target") == target
        and result.get("features") == features
    ):
        train = reg_metrics(result["y_train"], result["train_pred"])
        performance = pd.DataFrame({
            "Metric": result["metrics"].keys(),
            "Training": train.values(),
            "Test": result["metrics"].values(),
        })
        st.subheader("Model performance")
        st.dataframe(tidy_frame(performance, 3), use_container_width=True, hide_index=True)
        st.subheader("Intercept and slopes")
        coefficients = linear_coefficient_table(result)
        st.dataframe(coefficients, use_container_width=True, hide_index=True)
        if mode == "Simple regression" and len(coefficients) >= 2:
            intercept = coefficients.iloc[0]["Estimate"]
            slope = coefficients.iloc[1]["Estimate"]
            st.success(
                f"Estimated equation: {humanize(target)} = {intercept:.3f} + ({slope:.3f} × {humanize(features[0])})"
            )
        st.caption(
            "The intercept is the model's starting value. A slope is the expected change in the target for a one-unit "
            "predictor increase, with other predictors held fixed in multiple regression."
        )
        regression_plots(result, "linear", qq=True)
        numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(result["X_train"][c])]
        if STATSMODELS_AVAILABLE and len(numeric_features) >= 2:
            with st.expander("Optional multicollinearity check"):
                complete = result["X_train"][numeric_features].dropna()
                if len(complete) > len(numeric_features) + 1:
                    vif = pd.DataFrame({
                        "Predictor": [humanize(c) for c in numeric_features],
                        "Variance inflation factor": [
                            variance_inflation_factor(complete.values, i) for i in range(len(numeric_features))
                        ],
                    })
                    st.dataframe(tidy_frame(vif, 2), use_container_width=True, hide_index=True)
                    st.caption("Larger values indicate that predictors contain overlapping information.")

def page_ml_regression():
    df=get_df(); nums=num_cols(df); st.title("Machine Learning for Regression")
    guide("Compare a flexible model with simple baselines and explain its predictors.",["Choose a model","Adjust only its parameters","Compare with mean and linear baselines"],["Held-out error","Training-test gap","Importance and direction"],["State whether it beat the baselines","Identify an influential predictor","Discuss overfitting"],"Complexity must improve held-out performance to be useful.")
    before_you_run([
        ("Training data", "Rows used to fit the model."),
        ("Held-out test data", "Unseen rows used only to evaluate generalization."),
        ("Baseline", "A simple reference prediction, such as the training mean or linear regression."),
        ("Hyperparameter", "A model setting chosen before training, such as tree depth or regularization strength."),
        ("Tuning", "Comparing reasonable hyperparameter choices using validation rather than the final test set."),
        ("Regularization", "A penalty or restriction that discourages unnecessary model complexity."),
        ("Overfitting", "Excellent training performance but weaker performance on unseen data."),
        ("Mean absolute error and root mean squared error", "Both measure prediction mistakes and are better when lower; root mean squared error penalizes large mistakes more."),
    ])
    if not nums:return st.warning("A numerical target is required.")
    target=st.selectbox("Continuous target",nums,key="mlr_target"); options=[c for c in df if c!=target]; features=st.multiselect("Predictors",options,default=options[:min(6,len(options))],key="mlr_features")
    models=[m for m in REG_MODELS if m!="Linear Regression"]; name=st.selectbox("Model",models,index=models.index("Random Forest") if "Random Forest" in models else 0,key="mlr_model"); p=params_ui(name,"regression","mlr_",True); test=st.slider("Test proportion",.1,.5,.25,.05,key="mlr_test"); seed=st.number_input("Random seed",0,9999,42,key="mlr_seed")
    if st.button("Train regression model",type="primary",key="mlr_fit"):
        if not features:return st.warning("Select predictors.")
        X,y=split_xy(df,target,features); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test,random_state=seed); pipe=build_pipe(X,features,name,"regression",p,seed); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); trainpred=pipe.predict(Xtr)
        meanpred=np.repeat(ytr.mean(),len(yte)); linear=build_pipe(X,features,"Linear Regression","regression",{},seed); linear.fit(Xtr,ytr); linpred=linear.predict(Xte)
        save_result("mlr_result",{"problem":"regression","context":"machine learning regression","model_name":name,"parameters":p,"pipeline":pipe,"features":features,"target":target,"X_train":Xtr,"X_test":Xte,"y_train":ytr,"y_test":yte,"pred":pred,"train_pred":trainpred,"metrics":reg_metrics(yte,pred),"baselines":{"Mean baseline":reg_metrics(yte,meanpred),"Linear regression":reg_metrics(yte,linpred)},"dataset_name":st.session_state.dataset_name})
    result=st.session_state.get("mlr_result")
    if isinstance(result,dict) and result.get("target")==target and result.get("model_name")==name and result.get("features")==features:
        rows=[{"Model":name,**result["metrics"]}]+[{"Model":k,**v} for k,v in result["baselines"].items()]; st.dataframe(tidy_frame(pd.DataFrame(rows),3),use_container_width=True,hide_index=True)
        train_rmse=math.sqrt(mean_squared_error(result["y_train"],result["train_pred"])); st.metric("Training-test RMSE gap",f"{result['metrics']['Root mean squared error']-train_rmse:.3f}"); regression_plots(result,"mlr"); explanation_panel(result,"mlr_explain")

def classification_outputs(result,threshold,prefix):
    y=np.asarray(result["y_test"]); proba=result.get("proba"); pred=(proba[:,1]>=threshold).astype(int) if threshold is not None and proba is not None and result.get("n_classes")==2 else np.asarray(result["pred"])
    st.dataframe(metric_table(clf_metrics(y,pred,proba)),use_container_width=True,hide_index=True); labels=list(range(result["n_classes"])); names=result["class_names"]; c1,c2=st.columns(2)
    with c1:
        fig,ax=plt.subplots(figsize=(6.2,5)); sns.heatmap(confusion_matrix(y,pred,labels=labels),annot=True,fmt="d",xticklabels=names,yticklabels=names,ax=ax); ax.set_xlabel("Predicted"); ax.set_ylabel("Observed"); show_fig(fig,prefix+"_cm.png",prefix+"_cm_dl")
    with c2:
        fig,ax=plt.subplots(figsize=(6.2,5)); sns.heatmap(confusion_matrix(y,pred,labels=labels,normalize="true"),annot=True,fmt=".2f",xticklabels=names,yticklabels=names,ax=ax); ax.set_xlabel("Predicted"); ax.set_ylabel("Observed"); ax.set_title("Row-normalized confusion matrix"); show_fig(fig,prefix+"_cm_normalized.png",prefix+"_cmn_dl")
    if proba is not None and result["n_classes"]==2:
        score=proba[:,1]; fpr,tpr,_=roc_curve(y,score); precision,recall,_=precision_recall_curve(y,score); true,predprob=calibration_curve(y,score,n_bins=min(10,max(3,len(y)//15)),strategy="quantile"); tabs=st.tabs(["ROC","Precision-recall","Calibration","Probability distribution"])
        with tabs[0]:
            fig,ax=plt.subplots(figsize=(7,5)); ax.plot(fpr,tpr,label=f"AUC = {roc_auc_score(y,score):.3f}"); ax.plot([0,1],[0,1],linestyle="--"); ax.legend(frameon=False); ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate"); axis_style(ax); show_fig(fig,prefix+"_roc.png",prefix+"_roc_dl")
        with tabs[1]:
            fig,ax=plt.subplots(figsize=(7,5)); ax.plot(recall,precision,label=f"Average precision = {average_precision_score(y,score):.3f}"); ax.legend(frameon=False); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); axis_style(ax); show_fig(fig,prefix+"_pr.png",prefix+"_pr_dl")
        with tabs[2]:
            fig,ax=plt.subplots(figsize=(7,5)); ax.plot(predprob,true,marker="o"); ax.plot([0,1],[0,1],linestyle="--"); ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed event proportion"); axis_style(ax); show_fig(fig,prefix+"_calibration.png",prefix+"_cal_dl")
        with tabs[3]:
            fig,ax=plt.subplots(figsize=(8,5)); temp=pd.DataFrame({"p":score,"y":y})
            for label,g in temp.groupby("y"): ax.hist(g.p,bins=20,alpha=.55,label=names[int(label)])
            if threshold is not None: ax.axvline(threshold,linestyle="--",label="Threshold")
            ax.legend(frameon=False); ax.set_xlabel(f"Predicted probability of {names[1]}"); axis_style(ax); show_fig(fig,prefix+"_probabilities.png",prefix+"_prob_dl")
    with st.expander("Detailed class-by-class results"):
        report = pd.DataFrame(classification_report(y,pred,labels=labels,target_names=names,output_dict=True,zero_division=0)).T
        st.dataframe(tidy_frame(report,3),use_container_width=True)

def page_logistic():
    df=get_df(); targets=[c for c in class_targets(df) if df[c].dropna().nunique()==2]; st.title("Logistic Regression")
    guide("Model a binary outcome as a probability and examine thresholds and odds ratios.",["Choose a binary target","Change regularization","Move the threshold"],["False positives and negatives","Odds-ratio direction","Probability calibration"],["Interpret an odds ratio","Explain the threshold","State which error matters more"],"The 0.50 threshold is not automatically best.")
    before_you_run([
        ("Predicted probability", "The model's estimated probability that a row belongs to the positive class."),
        ("Decision threshold", "The probability cutoff used to convert a probability into a class prediction."),
        ("False positive", "The model predicts positive when the observed class is negative."),
        ("False negative", "The model predicts negative when the observed class is positive."),
        ("Odds ratio", "The multiplicative change in predicted odds for a one-unit predictor increase, holding other predictors fixed."),
        ("Precision", "Among predicted positives, the fraction that are truly positive."),
        ("Recall", "Among actual positives, the fraction correctly identified."),
        ("Calibration", "Whether predicted probabilities agree with observed frequencies."),
    ])
    if not targets:return st.warning("A binary target is required.")
    target=st.selectbox("Binary target",targets,key="log_target"); options=[c for c in df if c!=target]; features=st.multiselect("Predictors",options,default=options[:min(6,len(options))],key="log_features"); p=params_ui("Logistic Regression","classification","log_",True); test=st.slider("Test proportion",.1,.5,.25,.05,key="log_test"); seed=st.number_input("Random seed",0,9999,42,key="log_seed")
    if st.button("Fit logistic regression",type="primary",key="log_fit"):
        if not features:return st.warning("Select predictors.")
        X,yraw=split_xy(df,target,features); le=LabelEncoder(); y=pd.Series(le.fit_transform(yraw.astype(str)),index=yraw.index); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test,random_state=seed,stratify=y); pipe=build_pipe(X,features,"Logistic Regression","classification",p,seed); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); proba=pipe.predict_proba(Xte)
        save_result("log_result",{"problem":"classification","context":"logistic regression","model_name":"Logistic Regression","parameters":p,"pipeline":pipe,"features":features,"target":target,"X_train":Xtr,"X_test":Xte,"y_train":ytr,"y_test":yte,"pred":pred,"proba":proba,"n_classes":2,"class_names":le.classes_.tolist(),"metrics":clf_metrics(yte,pred,proba),"dataset_name":st.session_state.dataset_name})
    result=st.session_state.get("log_result")
    if isinstance(result,dict) and result.get("target")==target and result.get("features")==features:
        threshold=st.slider("Decision threshold",.05,.95,.5,.01,key="log_threshold")
        st.caption(f"Positive class: **{result['class_names'][1]}**")
        classification_outputs(result,threshold,"log")
        st.subheader("Logistic regression coefficients and odds ratios")
        st.dataframe(logistic_coefficient_table(result),use_container_width=True,hide_index=True)
        st.caption("An odds ratio above 1 increases the predicted odds of the positive class; below 1 decreases them.")
        explanation_panel(result,"log_explain")

def page_ml_classification():
    df=get_df(); targets=class_targets(df); st.title("Machine Learning for Classification")
    guide("Compare classifiers and explain errors and predictors beyond logistic regression.",["Choose a classifier","Adjust only its parameters","Inspect class balance"],["Accuracy versus balanced accuracy","Precision-recall trade-off","Importance and direction"],["Recommend a classifier","Identify an influential predictor","Explain a weakness"],"High accuracy can hide minority-class failure.")
    before_you_run([
        ("Confusion matrix", "A table comparing observed classes with predicted classes."),
        ("Class imbalance", "One class is much more common than another, which can make accuracy misleading."),
        ("Balanced accuracy", "The average recall across classes, giving each class equal importance."),
        ("Precision", "Among rows predicted as a class, the fraction that truly belong to it."),
        ("Recall", "Among rows truly in a class, the fraction the model finds."),
        ("F1-score", "The harmonic mean of precision and recall."),
        ("Class weighting", "Giving additional importance to underrepresented classes during training."),
        ("Hyperparameter", "A model setting, such as number of trees, support-vector penalty, or network size."),
    ])
    if not targets:return st.warning("A classification target is required.")
    target=st.selectbox("Target",targets,key="mlc_target"); options=[c for c in df if c!=target]; features=st.multiselect("Predictors",options,default=options[:min(6,len(options))],key="mlc_features"); models=[m for m in CLF_MODELS if m!="Logistic Regression"]; name=st.selectbox("Classifier",models,index=models.index("Random Forest") if "Random Forest" in models else 0,key="mlc_model"); p=params_ui(name,"classification","mlc_",True); test=st.slider("Test proportion",.1,.5,.25,.05,key="mlc_test"); seed=st.number_input("Random seed",0,9999,42,key="mlc_seed")
    if st.button("Train classifier",type="primary",key="mlc_fit"):
        if not features:return st.warning("Select predictors.")
        X,yraw=split_xy(df,target,features); le=LabelEncoder(); y=pd.Series(le.fit_transform(yraw.astype(str)),index=yraw.index); strat=y if y.value_counts().min()>=2 else None; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test,random_state=seed,stratify=strat); pipe=build_pipe(X,features,name,"classification",p,seed); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); proba=pipe.predict_proba(Xte) if hasattr(pipe,"predict_proba") else None; majority=np.repeat(ytr.mode().iloc[0],len(yte))
        save_result("mlc_result",{"problem":"classification","context":"machine learning classification","model_name":name,"parameters":p,"pipeline":pipe,"features":features,"target":target,"X_train":Xtr,"X_test":Xte,"y_train":ytr,"y_test":yte,"pred":pred,"proba":proba,"n_classes":y.nunique(),"class_names":le.classes_.tolist(),"metrics":clf_metrics(yte,pred,proba),"majority_accuracy":accuracy_score(yte,majority),"dataset_name":st.session_state.dataset_name})
    result=st.session_state.get("mlc_result")
    if isinstance(result,dict) and result.get("target")==target and result.get("model_name")==name and result.get("features")==features:
        st.metric("Majority-class baseline accuracy",f"{result['majority_accuracy']:.3f}"); threshold=st.slider("Decision threshold",.05,.95,.5,.01,key="mlc_threshold") if result["n_classes"]==2 and result.get("proba") is not None else None; classification_outputs(result,threshold,"mlc"); explanation_panel(result,"mlc_explain")

# -----------------------------------------------------------------------------
# Predictor selection and central explanation page
# -----------------------------------------------------------------------------
@dataclass
class MetricSpec:
    display:str; scoring:str; higher:bool; negate:bool
REG_METRICS={"Root mean squared error":MetricSpec("Root mean squared error","neg_root_mean_squared_error",False,True),"Mean absolute error":MetricSpec("Mean absolute error","neg_mean_absolute_error",False,True),"R-squared":MetricSpec("R-squared","r2",True,False)}
CLF_METRICS={"F1-score":MetricSpec("F1-score","f1_weighted",True,False),"Balanced accuracy":MetricSpec("Balanced accuracy","balanced_accuracy",True,False),"ROC area under the curve":MetricSpec("ROC area under the curve","roc_auc",True,False),"Average precision":MetricSpec("Average precision","average_precision",True,False)}
def cv_object(problem,y,folds,seed): return KFold(min(folds,len(y)),shuffle=True,random_state=seed) if problem=="regression" else StratifiedKFold(min(folds,int(y.value_counts().min())),shuffle=True,random_state=seed)
def subset_error(X, y, features, problem, name, parameters, cv, seed):
    """Return mean cross-validation error and its standard error for one predictor set."""
    pipe = build_pipe(X[list(features)], features, name, problem, parameters, seed)
    if problem == "regression":
        scores = cross_val_score(
            pipe,
            X[list(features)],
            y,
            cv=cv,
            scoring="neg_root_mean_squared_error",
        )
        errors = -scores
    else:
        scores = cross_val_score(
            pipe,
            X[list(features)],
            y,
            cv=cv,
            scoring="accuracy",
        )
        errors = 1 - scores
    mean_error = float(np.mean(errors))
    standard_error = float(np.std(errors, ddof=1) / math.sqrt(len(errors))) if len(errors) > 1 else 0.0
    return mean_error, standard_error


def page_predictor_selection():
    df = get_df()
    st.title("Predictor Selection")
    guide(
        "Build one predictor set step by step and select the set with the lowest cross-validation error.",
        ["Choose candidate predictors", "Choose an evaluation model", "Set the maximum number of predictors"],
        ["Which predictor is added at each step", "How validation error changes", "Whether the selected set performs well on untouched data"],
        ["List the selected predictors", "Report the lowest cross-validation error", "Report the untouched holdout error"],
        "The final holdout sample must not be used while choosing predictors.",
    )
    before_you_run([
        ("Forward selection", "Starts with no predictors and adds the predictor that most improves cross-validation performance at each step."),
        ("Cross-validation error", "Average validation error across repeated training and validation splits."),
        ("Candidate predictor", "A variable that is allowed to compete for entry into the model."),
        ("Stopping point", "The point where adding another predictor no longer improves validation performance."),
        ("Selection leakage", "Using final holdout outcomes to choose predictors, which makes final performance too optimistic."),
        ("Parsimony", "Preferring the simpler predictor set when extra predictors do not provide meaningful improvement."),
    ])
    st.info(
        "Method used: forward predictor selection. Starting with no predictors, the app tests every remaining "
        "candidate and adds the one that produces the lowest cross-validation error. It then selects the set "
        "size with the lowest error along the forward-selection path."
    )

    label = st.radio("Problem type", ["Regression", "Classification"], horizontal=True, key="sel_problem")
    problem = label.lower()
    targets = num_cols(df) if problem == "regression" else class_targets(df)
    models = (
        ["Linear Regression", "Ridge Regression", "Random Forest", "Gradient Boosting"]
        if problem == "regression"
        else ["Logistic Regression", "Random Forest", "Gradient Boosting", "Support Vector Machine"]
    )
    if not targets:
        return st.warning("No suitable target is available.")

    target = st.selectbox("Target", targets, key="sel_target")
    candidate_options = [c for c in df if c != target]
    candidates = st.multiselect(
        "Candidate predictors",
        candidate_options,
        default=candidate_options[: min(8, len(candidate_options))],
        key="sel_candidates",
    )
    if not candidates:
        return st.warning("Select candidate predictors.")
    if len(candidates) > 15:
        return st.warning("Use at most 15 candidate predictors for classroom speed.")

    name = st.selectbox("Evaluation model", models, key="sel_model")
    parameters = params_ui(name, problem, "sel_", False)
    error_name = "Root mean squared error" if problem == "regression" else "Classification error (1 - accuracy)"
    st.caption(f"Selection criterion: lowest cross-validated {error_name.lower()}.")

    c1, c2, c3 = st.columns(3)
    with c1:
        maximum_predictors = st.slider(
            "Maximum predictors",
            1,
            len(candidates),
            min(6, len(candidates)),
            key="sel_max",
        )
    with c2:
        folds = st.slider("Cross-validation folds", 3, 10, 5, key="sel_folds")
    with c3:
        holdout = st.slider("Untouched holdout proportion", 0.15, 0.40, 0.25, 0.05, key="sel_hold")
    seed = st.number_input("Random seed", 0, 9999, 42, key="sel_seed")

    if st.button("Select predictor set", type="primary", key="sel_run"):
        X, y_raw = split_xy(df, target, candidates)
        label_encoder = None
        if problem == "classification":
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y_raw.astype(str)), index=y_raw.index)
            if y.value_counts().min() < folds:
                return st.error("Reduce the number of folds because the smallest class has too few observations.")
            stratify = y
        else:
            y = pd.to_numeric(y_raw, errors="coerce")
            valid = y.notna()
            X, y = X.loc[valid], y.loc[valid]
            stratify = None

        X_development, X_holdout, y_development, y_holdout = train_test_split(
            X,
            y,
            test_size=holdout,
            random_state=seed,
            stratify=stratify,
        )
        if problem == "classification" and y_development.value_counts().min() < folds:
            return st.error("Reduce the number of folds because the development sample has too few observations in its smallest class.")
        if len(y_development) <= folds:
            return st.error("Reduce the number of folds or use a larger development sample.")
        cv = cv_object(problem, y_development, folds, seed)
        selected_path = []
        remaining = list(candidates)
        rows = []

        with st.spinner("Testing predictor sets by cross-validation..."):
            for step in range(1, maximum_predictors + 1):
                candidates_at_step = []
                for feature in remaining:
                    trial_features = selected_path + [feature]
                    mean_error, standard_error = subset_error(
                        X_development,
                        y_development,
                        trial_features,
                        problem,
                        name,
                        parameters,
                        cv,
                        seed,
                    )
                    candidates_at_step.append((mean_error, standard_error, feature))
                mean_error, standard_error, added_feature = min(candidates_at_step, key=lambda item: item[0])
                selected_path.append(added_feature)
                remaining.remove(added_feature)
                rows.append({
                    "Number of predictors": step,
                    "Predictor added": added_feature,
                    "Predictor set": selected_path.copy(),
                    "Mean cross-validation error": mean_error,
                    "Standard error": standard_error,
                })

        path_table = pd.DataFrame(rows)
        chosen_row = path_table.loc[path_table["Mean cross-validation error"].idxmin()]
        selected_features = list(chosen_row["Predictor set"])

        final_pipe = build_pipe(
            X_development[selected_features],
            selected_features,
            name,
            problem,
            parameters,
            seed,
        )
        final_pipe.fit(X_development[selected_features], y_development)
        predictions = final_pipe.predict(X_holdout[selected_features])
        probabilities = (
            final_pipe.predict_proba(X_holdout[selected_features])
            if problem == "classification" and hasattr(final_pipe, "predict_proba")
            else None
        )
        if problem == "regression":
            holdout_error = math.sqrt(mean_squared_error(y_holdout, predictions))
            metrics = reg_metrics(y_holdout, predictions)
        else:
            holdout_error = 1 - accuracy_score(y_holdout, predictions)
            metrics = clf_metrics(y_holdout, predictions, probabilities)

        result = {
            "selection_path": path_table,
            "chosen_features": selected_features,
            "lowest_cv_error": float(chosen_row["Mean cross-validation error"]),
            "holdout_error": float(holdout_error),
            "error_name": error_name,
            "problem": problem,
            "context": "predictor selection",
            "model_name": name,
            "parameters": parameters,
            "pipeline": final_pipe,
            "features": selected_features,
            "target": target,
            "X_train": X_development[selected_features],
            "X_test": X_holdout[selected_features],
            "y_train": y_development,
            "y_test": y_holdout,
            "pred": predictions,
            "proba": probabilities,
            "n_classes": y.nunique() if problem == "classification" else None,
            "class_names": label_encoder.classes_.tolist() if label_encoder is not None else None,
            "metrics": metrics,
            "dataset_name": st.session_state.dataset_name,
        }
        save_result("selection_result", result)

    result = st.session_state.get("selection_result")
    if isinstance(result, dict) and result.get("target") == target:
        path_table = result["selection_path"]
        display = path_table[["Number of predictors", "Predictor added", "Predictor set", "Mean cross-validation error"]].copy()
        display["Predictor added"] = display["Predictor added"].map(humanize)
        display["Predictor set"] = display["Predictor set"].apply(lambda values: ", ".join(map(humanize, values)))
        display = display.rename(columns={"Mean cross-validation error": result["error_name"]})
        st.dataframe(tidy_frame(display, 3), use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Selected predictors", len(result["chosen_features"]))
        c2.metric("Lowest cross-validation error", f"{result['lowest_cv_error']:.3f}")
        c3.metric("Untouched holdout error", f"{result['holdout_error']:.3f}")
        st.success("Selected set: " + ", ".join(map(humanize, result["chosen_features"])))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            path_table["Number of predictors"],
            path_table["Mean cross-validation error"],
            marker="o",
        )
        ax.axvline(len(result["chosen_features"]), linestyle="--", label="Selected set")
        ax.set_xlabel("Number of predictors")
        ax.set_ylabel(result["error_name"])
        ax.legend(frameon=False)
        axis_style(ax)
        show_fig(fig, "predictor_selection_error.png", "sel_dl")
        st.caption(
            "Each point is the mean cross-validation error for that predictor-set size. The dashed line marks "
            "the predictor set with the lowest mean error."
        )
        explanation_panel(result, "selection_explain")

def page_explanations():
    st.title("Model Explanations")
    guide(
        "Use two model-agnostic methods to identify which predictors matter to the fitted model.",
        ["Train a model first", "Calculate permutation importance", "Calculate a SHAP summary"],
        ["Importance magnitude", "Whether contributions usually raise or lower predictions", "Correlated predictors may share importance"],
        ["Identify the most influential predictor", "Explain its direction", "State that explanation is not causation"],
        "Explanations describe the fitted model, not a causal relationship.",
    )
    before_you_run([
        ("Global explanation", "Describes how the fitted model behaves across many observations rather than explaining only one row."),
        ("Permutation importance", "Measures how much held-out performance worsens when one predictor is shuffled."),
        ("SHAP value", "A model-based contribution showing how a predictor moves a prediction away from a reference prediction."),
        ("Importance magnitude", "Large magnitude means the model relies more strongly on that predictor; it does not prove causation."),
        ("Direction", "Positive and negative SHAP values indicate whether a predictor pushed predictions upward or downward for particular rows."),
        ("Correlated predictors", "Predictors carrying similar information may divide or exchange importance."),
    ])
    result = st.session_state.get("latest_model_result")
    if not isinstance(result, dict):
        return st.info("Train a regression, classification, selection, comparison, or forecasting model first.")
    st.markdown(
        f"**Latest model:** {result.get('model_name')}  \n**Target:** {humanize(result.get('target'))}  \n**Dataset:** {result.get('dataset_name')}"
    )
    st.dataframe(metric_table(result.get("metrics", {})), use_container_width=True, hide_index=True)
    explanation_panel(result, "central")

# -----------------------------------------------------------------------------
# Reliability pages
# -----------------------------------------------------------------------------
def page_comparison():
    df=get_df(); st.title("Model Evaluation and Comparison")
    guide("Compare models fairly using the same target, predictors, split, and metric.",["Choose several models","Set each model's own parameters","Add cross-validation"],["Held-out performance","Training-test gap","Prediction versus interpretability"],["Recommend a prediction model","Recommend an explanation model","Justify the metric"],"Models must be evaluated on the same held-out observations.")
    before_you_run([
        ("Fair comparison", "Every model must use the same target, predictors, preprocessing, training rows, and held-out rows."),
        ("Regression metric direction", "Mean absolute error and root mean squared error are better when lower; R-squared is generally better when higher."),
        ("Classification metric direction", "Balanced accuracy, F1-score, ROC area under the curve, and average precision are generally better when higher."),
        ("Training-test gap", "A large training advantage that disappears on test data suggests overfitting."),
        ("Cross-validation mean", "The average validation performance across folds."),
        ("Cross-validation variation", "How much performance changes across folds; lower variation indicates more stable performance."),
        ("Practical improvement", "A numerically better score may still be too small to justify a much more complex model."),
        ("Interpretability", "How easily a human can understand and communicate how the model reaches predictions."),
    ])
    label=st.radio("Problem type",["Regression","Classification"],horizontal=True,key="cmp_problem"); problem=label.lower(); targets=num_cols(df) if problem=="regression" else class_targets(df); metrics=REG_METRICS if problem=="regression" else CLF_METRICS; available=REG_MODELS if problem=="regression" else CLF_MODELS
    if not targets:return st.warning("No suitable target.")
    target=st.selectbox("Target",targets,key="cmp_target"); options=[c for c in df if c!=target]; features=st.multiselect("Predictors",options,default=options[:min(6,len(options))],key="cmp_features")
    defaults=["Linear Regression","Ridge Regression","Random Forest","Gradient Boosting"] if problem=="regression" else ["Logistic Regression","Linear Discriminant Analysis","Random Forest","Support Vector Machine"]
    selected=st.multiselect("Models",available,default=[m for m in defaults if m in available],key="cmp_models")
    if not features or not selected:return st.warning("Select predictors and models.")
    if len(selected)>7:return st.warning("Use no more than seven models.")
    parameters={}
    for i,name in enumerate(selected): parameters[name]=params_ui(name,problem,f"cmp{i}_",False)
    metric_name=st.selectbox("Primary metric",list(metrics),key="cmp_metric"); metric=metrics[metric_name]; c1,c2,c3=st.columns(3)
    with c1:test=st.slider("Test proportion",.1,.5,.25,.05,key="cmp_test")
    with c2:seed=st.number_input("Random seed",0,9999,42,key="cmp_seed")
    with c3:use_cv=st.checkbox("Also calculate cross-validation",True,key="cmp_cv")
    folds=st.slider("Cross-validation folds",3,10,5,key="cmp_folds") if use_cv else 5
    if st.button("Run fair comparison",type="primary",key="cmp_run"):
        X,yraw=split_xy(df,target,features)
        if problem=="classification":
            le=LabelEncoder(); y=pd.Series(le.fit_transform(yraw.astype(str)),index=yraw.index)
            if metric.scoring in ["roc_auc","average_precision"] and y.nunique()!=2:return st.error("This metric requires a binary target.")
            strat=y if y.value_counts().min()>=2 else None
        else:y=pd.to_numeric(yraw,errors="coerce"); valid=y.notna(); X,y=X.loc[valid],y.loc[valid]; strat=None
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test,random_state=seed,stratify=strat); cv=cv_object(problem,ytr,folds,seed) if use_cv else None; rows=[]; fitted={}
        for name in selected:
            try:
                pipe=build_pipe(X,features,name,problem,parameters[name],seed); pipe.fit(Xtr,ytr); trainpred=pipe.predict(Xtr); pred=pipe.predict(Xte); trainproba=pipe.predict_proba(Xtr) if problem=="classification" and hasattr(pipe,"predict_proba") else None; proba=pipe.predict_proba(Xte) if problem=="classification" and hasattr(pipe,"predict_proba") else None
                trainm=reg_metrics(ytr,trainpred) if problem=="regression" else clf_metrics(ytr,trainpred,trainproba); testm=reg_metrics(yte,pred) if problem=="regression" else clf_metrics(yte,pred,proba); tr=trainm[metric.display]; te=testm[metric.display]; gap=te-tr if not metric.higher else tr-te; row={"Model":name,"Training performance":tr,"Test performance":te,"Overfitting gap":gap}
                if cv:
                    scores=cross_val_score(pipe,Xtr,ytr,cv=cv,scoring=metric.scoring); shown=-scores if metric.negate else scores; row.update({"Cross-validation mean":shown.mean(),"Cross-validation standard deviation":shown.std(ddof=1)})
                rows.append(row); fitted[name]={"pipe":pipe,"pred":pred,"proba":proba,"trainpred":trainpred,"metrics":testm}
            except Exception as exc: rows.append({"Model":name,"Error":str(exc)})
        table=pd.DataFrame(rows); valid=table.dropna(subset=["Test performance"])
        if valid.empty:st.session_state.comparison={"table":table}; return
        best=valid.sort_values("Test performance",ascending=not metric.higher).iloc[0].Model; f=fitted[best]
        latest={"problem":problem,"context":"model comparison","model_name":best,"parameters":parameters[best],"pipeline":f["pipe"],"features":features,"target":target,"X_train":Xtr,"X_test":Xte,"y_train":ytr,"y_test":yte,"pred":f["pred"],"proba":f["proba"],"train_pred":f["trainpred"],"n_classes":y.nunique() if problem=="classification" else None,"class_names":le.classes_.tolist() if problem=="classification" else None,"metrics":f["metrics"],"dataset_name":st.session_state.dataset_name}
        st.session_state.comparison={"table":table,"best":best,"metric":metric.display}; st.session_state.latest_model_result=latest
    result=st.session_state.get("comparison")
    if isinstance(result,dict):
        st.dataframe(tidy_frame(result["table"],3),use_container_width=True,hide_index=True)
        if result.get("best"):
            st.success(f"Best held-out model by {result['metric']}: {result['best']}"); table=result["table"].dropna(subset=["Test performance"]).sort_values("Test performance"); fig,ax=plt.subplots(figsize=(8,max(4,len(table)*.42))); ax.barh(table.Model,table["Test performance"]); ax.set_xlabel(result["metric"]); axis_style(ax); show_fig(fig,"model_comparison.png","cmp_dl")

def page_cv():
    df = get_df()
    st.title("Cross-Validation and Model Selection")
    guide(
        "Compare a single split, k-fold, repeated k-fold, and leave-one-out validation.",
        ["Change strategy", "Change folds", "Compare score variability"],
        ["Mean performance", "Standard deviation and error", "Runtime"],
        ["Explain why one split can be unstable", "Report mean and standard deviation", "Recommend a strategy"],
        "Do not use shuffled validation for time series.",
    )
    before_you_run([
        ("Validation split", "A subset used to estimate performance while model settings are being chosen."),
        ("Fold", "One partition of the data used as validation while the remaining partitions train the model."),
        ("K-fold cross-validation", "Repeats training and validation so each fold serves as validation once."),
        ("Repeated k-fold", "Runs k-fold cross-validation several times with different fold assignments."),
        ("Leave-one-out", "Uses one observation for validation at a time; thorough but often computationally expensive."),
        ("Mean score", "The average performance across validation resamples."),
        ("Cross-validation standard deviation", "How much scores change across resamples; lower variation indicates more stable performance."),
        ("Time-series validation", "Uses earlier rows to predict later rows and never randomly shuffles time."),
    ])
    label = st.radio("Problem type", ["Regression", "Classification"], horizontal=True, key="cv_problem")
    problem = label.lower()
    targets = num_cols(df) if problem == "regression" else class_targets(df)
    metrics = REG_METRICS if problem == "regression" else CLF_METRICS
    models = REG_MODELS if problem == "regression" else CLF_MODELS
    if not targets:
        return st.warning("No suitable target.")
    target = st.selectbox("Target", targets, key="cv_target")
    options = [c for c in df if c != target]
    features = st.multiselect("Predictors", options, default=options[:min(5, len(options))], key="cv_features")
    name = st.selectbox("Model", models, key="cv_model")
    parameters = params_ui(name, problem, "cv_", False)
    metric_name = st.selectbox("Metric", list(metrics), key="cv_metric")
    metric = metrics[metric_name]
    strategy = st.selectbox(
        "Strategy",
        ["Single validation split", "K-fold cross-validation", "Repeated k-fold cross-validation", "Leave-one-out cross-validation"],
        key="cv_strategy",
    )
    seed = st.number_input("Random seed", 0, 9999, 42, key="cv_seed")
    folds = st.slider("Folds", 3, 10, 5, key="cv_folds") if "fold" in strategy.lower() and "leave" not in strategy.lower() else 5
    repeats = st.slider("Repetitions", 2, 10, 3, key="cv_repeats") if strategy.startswith("Repeated") else 1

    if st.button("Run resampling experiment", type="primary", key="cv_run"):
        if not features:
            return st.warning("Select predictors.")
        X, yraw = split_xy(df, target, features)
        class_names = None
        if problem == "classification":
            encoder = LabelEncoder()
            y = pd.Series(encoder.fit_transform(yraw.astype(str)), index=yraw.index)
            class_names = encoder.classes_.tolist()
            if metric.scoring in ["roc_auc", "average_precision"] and y.nunique() != 2:
                return st.error("This metric requires a binary target.")
        else:
            y = pd.to_numeric(yraw, errors="coerce")
            valid = y.notna()
            X, y = X.loc[valid], y.loc[valid]
        pipe = build_pipe(X, features, name, problem, parameters, seed)
        if strategy == "Single validation split":
            stratify = y if problem == "classification" and y.value_counts().min() >= 2 else None
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=seed, stratify=stratify)
            pipe.fit(Xtr, ytr)
            prediction = pipe.predict(Xte)
            probability = pipe.predict_proba(Xte) if problem == "classification" and hasattr(pipe, "predict_proba") else None
            scores = np.array([(reg_metrics(yte, prediction) if problem == "regression" else clf_metrics(yte, prediction, probability))[metric.display]])
        else:
            if strategy.startswith("Leave"):
                if len(X) > 300:
                    return st.error("Leave-one-out is limited to 300 rows in this classroom app.")
                if problem == "classification" and metric.scoring in ["roc_auc", "average_precision"]:
                    return st.error("ROC-AUC and average precision cannot be computed one observation at a time. Choose accuracy, balanced accuracy, precision, recall, or F1-score for leave-one-out validation.")
                cv = LeaveOneOut()
            elif strategy.startswith("Repeated"):
                if problem == "regression":
                    cv = RepeatedKFold(n_splits=min(folds, len(X)), n_repeats=repeats, random_state=seed)
                else:
                    if y.value_counts().min() < folds:
                        return st.error("Reduce folds.")
                    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
            else:
                if problem == "regression":
                    cv = KFold(min(folds, len(X)), shuffle=True, random_state=seed)
                else:
                    if y.value_counts().min() < folds:
                        return st.error("Reduce folds.")
                    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
            raw = cross_val_score(pipe, X, y, cv=cv, scoring=metric.scoring)
            scores = -raw if metric.negate else raw
        st.session_state.cv_result = {
            "result_type": "cross_validation",
            "scores": np.asarray(scores),
            "metric": metric.display,
            "strategy": strategy,
            "higher_is_better": bool(metric.higher),
            "problem": problem,
            "model_name": name,
            "target": target,
            "features": list(features),
            "folds": folds,
            "repeats": repeats,
            "class_names": class_names,
            "dataset_name": st.session_state.dataset_name,
        }

    result = st.session_state.get("cv_result")
    if isinstance(result, dict):
        scores = np.asarray(result["scores"], dtype=float)
        mean = float(scores.mean())
        standard_deviation = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean", f"{mean:.3f}")
        c2.metric("Standard deviation", f"{standard_deviation:.3f}")
        c3.metric("Resamples", len(scores))
        table = pd.DataFrame({"Resample": np.arange(1, len(scores) + 1), result["metric"]: scores})
        st.dataframe(tidy_frame(table, 3), use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.scatter(table.Resample, scores)
        ax.axhline(mean, linestyle="--")
        ax.set_xlabel("Resample")
        ax.set_ylabel(result["metric"])
        axis_style(ax)
        show_fig(fig, "cross_validation.png", "cv_dl")
        st.markdown("### What the cross-validation result means")
        for note in simple_result_notes(result):
            st.markdown("- " + note)

def _bootstrap_model_is_ready(result, df):
    required = {"pipeline", "problem", "features", "target", "X_train", "X_test", "y_train", "y_test", "pred"}
    if not isinstance(result, dict) or not required.issubset(result):
        return False
    if result.get("dataset_name") != st.session_state.get("dataset_name"):
        return False
    if result.get("target") not in df.columns:
        return False
    if any(feature not in df.columns for feature in result.get("features", [])):
        return False
    return True


def _bootstrap_model_preparation(df, seed):
    """Provide a self-contained model-training step for model-based bootstrap work."""
    current = st.session_state.get("latest_model_result")
    ready = _bootstrap_model_is_ready(current, df)
    if ready:
        st.success(
            f"Saved model ready: **{current.get('model_name', 'Model')}** predicting "
            f"**{humanize(current.get('target', 'target'))}** with {len(current.get('features', []))} predictor(s)."
        )
        train_new = st.checkbox("Train or replace the saved model", value=False, key="boot_replace_model")
    else:
        st.warning("No compatible model is available for this dataset. Train one below before bootstrapping a model metric or individual prediction.")
        train_new = True

    if train_new:
        with st.expander("1. Train a model for the bootstrap activity", expanded=True):
            st.caption("This model is used only to create the held-out metric or prediction that the bootstrap will resample. Recommended settings are used to keep Week 13 focused on uncertainty.")
            problem_label = st.radio("Model problem type", ["Regression", "Classification"], horizontal=True, key="boot_train_problem")
            problem = problem_label.lower()
            targets = num_cols(df) if problem == "regression" else class_targets(df)
            if not targets:
                st.warning("The current dataset has no suitable target for this model type.")
                return current if ready else None
            _clear_invalid_widget_value("boot_train_target", targets)
            target = st.selectbox("Model target", targets, format_func=humanize, key="boot_train_target")
            predictor_options = [c for c in df.columns if c != target]
            saved_features = [c for c in st.session_state.get("boot_train_features", []) if c in predictor_options]
            if "boot_train_features" not in st.session_state:
                st.session_state["boot_train_features"] = predictor_options[:min(5, len(predictor_options))]
            elif saved_features != st.session_state.get("boot_train_features"):
                st.session_state["boot_train_features"] = saved_features
            features = st.multiselect("Model predictors", predictor_options, format_func=humanize, key="boot_train_features")
            model_choices = (
                ["Linear Regression", "Ridge Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
                if problem == "regression"
                else ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "Support Vector Machine"]
            )
            _clear_invalid_widget_value("boot_train_model", model_choices)
            model_name = st.selectbox("Model", model_choices, key="boot_train_model")
            test_proportion = st.slider("Held-out test proportion", 0.15, 0.40, 0.25, 0.05, key="boot_train_test")
            if st.button("Train model for bootstrap", type="primary", key="boot_train_button"):
                if not features:
                    st.warning("Select at least one predictor.")
                else:
                    X, yraw = split_xy(df, target, features)
                    class_names = None
                    if problem == "classification":
                        encoder = LabelEncoder()
                        y = pd.Series(encoder.fit_transform(yraw.astype(str)), index=yraw.index)
                        class_names = encoder.classes_.tolist()
                        stratify = y if y.value_counts().min() >= 2 else None
                    else:
                        y = pd.to_numeric(yraw, errors="coerce")
                        valid = y.notna()
                        X, y = X.loc[valid], y.loc[valid]
                        stratify = None
                    if len(X) < 12:
                        st.error("At least 12 complete observations are required to train and evaluate the model.")
                    else:
                        Xtr, Xte, ytr, yte = train_test_split(
                            X, y, test_size=test_proportion, random_state=int(seed), stratify=stratify
                        )
                        pipe = build_pipe(X, features, model_name, problem, {}, int(seed))
                        pipe.fit(Xtr, ytr)
                        pred = pipe.predict(Xte)
                        proba = pipe.predict_proba(Xte) if problem == "classification" and hasattr(pipe, "predict_proba") else None
                        metrics = reg_metrics(yte, pred) if problem == "regression" else clf_metrics(yte, pred, proba)
                        latest = {
                            "problem": problem,
                            "context": "bootstrap model preparation",
                            "model_name": model_name,
                            "display_model_name": model_name,
                            "parameters": {},
                            "pipeline": pipe,
                            "features": list(features),
                            "target": target,
                            "X_train": Xtr,
                            "X_test": Xte,
                            "y_train": ytr,
                            "y_test": yte,
                            "pred": pred,
                            "proba": proba,
                            "n_classes": int(y.nunique()) if problem == "classification" else None,
                            "class_names": class_names,
                            "metrics": metrics,
                            "dataset_name": st.session_state.dataset_name,
                        }
                        st.session_state.latest_model_result = latest
                        st.session_state.pop("bootstrap", None)
                        st.success("Model trained. The model-based bootstrap controls are now ready.")
                        st.rerun()
    return st.session_state.get("latest_model_result") if _bootstrap_model_is_ready(st.session_state.get("latest_model_result"), df) else None


def page_bootstrap():
    df = get_df()
    st.title("Bootstrap and Uncertainty")
    guide(
        "Estimate uncertainty for a statistic, slope, model metric, or individual prediction.",
        ["Choose a quantity", "Train a model when the quantity is model-based", "Change bootstrap samples", "Inspect the distribution"],
        ["Bias", "Standard error", "Confidence interval width"],
        ["Report estimate and interval", "Explain bias and standard error", "Interpret the interval", "State a limitation"],
        "A confidence interval is not a range containing 95% of observations.",
    )
    before_you_run([
        ("Bootstrap resample", "A sample drawn with replacement from the observed rows, usually with the same number of rows as the original sample."),
        ("Original estimate", "The statistic, slope, metric, or prediction calculated before resampling."),
        ("Bootstrap distribution", "The collection of estimates produced across many bootstrap resamples."),
        ("Estimated bias", "Average bootstrap estimate minus the original estimate."),
        ("Bootstrap standard error", "The standard deviation of bootstrap estimates; it measures sampling uncertainty."),
        ("Percentile 95% interval", "The interval between the 2.5th and 97.5th percentiles of bootstrap estimates."),
        ("Interval width", "Upper endpoint minus lower endpoint; wider intervals indicate greater uncertainty."),
        ("Important interpretation", "The interval describes uncertainty in the estimated quantity, not where 95% of individual observations must fall."),
    ])
    mode = st.selectbox(
        "Quantity",
        ["Descriptive statistic", "Simple-regression slope", "Saved model metric", "Saved individual prediction"],
        key="boot_mode",
    )
    large = mode in ["Descriptive statistic", "Simple-regression slope"]
    B = st.slider(
        "Bootstrap samples",
        100 if large else 20,
        5000 if large else 500,
        1000 if large else 200,
        100 if large else 20,
        key="boot_B",
    )
    seed = st.number_input("Random seed", 0, 9999, 42, key="boot_seed")

    if mode == "Descriptive statistic":
        nums = num_cols(df)
        if not nums:
            return st.warning("A numerical variable is required.")
        var = st.selectbox("Variable", nums, format_func=humanize, key="boot_var")
        statname = st.selectbox("Statistic", ["Mean", "Median", "Standard deviation"], key="boot_stat")
        data = df[var].dropna().to_numpy()
        if st.button("Run bootstrap", type="primary", key="boot_run1"):
            rng = np.random.default_rng(seed)
            fn = np.mean if statname == "Mean" else np.median if statname == "Median" else lambda values: np.std(values, ddof=1)
            original = float(fn(data))
            values = np.array([fn(rng.choice(data, len(data), replace=True)) for _ in range(B)])
            st.session_state.bootstrap = {
                "result_type": "bootstrap",
                "values": values,
                "original": original,
                "label": f"{statname} of {humanize(var)}",
                "quantity_kind": "descriptive_statistic",
                "variable": var,
                "bootstrap_samples": int(B),
            }
    elif mode == "Simple-regression slope":
        nums = num_cols(df)
        if len(nums) < 2:
            return st.warning("Two numerical variables are required.")
        x = st.selectbox("Predictor", nums, format_func=humanize, key="boot_x")
        y = st.selectbox("Outcome", [c for c in nums if c != x], format_func=humanize, key="boot_y")
        clean = df[[x, y]].dropna()
        if st.button("Bootstrap slope", type="primary", key="boot_run2"):
            rng = np.random.default_rng(seed)
            original = float(np.polyfit(clean[x], clean[y], 1)[0])
            values = []
            for _ in range(B):
                sample = clean.iloc[rng.integers(0, len(clean), len(clean))]
                values.append(np.polyfit(sample[x], sample[y], 1)[0])
            st.session_state.bootstrap = {
                "result_type": "bootstrap",
                "values": np.array(values),
                "original": original,
                "label": f"Slope for {humanize(x)} predicting {humanize(y)}",
                "quantity_kind": "slope",
                "predictor": x,
                "outcome": y,
                "bootstrap_samples": int(B),
            }
    elif mode == "Saved model metric":
        model_result = _bootstrap_model_preparation(df, seed)
        if model_result is not None:
            choices = ["Mean absolute error", "Root mean squared error", "R-squared"] if model_result["problem"] == "regression" else ["Accuracy", "Balanced accuracy", "F1-score"]
            metric = st.selectbox("Metric to bootstrap", choices, key="boot_metric")
            if st.button("Bootstrap model metric", type="primary", key="boot_run3"):
                rng = np.random.default_rng(seed)
                yt = np.asarray(model_result["y_test"])
                pred = np.asarray(model_result["pred"])
                proba = model_result.get("proba")
                original = (reg_metrics(yt, pred) if model_result["problem"] == "regression" else clf_metrics(yt, pred, proba))[metric]
                values = []
                for _ in range(B):
                    index = rng.integers(0, len(yt), len(yt))
                    value = (
                        reg_metrics(yt[index], pred[index])
                        if model_result["problem"] == "regression"
                        else clf_metrics(yt[index], pred[index], proba[index] if proba is not None else None)
                    )[metric]
                    values.append(value)
                st.session_state.bootstrap = {
                    "result_type": "bootstrap",
                    "values": np.array(values),
                    "original": float(original),
                    "label": metric,
                    "quantity_kind": "model_metric",
                    "model_name": model_result.get("model_name"),
                    "target": model_result.get("target"),
                    "bootstrap_samples": int(B),
                }
    else:
        model_result = _bootstrap_model_preparation(df, seed)
        if model_result is not None:
            if model_result.get("model_family") == "lstm":
                st.info("Individual-prediction bootstrap is not enabled for the LSTM because it would require retraining the recurrent network hundreds of times. You can still bootstrap its saved held-out error metric.")
            else:
                row = st.slider("Held-out observation", 0, len(model_result["X_test"]) - 1, 0, key="boot_row")
                st.dataframe(model_result["X_test"].iloc[[row]], use_container_width=True)
                class_index = None
                class_label = None
                if model_result["problem"] == "classification" and hasattr(model_result["pipeline"], "predict_proba"):
                    class_names = model_result.get("class_names") or [str(i) for i in range(model_result.get("n_classes", 2))]
                    class_label = st.selectbox("Class probability to bootstrap", class_names, index=min(1, len(class_names) - 1), key="boot_prediction_class")
                    class_index = class_names.index(class_label)
                if st.button("Bootstrap individual prediction", type="primary", key="boot_run4"):
                    rng = np.random.default_rng(seed)
                    Xtr = model_result["X_train"].reset_index(drop=True)
                    ytr = pd.Series(model_result["y_train"]).reset_index(drop=True)
                    xnew = model_result["X_test"].iloc[[row]]
                    pipe = model_result["pipeline"]
                    if model_result["problem"] == "classification" and class_index is not None:
                        original = float(pipe.predict_proba(xnew)[0, class_index])
                        label = f"Predicted probability of {class_label}"
                    else:
                        original = float(np.ravel(pipe.predict(xnew))[0])
                        label = f"Prediction of {humanize(model_result.get('target', 'target'))}"
                    values = []
                    bar = st.progress(0)
                    for i in range(B):
                        indices = rng.integers(0, len(Xtr), len(Xtr))
                        model = clone(pipe)
                        model.fit(Xtr.iloc[indices], ytr.iloc[indices])
                        if model_result["problem"] == "classification" and class_index is not None:
                            value = float(model.predict_proba(xnew)[0, class_index])
                        else:
                            value = float(np.ravel(model.predict(xnew))[0])
                        values.append(value)
                        if (i + 1) % max(1, B // 20) == 0:
                            bar.progress((i + 1) / B)
                    bar.empty()
                    st.session_state.bootstrap = {
                        "result_type": "bootstrap",
                        "values": np.array(values),
                        "original": original,
                        "label": label,
                        "quantity_kind": "individual_prediction",
                        "model_name": model_result.get("model_name"),
                        "target": model_result.get("target"),
                        "held_out_row": int(row),
                        "bootstrap_samples": int(B),
                    }

    result = st.session_state.get("bootstrap")
    if isinstance(result, dict):
        values = np.asarray(result["values"], dtype=float)
        original = float(result["original"])
        lo, hi = np.percentile(values, [2.5, 97.5])
        bias = float(values.mean() - original)
        standard_error = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original estimate", f"{original:.4f}")
        c2.metric("Estimated bias", f"{bias:.4f}")
        c3.metric("Bootstrap standard error", f"{standard_error:.4f}")
        c4.metric("Percentile 95% interval", f"[{lo:.4f}, {hi:.4f}]")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(values, kde=True, ax=ax)
        ax.axvline(original, label="Original estimate")
        ax.axvline(lo, linestyle="--", label="2.5th percentile")
        ax.axvline(hi, linestyle="--", label="97.5th percentile")
        ax.legend(frameon=False)
        ax.set_xlabel(result["label"])
        ax.set_ylabel("Number of bootstrap resamples")
        axis_style(ax)
        show_fig(fig, "bootstrap.png", "boot_dl")
        st.markdown("### What the bootstrap result means")
        for note in simple_result_notes(result):
            st.markdown("- " + note)

def parse_ints(text):
    return sorted(set(int(x.strip()) for x in text.split(",") if x.strip().isdigit() and int(x.strip()) > 0))


def forecast_frame(df, date, target, lags, windows, horizon, exogenous, season):
    ordered = df[[date, target] + list(exogenous)].copy()
    ordered[date] = pd.to_datetime(ordered[date], errors="coerce")
    ordered = ordered.dropna(subset=[date, target]).sort_values(date).reset_index(drop=True)
    out = pd.DataFrame({
        "forecast_date": ordered[date].shift(-horizon),
        "forecast_target": ordered[target].shift(-horizon),
        "current_value": ordered[target],
    })
    feature_columns = ["current_value"]
    for lag in lags:
        column = f"lag_{lag}"
        out[column] = ordered[target].shift(lag)
        feature_columns.append(column)
    for window in windows:
        mean_column = f"rolling_mean_{window}"
        std_column = f"rolling_std_{window}"
        out[mean_column] = ordered[target].rolling(window, min_periods=window).mean()
        out[std_column] = ordered[target].rolling(window, min_periods=window).std()
        feature_columns.extend([mean_column, std_column])
    for column in exogenous:
        out[column] = ordered[column]
        feature_columns.append(column)
    seasonal_shift = season - horizon
    out["seasonal_naive"] = ordered[target].shift(seasonal_shift) if seasonal_shift >= 0 else np.nan
    out["mean_baseline"] = ordered[target].expanding().mean()
    required = ["forecast_date", "forecast_target"] + feature_columns
    return out.dropna(subset=required).reset_index(drop=True)


def lstm_sequence_data(df, date, target, exogenous, lookback, horizon, season):
    columns = [date, target] + list(exogenous)
    ordered = df[columns].copy()
    ordered[date] = pd.to_datetime(ordered[date], errors="coerce")
    for column in [target] + list(exogenous):
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
    ordered = ordered.dropna().sort_values(date).reset_index(drop=True)
    values = ordered[[target] + list(exogenous)].to_numpy(dtype=np.float32)
    target_values = ordered[target].to_numpy(dtype=np.float32)
    dates = ordered[date].to_numpy()

    sequences = []
    outcomes = []
    forecast_dates = []
    current_values = []
    seasonal_values = []
    mean_values = []
    for origin in range(lookback - 1, len(ordered) - horizon):
        target_index = origin + horizon
        sequences.append(values[origin - lookback + 1: origin + 1])
        outcomes.append(target_values[target_index])
        forecast_dates.append(dates[target_index])
        current_values.append(target_values[origin])
        seasonal_index = target_index - season
        seasonal_values.append(target_values[seasonal_index] if seasonal_index >= 0 else np.nan)
        mean_values.append(float(np.mean(target_values[: origin + 1])))

    return {
        "X": np.asarray(sequences, dtype=np.float32),
        "y": np.asarray(outcomes, dtype=np.float32),
        "dates": np.asarray(forecast_dates),
        "current": np.asarray(current_values, dtype=np.float32),
        "seasonal": np.asarray(seasonal_values, dtype=np.float32),
        "mean": np.asarray(mean_values, dtype=np.float32),
        "channel_names": [target] + list(exogenous),
    }


def lstm_params_ui(prefix="fc_lstm_"):
    parameters = {}
    with st.expander("⚙️ Parameters for Long Short-Term Memory Recurrent Neural Network (LSTM-RNN)", expanded=True):
        parameters["lookback"] = st.slider("Lookback sequence length", 3, 120, 12, 1, key=prefix + "lookback")
        parameters["units"] = st.slider("LSTM memory units", 8, 128, 32, 8, key=prefix + "units")
        parameters["dropout"] = st.slider("Dropout", 0.0, 0.5, 0.1, 0.05, key=prefix + "dropout")
        parameters["dense_units"] = st.slider("Dense-layer neurons; 0 removes the layer", 0, 128, 16, 8, key=prefix + "dense")
        parameters["learning_rate"] = st.number_input("Learning rate", 0.0001, 0.1, 0.001, format="%.4f", key=prefix + "lr")
        parameters["epochs"] = st.slider("Maximum training epochs", 10, 200, 50, 10, key=prefix + "epochs")
        parameters["batch_size"] = st.select_slider("Batch size", options=[8, 16, 32, 64, 128], value=32, key=prefix + "batch")
        parameters["validation_fraction"] = st.slider("Final training fraction used for validation", 0.10, 0.30, 0.20, 0.05, key=prefix + "validation")
        parameters["patience"] = st.slider("Early-stopping patience", 3, 20, 7, 1, key=prefix + "patience")
        st.caption(
            "An LSTM receives an ordered sequence rather than one fixed predictor row. Its recurrent memory can "
            "learn patterns across earlier time steps."
        )
    return parameters


def build_lstm(input_shape, parameters, seed=42):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    layers = [
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(parameters.get("units", 32), dropout=parameters.get("dropout", 0.1)),
    ]
    if parameters.get("dense_units", 16) > 0:
        layers.append(tf.keras.layers.Dense(parameters["dense_units"], activation="relu"))
    layers.append(tf.keras.layers.Dense(1))
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=parameters.get("learning_rate", 0.001)),
        loss="mse",
    )
    return model


def inverse_lstm_prediction(model, sequences, target_scaler):
    scaled = model.predict(sequences, verbose=0).reshape(-1, 1)
    return target_scaler.inverse_transform(scaled).ravel()


def finite_reg_metrics(y_true, prediction):
    y_true = np.asarray(y_true)
    prediction = np.asarray(prediction)
    valid = np.isfinite(y_true) & np.isfinite(prediction)
    if valid.sum() < 2:
        return None
    return reg_metrics(y_true[valid], prediction[valid])


def lstm_explanation_panel(result, prefix):
    st.subheader("How the LSTM used the sequence")
    t1, t2, t3 = st.tabs(["Sequence permutation importance", "Lag sensitivity", "Training history"])

    with t1:
        st.info(
            "Sequence permutation importance shuffles one input channel across held-out sequences while preserving "
            "the order inside each sequence. If root mean squared error rises substantially, the LSTM relied on that channel."
        )
        repeats = st.slider("Permutation repetitions", 3, 20, 5, key=prefix + "_seq_repeats")
        if st.button("Calculate sequence importance", key=prefix + "_seq_button"):
            rng = np.random.default_rng(42)
            base_rmse = math.sqrt(mean_squared_error(result["y_test"], result["pred"]))
            rows = []
            for channel_index, channel_name in enumerate(result["sequence_channel_names"]):
                increases = []
                for _ in range(repeats):
                    changed = result["X_test_sequence"].copy()
                    order = rng.permutation(len(changed))
                    changed[:, :, channel_index] = changed[order, :, channel_index]
                    changed_prediction = inverse_lstm_prediction(result["lstm_model"], changed, result["target_scaler"])
                    changed_rmse = math.sqrt(mean_squared_error(result["y_test"], changed_prediction))
                    increases.append(changed_rmse - base_rmse)
                rows.append({
                    "Sequence input": channel_name,
                    "Mean increase in RMSE": float(np.mean(increases)),
                    "Standard deviation": float(np.std(increases, ddof=1)) if len(increases) > 1 else 0.0,
                })
            st.session_state[prefix + "_sequence_importance"] = pd.DataFrame(rows).sort_values("Mean increase in RMSE", ascending=False)
        table = st.session_state.get(prefix + "_sequence_importance")
        if isinstance(table, pd.DataFrame):
            st.dataframe(tidy_frame(table,3), use_container_width=True, hide_index=True)
            top = table.sort_values("Mean increase in RMSE")
            fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.45)))
            ax.barh(
                top["Sequence input"].map(humanize),
                top["Mean increase in RMSE"],
                xerr=top["Standard deviation"],
            )
            ax.set_xlabel("Increase in held-out RMSE after shuffling")
            axis_style(ax)
            show_fig(fig, "lstm_sequence_importance.png", prefix + "_seq_dl")
            st.caption(
                "Longer positive bars indicate sequence inputs whose disruption harms forecasting more. Error bars "
                "show variation across repeated shuffles."
            )

    with t2:
        st.info(
            "Lag sensitivity replaces the target value at one position in every held-out sequence with its training "
            "mean. A larger increase in error indicates that the LSTM relied more strongly on that position in the lookback window."
        )
        if st.button("Calculate lag sensitivity", key=prefix + "_lag_button"):
            base_rmse = math.sqrt(mean_squared_error(result["y_test"], result["pred"]))
            lookback = result["lookback"]
            rows = []
            for position in range(lookback):
                changed = result["X_test_sequence"].copy()
                changed[:, position, 0] = 0.0
                changed_prediction = inverse_lstm_prediction(result["lstm_model"], changed, result["target_scaler"])
                changed_rmse = math.sqrt(mean_squared_error(result["y_test"], changed_prediction))
                steps_before = lookback - 1 - position
                rows.append({
                    "Steps before forecast origin": steps_before,
                    "Lag label": "Current value (t)" if steps_before == 0 else f"t-{steps_before}",
                    "Increase in RMSE": changed_rmse - base_rmse,
                })
            st.session_state[prefix + "_lag_sensitivity"] = pd.DataFrame(rows).sort_values("Steps before forecast origin")
        lag_table = st.session_state.get(prefix + "_lag_sensitivity")
        if isinstance(lag_table, pd.DataFrame):
            st.dataframe(lag_table, use_container_width=True, hide_index=True)
            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.plot(lag_table["Steps before forecast origin"], lag_table["Increase in RMSE"], marker="o")
            ax.set_xlabel("Steps before forecast origin (0 is the current value)")
            ax.set_ylabel("Increase in held-out RMSE")
            axis_style(ax)
            show_fig(fig, "lstm_lag_sensitivity.png", prefix + "_lag_dl")
            st.caption(
                "Points farther to the right represent older positions in the lookback sequence. Positive values "
                "mean the forecast became worse when that position was neutralized."
            )

    with t3:
        st.info(
            "The training-history chart shows loss after each epoch. Training loss measures fit to the training "
            "sequences; validation loss measures performance on the final portion of the training period used for early stopping."
        )
        history = result.get("training_history", {})
        if history:
            history_table = pd.DataFrame(history)
            history_table.index = np.arange(1, len(history_table) + 1)
            history_table.index.name = "Epoch"
            st.dataframe(history_table, use_container_width=True)
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.plot(history_table.index, history_table["loss"], label="Training loss")
            if "val_loss" in history_table:
                ax.plot(history_table.index, history_table["val_loss"], label="Validation loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Mean squared error on scaled target")
            ax.legend(frameon=False)
            axis_style(ax)
            show_fig(fig, "lstm_training_history.png", prefix + "_history_dl")
            st.caption(
                "A widening gap between training and validation loss can indicate overfitting. Early stopping restores "
                "the weights from the best validation epoch."
            )


def page_forecasting():
    df = get_df()
    st.title("Time Series Forecasting")
    guide(
        "Preserve time order, compare simple baselines with machine-learning forecasts, and use an LSTM for ordered sequences.",
        ["Change forecast horizon", "Change lag or lookback information", "Compare a tabular model with an LSTM"],
        ["Time-ordered error", "Whether the model beats naïve forecasts", "Which past values or sequence inputs matter"],
        ["Describe the backtest", "Report baseline and model error", "Explain one important lag or sequence input"],
        "Never randomly split a time series.",
    )
    before_you_run([
        ("Forecast horizon", "How many future rows ahead the model predicts. Horizon 1 predicts the next row; horizon 7 predicts seven rows ahead."),
        ("Final chronological test period", "The final portion of the time series hidden from training and used to evaluate forecasts of later observations."),
        ("Seasonal cycle length", "The number of rows in one repeating cycle, such as 24 for a daily cycle in hourly data or 7 for a weekly cycle in daily data."),
        ("Past lag", "An earlier target value. Lag 1 is one row earlier, lag 7 is seven rows earlier."),
        ("Rolling window", "A summary of the most recent rows, such as the seven-row mean or standard deviation."),
        ("Predictor known at forecast origin", "An additional variable genuinely available when the forecast is made. Future information must not be used."),
        ("LSTM lookback sequence", "The consecutive past time steps supplied to the recurrent neural network for each prediction."),
        ("Naïve latest-value baseline", "Predicts that the future target will equal the most recently observed target."),
        ("Seasonal naïve baseline", "Predicts using the target observed one full seasonal cycle earlier."),
        ("Expanding historical mean baseline", "Predicts using the average of all target values observed up to the forecast origin."),
        ("Mean absolute error", "The average absolute forecast mistake. Lower is better."),
        ("Root mean squared error", "Forecast error that gives extra weight to large mistakes. Lower is better."),
        ("R-squared", "The proportion of held-out variation explained by the forecast model; higher is generally better, but it must be read alongside forecast error and baselines."),
        ("Time-series cross-validation", "Repeats evaluation across expanding chronological splits so earlier rows always predict later rows."),
    ], note="A sophisticated forecasting model is useful only when it improves on simple, realistic baselines evaluated on the same future test period.")
    numbers = num_cols(df)
    if not numbers:
        return st.warning("A numerical target is required.")

    date = st.selectbox("Date or time column", df.columns, key="fc_date")
    target = st.selectbox("Forecast target", numbers, key="fc_target")
    model_options = REG_MODELS + ["Long Short-Term Memory Recurrent Neural Network (LSTM-RNN)"]
    name = st.selectbox(
        "Forecast model",
        model_options,
        index=model_options.index("Ridge Regression"),
        key="fc_model",
    )

    try:
        parsed = pd.to_datetime(df[date], errors="raise")
        if parsed.nunique() < 10:
            raise ValueError
    except Exception:
        return st.warning("Choose a usable date or time column.")
    st.caption(describe_time_spacing(parsed) + " Forecast horizon and seasonal cycle length are counted in these rows.")

    is_lstm = name.startswith("Long Short-Term Memory")
    if is_lstm:
        exogenous_options = [c for c in numbers if c not in [target, date]]
        exogenous = st.multiselect(
            "Optional numerical sequence inputs",
            exogenous_options,
            key="fc_exog_lstm",
        )
    else:
        exogenous = st.multiselect(
            "Optional predictors known at the forecast origin",
            [c for c in df if c not in [date, target]],
            key="fc_exog_tabular",
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        horizon = st.slider("Forecast horizon (future time steps)", 1, 30, 1, key="fc_h", help="The number of rows ahead to predict. A value of 1 predicts the next row.")
    with c2:
        test_percent = st.slider("Final period used for testing (%)", 15, 40, 25, 5, key="fc_test")
    with c3:
        season = st.slider("Seasonal cycle length (rows)", 2, 365, 7, key="fc_season", help="The number of rows in one repeating cycle, used by the seasonal naïve baseline.")
    st.caption(
        f"This run predicts **{horizon} row{'s' if horizon != 1 else ''} ahead**. "
        f"The seasonal baseline looks back **{season} rows** for the corresponding value from the previous cycle."
    )
    seed = st.number_input("Random seed", 0, 9999, 42, key="fc_seed")

    if is_lstm:
        parameters = lstm_params_ui()
        st.info(
            "The LSTM uses the previous lookback sequence to forecast the target at the selected horizon. "
            "It is evaluated on the final chronological test period and compared with simple forecasting baselines."
        )
        if st.button("Train LSTM and run time-ordered backtest", type="primary", key="fc_lstm_run"):
            try:
                import tensorflow as tf
            except Exception as exc:
                return st.error(f"TensorFlow is required for the LSTM option: {exc}")

            sequence_data = lstm_sequence_data(
                df,
                date,
                target,
                exogenous,
                parameters["lookback"],
                horizon,
                season,
            )
            if len(sequence_data["y"]) < 50:
                return st.warning("Too few ordered sequences remain. Reduce the lookback or horizon, or use a longer time series.")
            cut = int(len(sequence_data["y"]) * (1 - test_percent / 100))
            if cut < 30 or len(sequence_data["y"]) - cut < 10:
                return st.warning("The chronological training or test period is too small for this LSTM configuration.")

            X_train_raw = sequence_data["X"][:cut]
            X_test_raw = sequence_data["X"][cut:]
            y_train = sequence_data["y"][:cut]
            y_test = sequence_data["y"][cut:]

            feature_scaler = StandardScaler()
            feature_scaler.fit(X_train_raw.reshape(-1, X_train_raw.shape[-1]))
            X_train = feature_scaler.transform(X_train_raw.reshape(-1, X_train_raw.shape[-1])).reshape(X_train_raw.shape)
            X_test = feature_scaler.transform(X_test_raw.reshape(-1, X_test_raw.shape[-1])).reshape(X_test_raw.shape)
            target_scaler = StandardScaler()
            y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

            model = build_lstm(X_train.shape[1:], parameters, seed)
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=parameters["patience"],
                restore_best_weights=True,
            )
            with st.spinner("Training the LSTM on ordered sequences..."):
                history = model.fit(
                    X_train,
                    y_train_scaled,
                    epochs=parameters["epochs"],
                    batch_size=parameters["batch_size"],
                    validation_split=parameters["validation_fraction"],
                    shuffle=False,
                    callbacks=[early_stopping],
                    verbose=0,
                )
            prediction = inverse_lstm_prediction(model, X_test, target_scaler)
            train_prediction = inverse_lstm_prediction(model, X_train, target_scaler)

            comparison_rows = [{"Model": name, **reg_metrics(y_test, prediction)}]
            baseline_series = [
                ("Naïve latest value", sequence_data["current"][cut:]),
                (f"Seasonal naïve (cycle = {season} rows)", sequence_data["seasonal"][cut:]),
                ("Expanding historical mean", sequence_data["mean"][cut:]),
            ]
            for baseline_name, baseline_prediction in baseline_series:
                metrics = finite_reg_metrics(y_test, baseline_prediction)
                if metrics is not None:
                    comparison_rows.append({"Model": baseline_name, **metrics})

            result = {
                "problem": "regression",
                "context": "forecasting",
                "model_family": "lstm",
                "model_name": name,
                "parameters": parameters,
                "lstm_model": model,
                "feature_scaler": feature_scaler,
                "target_scaler": target_scaler,
                "features": sequence_data["channel_names"],
                "sequence_channel_names": sequence_data["channel_names"],
                "lookback": parameters["lookback"],
                "target": target,
                "date_column": date,
                "horizon": int(horizon),
                "season": int(season),
                "test_percent": int(test_percent),
                "X_train_sequence": X_train,
                "X_test_sequence": X_test,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "pred": prediction,
                "train_pred": train_prediction,
                "metrics": reg_metrics(y_test, prediction),
                "comparison": pd.DataFrame(comparison_rows),
                "dates": sequence_data["dates"][cut:],
                "training_history": history.history,
                "dataset_name": st.session_state.dataset_name,
            }
            save_result("forecast_result", result)
    else:
        lags = parse_ints(st.text_input("Past lags", "1,2,3,7", key="fc_lags"))
        windows = parse_ints(st.text_input("Rolling windows", "3,7", key="fc_windows"))
        parameters = params_ui(name, "regression", "fc_", True)
        use_cv = st.checkbox("Expanding time-series cross-validation", True, key="fc_cv")
        splits = st.slider("Time splits", 3, 10, 5, key="fc_splits") if use_cv else 5
        if st.button("Run time-ordered backtest", type="primary", key="fc_run"):
            frame = forecast_frame(df, date, target, lags, windows, horizon, exogenous, season)
            if len(frame) < 40:
                return st.warning("Too few rows remain. Reduce lags, windows, or horizon.")
            features = [c for c in frame if c not in ["forecast_date", "forecast_target", "seasonal_naive", "mean_baseline"]]
            cut = int(len(frame) * (1 - test_percent / 100))
            train, test = frame.iloc[:cut], frame.iloc[cut:]
            X_train, y_train = train[features], train.forecast_target
            X_test, y_test = test[features], test.forecast_target
            pipe = build_pipe(X_train, features, name, "regression", parameters, seed)
            pipe.fit(X_train, y_train)
            prediction = pipe.predict(X_test)

            comparison_rows = [{"Model": name, **reg_metrics(y_test, prediction)}]
            baseline_series = [
                ("Naïve latest value", test.current_value),
                (f"Seasonal naïve (cycle = {season} rows)", test.seasonal_naive),
                ("Expanding historical mean", test.mean_baseline),
            ]
            for baseline_name, baseline_prediction in baseline_series:
                metrics = finite_reg_metrics(y_test, baseline_prediction)
                if metrics is not None:
                    comparison_rows.append({"Model": baseline_name, **metrics})

            cv_scores = None
            if use_cv:
                safe_splits = min(splits, max(2, len(train) // 10))
                cv = TimeSeriesSplit(n_splits=safe_splits)
                cv_scores = -cross_val_score(
                    pipe,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring="neg_root_mean_squared_error",
                )
            save_result("forecast_result", {
                "problem": "regression",
                "context": "forecasting",
                "model_family": "tabular",
                "model_name": name,
                "parameters": parameters,
                "pipeline": pipe,
                "features": features,
                "target": target,
                "date_column": date,
                "horizon": int(horizon),
                "season": int(season),
                "test_percent": int(test_percent),
                "lags": list(lags),
                "rolling_windows": list(windows),
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "pred": prediction,
                "train_pred": pipe.predict(X_train),
                "metrics": reg_metrics(y_test, prediction),
                "comparison": pd.DataFrame(comparison_rows),
                "dates": test.forecast_date.to_numpy(),
                "cv_scores": cv_scores,
                "dataset_name": st.session_state.dataset_name,
            })

    result = st.session_state.get("forecast_result")
    if isinstance(result, dict) and result.get("target") == target and result.get("model_name") == name:
        st.dataframe(tidy_frame(result["comparison"],3), use_container_width=True, hide_index=True)
        st.markdown("### What the forecast comparison means")
        for note in forecast_comparison_notes(result):
            st.markdown("- " + note)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(pd.to_datetime(result["dates"]), result["y_test"], label="Observed")
        ax.plot(pd.to_datetime(result["dates"]), result["pred"], label="Forecast")
        ax.legend(frameon=False)
        ax.set_xlabel("Date")
        ax.set_ylabel(humanize(target))
        axis_style(ax)
        show_fig(fig, "forecast_backtest.png", "fc_plot_dl")
        st.caption(
            "The chart displays the final chronological holdout period. The model was trained only on observations "
            "that occurred before this test period."
        )
        if result.get("model_family") == "lstm":
            lstm_explanation_panel(result, "forecast_lstm_explain")
        else:
            if result.get("cv_scores") is not None:
                scores = result["cv_scores"]
                st.write(f"**Time-series cross-validation RMSE:** {scores.mean():.3f} ± {scores.std(ddof=1):.3f}")
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.scatter(np.arange(1, len(scores) + 1), scores)
                ax.axhline(scores.mean(), linestyle="--")
                ax.set_xlabel("Time split")
                ax.set_ylabel("RMSE")
                axis_style(ax)
                show_fig(fig, "forecast_cv.png", "fc_cv_dl")
                st.caption("Each point is the error from one expanding time-series split; the dashed line is the mean error.")
            explanation_panel(result, "forecast_explain")

# -----------------------------------------------------------------------------
# Computer vision
# -----------------------------------------------------------------------------
IMAGE_CLASSIFIER_NOTES = {
    "Logistic Regression": (
        "**Image input:** flattened red, green, and blue pixel values plus simple color summaries. "
        "**How it decides:** learns one weighted probability rule for each class. "
        "**Teaching point:** a clear linear baseline, but it does not explicitly learn nearby edges, textures, or shapes."
    ),
    "Linear Support Vector Machine": (
        "**Image input:** flattened pixel values and color summaries. "
        "**How it decides:** finds a straight maximum-margin boundary between image classes. "
        "**Teaching point:** often a strong conventional baseline, but it does not preserve the image's two-dimensional layout."
    ),
    "Radial Support Vector Machine": (
        "**Image input:** flattened pixel values and color summaries. "
        "**How it decides:** uses similarity-based curved boundaries to separate classes. "
        "**Teaching point:** can learn nonlinear class boundaries, although it still receives the image as one long feature row."
    ),
    "Random Forest": (
        "**Image input:** flattened pixel values and color summaries. "
        "**How it decides:** many decision trees split on pixel or color values and vote for the final class. "
        "**Teaching point:** captures nonlinear rules, but individual splits do not naturally represent local visual shapes."
    ),
    "K-Nearest Neighbors": (
        "**Image input:** flattened pixel values and color summaries. "
        "**How it decides:** finds the most numerically similar training images and uses their labels. "
        "**Teaching point:** easy to understand, but similarity can be sensitive to image position, lighting, and background."
    ),
    "Feedforward Neural Network (FFNN)": (
        "**Image input:** flattened pixel values and color summaries. "
        "**How it decides:** passes the long feature row through hidden neurons to learn nonlinear combinations of pixels. "
        "**Teaching point:** more flexible than a linear model, but it still does not explicitly preserve where pixels are located."
    ),
    "Small Convolutional Neural Network (CNN)": (
        "**Image input:** the original height × width × 3 color grid. "
        "**How it decides:** small convolution filters scan nearby pixels to learn edges, textures, and shapes; pooling reduces the representation before the final class probabilities are produced. "
        "**Teaching point:** unlike the flattened models, it is designed specifically to use the spatial structure of images."
    ),
}

IMAGE_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


def image_array(img, size=64):
    return np.asarray(img.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0


def image_vector_from_array(arr):
    return np.concatenate([
        arr.reshape(-1),
        arr.mean(axis=(0, 1)),
        arr.std(axis=(0, 1)),
    ])


def image_vector(img, size=64):
    return image_vector_from_array(image_array(img, size))


def normalize_image_split(parts):
    """Return train/val/test when a split folder appears above the class folder."""
    for part in reversed(parts[:-2]):
        normalized = IMAGE_SPLIT_ALIASES.get(part.strip().lower())
        if normalized:
            return normalized
    return "unspecified"


def infer_image_sequence_group(file_name):
    """Conservatively infer repeated capture groups such as paper01-005.

    The prefix must itself end in a digit. This avoids treating ordinary names such as
    image_001, image_002, ... as one giant sequence merely because they share 'image'.
    """
    stem = file_name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    patterns = [
        r"^(.+?\d+)[-_](?:frame[-_]?)?\d+$",
        r"^(.+?\d+)[-_](?:img|image)[-_]?\d+$",
    ]
    for pattern in patterns:
        match = re.match(pattern, stem, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return None


def _stable_image_seed(text, seed=42):
    digest = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
    return (seed + int(digest[:8], 16)) % (2**32 - 1)


def _balanced_image_sample(records, maximum, seed=42):
    """Sample across detected groups instead of taking the first files in ZIP order."""
    if len(records) <= maximum:
        return list(records)
    buckets = {}
    for record in records:
        bucket = record.get("sequence_group") or "file::" + record["file"]
        buckets.setdefault(bucket, []).append(record)
    rng = np.random.default_rng(_stable_image_seed(records[0]["sample_key"], seed))
    bucket_names = list(buckets)
    rng.shuffle(bucket_names)
    for bucket_name in bucket_names:
        rng.shuffle(buckets[bucket_name])
    selected = []
    while len(selected) < maximum:
        progressed = False
        for bucket_name in bucket_names:
            if buckets[bucket_name] and len(selected) < maximum:
                selected.append(buckets[bucket_name].pop())
                progressed = True
        if not progressed:
            break
    return selected


def _image_archive_records(upload):
    try:
        upload.seek(0)
    except Exception:
        pass
    valid = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    records = []
    with zipfile.ZipFile(upload) as archive:
        names = [
            name for name in archive.namelist()
            if name.lower().endswith(valid) and not name.startswith("__MACOSX/")
        ]
        for name in names:
            parts = [part for part in name.split("/") if part]
            if len(parts) < 2:
                continue
            label = parts[-2]
            split = normalize_image_split(parts)
            records.append({
                "file": name,
                "class": label,
                "split": split,
                "sequence_group": infer_image_sequence_group(name),
                "sample_key": f"{split}::{label}",
            })
    try:
        upload.seek(0)
    except Exception:
        pass
    return records


def _split_pair_rows(records, value_field):
    """List values occurring in more than one predefined split for the same class."""
    locations = {}
    for record in records:
        value = record.get(value_field)
        split = record.get("split")
        if not value or split == "unspecified":
            continue
        key = (record["class"], value)
        locations.setdefault(key, set()).add(split)
    rows = []
    for (label, value), splits in locations.items():
        if len(splits) > 1:
            rows.append({
                "Class": label,
                "Related-image group": value,
                "Appears in splits": ", ".join(sorted(splits)),
            })
    return rows


def read_image_zip_dataset(upload, max_per_class=60, size=64):
    """Read an image ZIP while preserving split, class, sequence, and duplicate metadata."""
    all_records = _image_archive_records(upload)
    if not all_records:
        return {
            "vectors": np.empty((0, 0), dtype=np.float32),
            "images": np.empty((0, size, size, 3), dtype=np.float32),
            "labels": np.array([]),
            "manifest": pd.DataFrame(),
            "audit": {},
        }

    by_source = {}
    for record in all_records:
        key = (record["split"], record["class"])
        by_source.setdefault(key, []).append(record)
    selected_records = []
    for key, records in sorted(by_source.items()):
        selected_records.extend(_balanced_image_sample(records, max_per_class, seed=42))
    selected_names = {record["file"] for record in selected_records}

    vectors, images, labels, loaded_rows = [], [], [], []
    hash_locations = {}
    try:
        upload.seek(0)
    except Exception:
        pass
    with zipfile.ZipFile(upload) as archive:
        for record in all_records:
            try:
                raw = archive.read(record["file"])
            except Exception:
                continue
            digest = hashlib.sha256(raw).hexdigest()
            record["sha256"] = digest
            hash_locations.setdefault(digest, []).append(record)
            if record["file"] not in selected_names:
                continue
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                arr = image_array(img, size)
            except Exception:
                continue
            images.append(arr)
            vectors.append(image_vector_from_array(arr))
            labels.append(record["class"])
            loaded_rows.append({
                "File": record["file"],
                "Split": record["split"],
                "Class": record["class"],
                "Sequence group": record.get("sequence_group") or "Not detected",
                "Width": img.width,
                "Height": img.height,
                "SHA256": digest,
            })
    try:
        upload.seek(0)
    except Exception:
        pass

    if not vectors:
        return {
            "vectors": np.empty((0, 0), dtype=np.float32),
            "images": np.empty((0, size, size, 3), dtype=np.float32),
            "labels": np.array([]),
            "manifest": pd.DataFrame(),
            "audit": {},
        }

    manifest = pd.DataFrame(loaded_rows)
    split_array = manifest["Split"].to_numpy(dtype=object)
    sequence_array = manifest["Sequence group"].replace("Not detected", None).to_numpy(dtype=object)
    hash_array = manifest["SHA256"].to_numpy(dtype=object)
    file_array = manifest["File"].to_numpy(dtype=object)
    label_array = np.asarray(labels)

    duplicate_hashes = {
        digest for digest, records in hash_locations.items() if len(records) > 1
    }
    group_keys, meaningful_groups = [], []
    for label, sequence, digest, file_name in zip(
        label_array, sequence_array, hash_array, file_array
    ):
        if sequence:
            group_keys.append(f"{label}::sequence::{sequence}")
            meaningful_groups.append(True)
        elif digest in duplicate_hashes:
            group_keys.append(f"{label}::duplicate::{digest}")
            meaningful_groups.append(True)
        else:
            group_keys.append(f"{label}::file::{file_name}")
            meaningful_groups.append(False)

    sequence_overlap_rows = _split_pair_rows(all_records, "sequence_group")
    duplicate_overlap_rows = []
    duplicate_label_conflict_rows = []
    for digest, records in hash_locations.items():
        splits = {record["split"] for record in records if record["split"] != "unspecified"}
        classes = sorted({record["class"] for record in records})
        if len(splits) > 1:
            duplicate_overlap_rows.append({
                "Class": ", ".join(classes),
                "SHA256 prefix": digest[:12],
                "Appears in splits": ", ".join(sorted(splits)),
            })
        if len(classes) > 1:
            duplicate_label_conflict_rows.append({
                "SHA256 prefix": digest[:12],
                "Conflicting classes": ", ".join(classes),
                "Files": len(records),
            })

    split_counts_full = pd.DataFrame(all_records).groupby(
        ["split", "class"]
    ).size().reset_index(name="Images in ZIP")
    split_counts_loaded = manifest.groupby(
        ["Split", "Class"]
    ).size().reset_index(name="Images loaded")
    predefined_splits = {
        record["split"] for record in all_records if record["split"] != "unspecified"
    }
    split_classes = {
        split: {record["class"] for record in all_records if record["split"] == split}
        for split in predefined_splits
    }
    has_predefined = "train" in predefined_splits and bool({"val", "test"} & predefined_splits)
    class_mismatch = False
    if has_predefined:
        train_classes = split_classes.get("train", set())
        for evaluation_split in [split for split in ("val", "test") if split in predefined_splits]:
            if split_classes.get(evaluation_split, set()) != train_classes:
                class_mismatch = True

    known_contamination = bool(
        sequence_overlap_rows
        or duplicate_overlap_rows
        or duplicate_label_conflict_rows
        or class_mismatch
    )
    predefined_clean = bool(has_predefined and not known_contamination)

    labels_by_class = {}
    for index, label in enumerate(label_array):
        labels_by_class.setdefault(label, []).append(index)
    group_aware_possible = True
    for label, indices in labels_by_class.items():
        class_groups = {group_keys[index] for index in indices}
        if len(class_groups) < 2:
            group_aware_possible = False
            break
    meaningful_group_count = len({
        group_keys[index] for index, is_meaningful in enumerate(meaningful_groups)
        if is_meaningful
    })
    meaningful_coverage = float(np.mean(meaningful_groups)) if meaningful_groups else 0.0
    reliable_group_signal = meaningful_group_count > 0 or bool(
        sequence_overlap_rows or duplicate_overlap_rows
    )

    audit = {
        "has_predefined": has_predefined,
        "predefined_splits": sorted(predefined_splits),
        "predefined_clean": predefined_clean,
        "class_mismatch": class_mismatch,
        "known_contamination": known_contamination,
        "sequence_overlap_rows": sequence_overlap_rows,
        "duplicate_overlap_rows": duplicate_overlap_rows,
        "duplicate_label_conflict_rows": duplicate_label_conflict_rows,
        "group_aware_possible": group_aware_possible,
        "reliable_group_signal": reliable_group_signal,
        "meaningful_group_count": meaningful_group_count,
        "meaningful_coverage": meaningful_coverage,
        "split_counts_full": split_counts_full,
        "split_counts_loaded": split_counts_loaded,
        "total_files": len(all_records),
        "loaded_files": len(label_array),
    }
    return {
        "vectors": np.vstack(vectors).astype(np.float32),
        "images": np.stack(images).astype(np.float32),
        "labels": label_array,
        "manifest": manifest,
        "splits": split_array,
        "sequence_groups": sequence_array,
        "hashes": hash_array,
        "files": file_array,
        "group_keys": np.asarray(group_keys, dtype=object),
        "meaningful_groups": np.asarray(meaningful_groups, dtype=bool),
        "audit": audit,
    }


def read_zip(upload, max_per_class=60, size=64):
    """Compatibility wrapper for earlier tests and external classroom code."""
    dataset = read_image_zip_dataset(upload, max_per_class, size)
    return dataset["vectors"], dataset["images"], dataset["labels"], dataset["manifest"]


def _stratified_group_holdout_indices(labels, groups, test_size=0.25, seed=42):
    """Place complete groups in one side while retaining every class in both sides."""
    labels = np.asarray(labels)
    groups = np.asarray(groups, dtype=object)
    training_indices, evaluation_indices = [], []
    for class_number, label in enumerate(sorted(np.unique(labels).tolist())):
        class_indices = np.where(labels == label)[0]
        group_map = {}
        for index in class_indices:
            group_map.setdefault(groups[index], []).append(int(index))
        if len(group_map) < 2:
            raise ValueError(f"Class '{label}' does not contain at least two independent groups.")
        rng = np.random.default_rng(_stable_image_seed(f"{label}::{class_number}", seed))
        items = list(group_map.items())
        rng.shuffle(items)
        target = max(1, int(round(len(class_indices) * float(test_size))))
        selected_groups, selected_count = [], 0
        remaining = list(items)
        while remaining and len(selected_groups) < len(items) - 1:
            best_position = min(
                range(len(remaining)),
                key=lambda position: abs(
                    target - (selected_count + len(remaining[position][1]))
                ),
            )
            group_name, group_indices = remaining.pop(best_position)
            current_distance = abs(target - selected_count)
            proposed_distance = abs(target - (selected_count + len(group_indices)))
            if selected_groups and proposed_distance > current_distance and selected_count >= target:
                break
            selected_groups.append((group_name, group_indices))
            selected_count += len(group_indices)
            if selected_count >= target:
                break
        if not selected_groups:
            selected_groups = [items[0]]
        evaluation_group_names = {group_name for group_name, _ in selected_groups}
        for group_name, group_indices in items:
            if group_name in evaluation_group_names:
                evaluation_indices.extend(group_indices)
            else:
                training_indices.extend(group_indices)
    return np.asarray(sorted(training_indices)), np.asarray(sorted(evaluation_indices))


def _random_image_holdout_indices(labels, test_size=0.25, seed=42):
    indices = np.arange(len(labels))
    try:
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
    except Exception:
        return train_test_split(indices, test_size=test_size, random_state=seed)


def image_evaluation_plan(dataset, holdout_fraction=0.25, seed=42):
    labels = dataset["labels"]
    splits = dataset["splits"]
    audit = dataset["audit"]

    if audit.get("duplicate_label_conflict_rows"):
        return {
            "method": "blocked",
            "train_indices": np.array([], dtype=int),
            "evaluation_indices": np.array([], dtype=int),
            "cnn_validation_indices": np.array([], dtype=int),
            "evaluation_split": None,
            "evaluation_label": "Evaluation",
            "design": "Evaluation blocked",
            "warning": (
                "At least one exact image file appears under different class labels. "
                "Correct the ZIP because no honest classifier can learn from contradictory labels."
            ),
        }

    if audit.get("predefined_clean"):
        train_indices = np.where(splits == "train")[0]
        evaluation_split = "test" if "test" in audit["predefined_splits"] else "val"
        evaluation_indices = np.where(splits == evaluation_split)[0]
        provided_validation_indices = (
            np.where(splits == "val")[0]
            if evaluation_split == "test" and "val" in audit["predefined_splits"]
            else np.array([], dtype=int)
        )
        return {
            "method": "predefined",
            "train_indices": train_indices,
            "evaluation_indices": evaluation_indices,
            "cnn_validation_indices": provided_validation_indices,
            "evaluation_split": evaluation_split,
            "evaluation_label": "Final test" if evaluation_split == "test" else "Validation",
            "design": (
                "Provided train/validation/test split"
                if evaluation_split == "test" and len(provided_validation_indices)
                else f"Provided train/{evaluation_split} split"
            ),
            "warning": None,
        }

    should_use_groups = bool(
        audit.get("group_aware_possible") and audit.get("reliable_group_signal")
    )
    if should_use_groups:
        train_indices, evaluation_indices = _stratified_group_holdout_indices(
            labels,
            dataset["group_keys"],
            test_size=holdout_fraction,
            seed=seed,
        )
        warning = None
        if audit.get("has_predefined") and audit.get("known_contamination"):
            warning = (
                "The supplied split was not used because related sequences, exact duplicates, "
                "or inconsistent classes were detected across folders. Complete known groups were "
                "reassigned to one side only."
            )
        return {
            "method": "group-aware",
            "train_indices": train_indices,
            "evaluation_indices": evaluation_indices,
            "cnn_validation_indices": np.array([], dtype=int),
            "evaluation_split": "group-aware holdout",
            "evaluation_label": "Group-aware holdout",
            "design": "Sequence- and duplicate-aware holdout",
            "warning": warning,
        }

    if audit.get("has_predefined") and audit.get("known_contamination"):
        return {
            "method": "blocked",
            "train_indices": np.array([], dtype=int),
            "evaluation_indices": np.array([], dtype=int),
            "cnn_validation_indices": np.array([], dtype=int),
            "evaluation_split": None,
            "evaluation_label": "Evaluation",
            "design": "Evaluation blocked",
            "warning": (
                "The provided folders contain known overlap, but the loaded subset does not contain "
                "enough independent groups to rebuild an honest holdout. Increase the maximum images "
                "per class or provide a cleaner split."
            ),
        }

    train_indices, evaluation_indices = _random_image_holdout_indices(
        labels, test_size=holdout_fraction, seed=seed
    )
    return {
        "method": "random",
        "train_indices": np.asarray(train_indices),
        "evaluation_indices": np.asarray(evaluation_indices),
        "cnn_validation_indices": np.array([], dtype=int),
        "evaluation_split": "random holdout",
        "evaluation_label": "Random holdout",
        "design": "Stratified random image-level holdout",
        "warning": (
            "No reliable predefined split or repeated sequence groups were detected. Related or "
            "near-duplicate images could still occur on both sides if the filenames do not identify them."
        ),
    }


def _cnn_fit_validation_indices(dataset, plan, seed=42):
    provided = np.asarray(plan.get("cnn_validation_indices", []), dtype=int)
    if len(provided):
        return np.asarray(plan["train_indices"], dtype=int), provided, "Provided validation folder"

    training_indices = np.asarray(plan["train_indices"], dtype=int)
    labels = dataset["labels"][training_indices]
    groups = dataset["group_keys"][training_indices]
    meaningful = dataset["meaningful_groups"][training_indices]
    can_group = bool(np.any(meaningful))
    if can_group:
        try:
            relative_fit, relative_validation = _stratified_group_holdout_indices(
                labels, groups, test_size=0.20, seed=seed + 1
            )
            return (
                training_indices[relative_fit],
                training_indices[relative_validation],
                "Internal group-aware validation subset",
            )
        except Exception:
            pass
    try:
        relative_fit, relative_validation = train_test_split(
            np.arange(len(training_indices)),
            test_size=0.20,
            random_state=seed + 1,
            stratify=labels,
        )
    except Exception:
        relative_fit, relative_validation = train_test_split(
            np.arange(len(training_indices)),
            test_size=0.20,
            random_state=seed + 1,
        )
    return (
        training_indices[relative_fit],
        training_indices[relative_validation],
        "Internal stratified validation subset",
    )


@st.cache_resource(show_spinner=False)
def imagenet_model():
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    return MobileNetV2(weights="imagenet")


def instant_predict(img, top=5):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
    from tensorflow.keras.preprocessing.image import img_to_array
    x = np.expand_dims(img_to_array(img.convert("RGB").resize((224, 224))), 0)
    output = imagenet_model().predict(preprocess_input(x), verbose=0)
    decoded = decode_predictions(output, top=top)[0]
    return pd.DataFrame([
        {"Prediction": label.replace("_", " "), "Confidence": float(probability)}
        for _, label, probability in decoded
    ])


def tensorflow_ok():
    try:
        import tensorflow  # noqa: F401
        return True, ""
    except Exception as exc:
        return False, str(exc)


def image_model(name, parameters):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, C=parameters.get("C", 1.0))
    if name == "Linear Support Vector Machine":
        return SVC(kernel="linear", C=parameters.get("C", 1.0), probability=True)
    if name == "Radial Support Vector Machine":
        return SVC(kernel="rbf", C=parameters.get("C", 1.0), probability=True)
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=parameters.get("trees", 150),
            max_depth=parameters.get("depth"),
            random_state=42,
        )
    if name == "K-Nearest Neighbors":
        return KNeighborsClassifier(n_neighbors=parameters.get("neighbors", 5))
    return MLPClassifier(
        hidden_layer_sizes=(parameters.get("hidden", 64),),
        max_iter=400,
        random_state=42,
    )


def build_small_cnn(input_shape, class_count, parameters, seed=42):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(parameters.get("dense", 32), activation="relu"),
        tf.keras.layers.Dropout(parameters.get("dropout", 0.20)),
        tf.keras.layers.Dense(class_count, activation="softmax"),
    ])
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=parameters.get("learning_rate", 0.001)
    )
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def image_params(name):
    parameters = {}
    with st.expander(f"⚙️ Parameters for {name}", expanded=True):
        if name in [
            "Logistic Regression",
            "Linear Support Vector Machine",
            "Radial Support Vector Machine",
        ]:
            parameters["C"] = st.slider(
                "Penalty C", 0.01, 10.0, 1.0, 0.01, key="img_C"
            )
        elif name == "Random Forest":
            parameters["trees"] = st.slider(
                "Trees", 50, 400, 150, 50, key="img_trees"
            )
            depth = st.slider(
                "Maximum depth; 0 is unlimited", 0, 30, 10, key="img_depth"
            )
            parameters["depth"] = None if depth == 0 else depth
        elif name == "K-Nearest Neighbors":
            parameters["neighbors"] = st.slider(
                "Neighbors", 1, 20, 5, key="img_neighbors"
            )
        elif name == "Small Convolutional Neural Network (CNN)":
            parameters["dense"] = st.slider(
                "Dense-layer neurons", 16, 128, 32, 16, key="img_cnn_dense"
            )
            parameters["dropout"] = st.slider(
                "Dropout", 0.0, 0.5, 0.20, 0.05, key="img_cnn_dropout"
            )
            parameters["learning_rate"] = st.number_input(
                "Learning rate",
                min_value=0.0001,
                max_value=0.01,
                value=0.001,
                step=0.0001,
                format="%.4f",
                key="img_cnn_lr",
            )
            parameters["epochs"] = st.slider(
                "Maximum training epochs", 5, 50, 15, 5, key="img_cnn_epochs"
            )
            parameters["batch_size"] = st.select_slider(
                "Batch size", options=[8, 16, 32], value=16, key="img_cnn_batch"
            )
            parameters["patience"] = st.slider(
                "Early-stopping patience", 2, 8, 4, 1, key="img_cnn_patience"
            )
            st.caption(
                "The architecture is intentionally small: two convolution-and-pooling stages, "
                "one dense layer, and a softmax output."
            )
        else:
            parameters["hidden"] = st.slider(
                "Hidden neurons", 16, 256, 64, 16, key="img_hidden"
            )
    return parameters


def image_model_note(name):
    st.markdown("#### How the selected model classifies images")
    st.info(IMAGE_CLASSIFIER_NOTES[name])


def image_result_display(result, labels):
    metrics = result["metrics"]
    evaluation_label = result.get("evaluation_label", "Holdout")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{evaluation_label} accuracy", f"{metrics['Accuracy']:.3f}")
    c2.metric("Weighted F1", f"{metrics['F1-score']:.3f}")
    c3.metric("Classes", len(labels))
    st.caption(
        f"Evaluation design: **{result.get('evaluation_design', 'Held-out evaluation')}**. "
        f"Training images: **{result.get('training_count', 0)}**; "
        f"evaluation images: **{result.get('evaluation_count', 0)}**."
    )

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        confusion_matrix(result["y_test"], result["pred"], labels=labels),
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    show_fig(fig, "image_confusion.png", "img_cm_dl")
    st.dataframe(
        pd.DataFrame(
            classification_report(
                result["y_test"],
                result["pred"],
                output_dict=True,
                zero_division=0,
            )
        ).T,
        use_container_width=True,
    )
    st.markdown("### What this image-classification result means")
    for note in simple_result_notes(result):
        st.markdown("- " + note)
    if result.get("evaluation_method") == "random":
        st.warning(
            "This score comes from a random image-level holdout. It is useful for classroom practice, "
            "but it is weaker evidence of real-world generalization when related images may exist."
        )


def _render_image_audit(dataset, plan):
    audit = dataset["audit"]
    st.markdown("### Evaluation audit")
    with st.expander("See how the app protects the evaluation", expanded=True):
        st.dataframe(audit["split_counts_full"], use_container_width=True, hide_index=True)
        st.caption(
            "The table above describes the complete ZIP. The model uses the sampled images shown in the loaded-count table below."
        )
        st.dataframe(audit["split_counts_loaded"], use_container_width=True, hide_index=True)
        if plan["method"] == "predefined":
            st.success(
                f"Clean predefined folders detected. The app will preserve them and use a **{plan['design']}**."
            )
        elif plan["method"] == "group-aware":
            st.success(
                "The app will use a group-aware holdout. Every known sequence or exact-duplicate group stays entirely in training or entirely in evaluation."
            )
        elif plan["method"] == "random":
            st.warning(plan["warning"])
        else:
            st.error(plan["warning"])

        if audit.get("sequence_overlap_rows"):
            st.warning(
                f"Detected **{len(audit['sequence_overlap_rows'])}** class-sequence group(s) appearing in more than one supplied split."
            )
            st.dataframe(pd.DataFrame(audit["sequence_overlap_rows"]), use_container_width=True, hide_index=True)
        else:
            st.caption("No filename-based sequence overlap was detected across supplied splits.")
        if audit.get("duplicate_overlap_rows"):
            st.warning(
                f"Detected **{len(audit['duplicate_overlap_rows'])}** exact image hash(es) appearing in more than one supplied split."
            )
            st.dataframe(pd.DataFrame(audit["duplicate_overlap_rows"]), use_container_width=True, hide_index=True)
        else:
            st.caption("No exact duplicate image bytes were detected across supplied splits.")
        if audit.get("duplicate_label_conflict_rows"):
            st.error(
                "Exact image files were found under conflicting class labels. Training is blocked until the labels are corrected."
            )
            st.dataframe(
                pd.DataFrame(audit["duplicate_label_conflict_rows"]),
                use_container_width=True,
                hide_index=True,
            )
        if audit.get("class_mismatch"):
            st.error("The supplied train, validation, or test folders do not contain the same class set.")
        st.caption(
            f"Known related-group coverage among loaded images: **{audit.get('meaningful_coverage', 0.0):.1%}**. "
            "Filename grouping is conservative and cannot detect every visually similar image."
        )

    training_indices = np.asarray(plan.get("train_indices", []), dtype=int)
    evaluation_indices = np.asarray(plan.get("evaluation_indices", []), dtype=int)
    if len(training_indices) and len(evaluation_indices):
        c1, c2, c3 = st.columns(3)
        c1.metric("Training images", len(training_indices))
        c2.metric("Evaluation images", len(evaluation_indices))
        train_groups = set(dataset["group_keys"][training_indices])
        evaluation_groups = set(dataset["group_keys"][evaluation_indices])
        c3.metric("Known group overlap", len(train_groups & evaluation_groups))


def page_computer_vision():
    st.title("Computer Vision")
    guide(
        "Represent images numerically and compare pretrained recognition with a classroom-trained classifier.",
        ["Upload one image", "Upload labeled image folders", "Choose a classifier"],
        ["Pixels and channels", "Confidence is not certainty", "Validation errors"],
        ["Report the prediction", "Explain an error", "State why validation matters"],
        "A general ImageNet prediction is not a specialist diagnosis.",
    )
    before_you_run([
        ("Pixel", "A numerical image element at a particular row and column."),
        ("Image channel", "A numerical layer such as red, green, or blue."),
        ("Class label", "The category assigned to an image for supervised learning."),
        ("Flattened image features", "Pixel values rearranged into one long row so a conventional model can read them."),
        ("Convolution", "A small learnable filter that scans nearby pixels to detect local patterns such as edges and textures."),
        ("Pooling", "A step that reduces the spatial size of learned feature maps while retaining important patterns."),
        ("Predefined split", "Training, validation, and test folders supplied by the dataset creator and kept separate when they pass the leakage audit."),
        ("Group leakage", "Related frames or versions of the same source image appear in both training and evaluation, making performance look too strong."),
        ("Group-aware holdout", "A split that keeps every known sequence or duplicate group entirely on one side."),
        ("Prediction confidence", "The model's estimated probability or score; it is not a guarantee of correctness."),
        ("Validation accuracy", "Accuracy on a provided validation folder; useful for evaluation, but not an untouched final test if it guides model choices."),
        ("Final test accuracy", "Accuracy on a separate test folder that was not used for training or early stopping."),
        ("Weighted F1-score", "A precision-recall summary that accounts for class frequencies."),
        ("Confusion matrix", "A table showing which observed classes were predicted correctly or confused with other classes."),
        ("Generalization", "How well the model works on genuinely new images not used for training."),
    ])

    mode = st.radio(
        "Activity",
        ["Instant image recognition", "Train a small image classifier"],
        horizontal=True,
        key="cv_mode",
    )
    if mode.startswith("Instant"):
        ok, error = tensorflow_ok()
        if not ok:
            st.error("TensorFlow is required for instant recognition.")
            st.caption(error)
            return
        upload = st.file_uploader(
            "Upload image", type=["png", "jpg", "jpeg", "webp"], key="instant_upload"
        )
        top = st.slider("Top predictions", 1, 10, 5, key="instant_top")
        if upload:
            img = Image.open(upload).convert("RGB")
            st.image(img, width=400)
            arr = np.asarray(img)
            c1, c2, c3 = st.columns(3)
            c1.metric("Height", arr.shape[0])
            c2.metric("Width", arr.shape[1])
            c3.metric("Channels", arr.shape[2])
            if st.button("Recognize image", type="primary", key="instant_run"):
                with st.spinner("Predicting..."):
                    table = instant_predict(img, top)
                st.dataframe(table, use_container_width=True)
                st.success(
                    f"Top prediction: {table.iloc[0].Prediction} "
                    f"({table.iloc[0].Confidence:.1%})"
                )
        return

    st.markdown("""The app accepts either of these structures:

**Preferred predefined split**
```text
dataset.zip
├── train/
│   ├── class_1/
│   └── class_2/
├── val/                 # optional when test exists
│   ├── class_1/
│   └── class_2/
└── test/                # optional
    ├── class_1/
    └── class_2/
```

**Class folders only**
```text
dataset.zip
├── class_1/
└── class_2/
```

The app audits predefined folders before using them. If related sequences cross the supplied folders, it rebuilds a group-aware holdout. A random image-level holdout is used only when no reliable split or group structure is available.""")

    c1, c2 = st.columns(2)
    with c1:
        size = st.selectbox("Image resize", [32, 48, 64, 96], index=2, key="img_size")
    with c2:
        maxclass = st.slider(
            "Maximum images per class per source split",
            10,
            200,
            60,
            10,
            key="img_max",
        )
    st.caption(
        "For a ZIP with train and validation folders, the limit is applied separately to each class in each folder."
    )

    upload = st.file_uploader("Upload labeled ZIP", type=["zip"], key="img_zip")
    if upload is None:
        return

    upload_size = getattr(upload, "size", None)
    cache_key = (
        getattr(upload, "file_id", None),
        getattr(upload, "name", "uploaded.zip"),
        upload_size,
        int(size),
        int(maxclass),
    )
    if st.session_state.get("image_dataset_cache_key") != cache_key:
        with st.spinner("Reading images and auditing the evaluation structure..."):
            dataset = read_image_zip_dataset(upload, maxclass, size)
        st.session_state.image_dataset_cache_key = cache_key
        st.session_state.image_dataset_cache = dataset
        for key in [
            "image_pipe",
            "image_cnn_model",
            "image_label_encoder",
            "image_model_kind",
            "image_trained_model_name",
            "image_result",
        ]:
            st.session_state.pop(key, None)
    else:
        dataset = st.session_state.get("image_dataset_cache", {})

    y = dataset.get("labels", np.array([]))
    if len(y) == 0:
        return st.error("No usable images found. Check the folder structure.")
    if len(np.unique(y)) < 2:
        return st.error("At least two image-class folders are required.")

    audit = dataset["audit"]
    if audit.get("predefined_clean"):
        holdout_fraction = 0.25
        st.caption("The test-proportion control is not needed because a clean supplied holdout will be preserved.")
    else:
        holdout_fraction = st.slider(
            "Holdout proportion when the app must create the split",
            0.15,
            0.50,
            0.25,
            0.05,
            key="img_test",
        )
    plan = image_evaluation_plan(dataset, holdout_fraction, seed=42)

    st.success(f"Loaded {len(y)} images from {len(np.unique(y))} classes.")
    st.dataframe(
        dataset["manifest"].drop(columns=["SHA256"], errors="ignore").head(50),
        use_container_width=True,
    )
    st.bar_chart(pd.Series(y).value_counts())
    _render_image_audit(dataset, plan)
    if plan["method"] == "blocked":
        return
    if plan.get("warning"):
        st.warning(plan["warning"])

    classifier_options = [
        "Logistic Regression",
        "Linear Support Vector Machine",
        "Radial Support Vector Machine",
        "Random Forest",
        "K-Nearest Neighbors",
        "Feedforward Neural Network (FFNN)",
        "Small Convolutional Neural Network (CNN)",
    ]
    name = st.selectbox("Classifier", classifier_options, key="img_model")
    image_model_note(name)
    parameters = image_params(name)

    if st.button("Train image classifier", type="primary", key="img_train"):
        train_indices = np.asarray(plan["train_indices"], dtype=int)
        evaluation_indices = np.asarray(plan["evaluation_indices"], dtype=int)
        X_vectors = dataset["vectors"]
        X_images = dataset["images"]

        if name == "Small Convolutional Neural Network (CNN)":
            ok, error = tensorflow_ok()
            if not ok:
                st.error("TensorFlow is required for the convolutional neural network option.")
                st.caption(error)
                return

            label_encoder = LabelEncoder()
            encoded_y = label_encoder.fit_transform(y)
            fit_indices, validation_indices, validation_design = _cnn_fit_validation_indices(
                dataset, plan, seed=42
            )

            import tensorflow as tf

            model = build_small_cnn(
                X_images.shape[1:],
                len(label_encoder.classes_),
                parameters,
                seed=42,
            )
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=parameters["patience"],
                restore_best_weights=True,
            )
            with st.spinner("Training the small convolutional neural network..."):
                history = model.fit(
                    X_images[fit_indices],
                    encoded_y[fit_indices],
                    validation_data=(
                        X_images[validation_indices],
                        encoded_y[validation_indices],
                    ),
                    epochs=parameters["epochs"],
                    batch_size=parameters["batch_size"],
                    shuffle=True,
                    callbacks=[early_stopping],
                    verbose=0,
                )
                probabilities = model.predict(X_images[evaluation_indices], verbose=0)

            predicted_codes = probabilities.argmax(axis=1)
            pred = label_encoder.inverse_transform(predicted_codes)
            y_evaluation = y[evaluation_indices]
            st.session_state.image_cnn_model = model
            st.session_state.image_label_encoder = label_encoder
            st.session_state.image_model_kind = "cnn"
            st.session_state.image_trained_model_name = name
            st.session_state.image_size = size
            st.session_state.pop("image_pipe", None)
            st.caption(
                f"Training stopped after {len(history.history.get('loss', []))} epoch(s). "
                f"Early stopping used: {validation_design}. The reported score uses a separate evaluation set."
            )
            feature_description = [
                "two-dimensional RGB images",
                "learned convolutional edges, textures, and shapes",
            ]
        else:
            X_train = X_vectors[train_indices]
            X_evaluation = X_vectors[evaluation_indices]
            y_train = y[train_indices]
            y_evaluation = y[evaluation_indices]
            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("model", image_model(name, parameters)),
            ])
            with st.spinner(f"Training {name}..."):
                pipe.fit(X_train, y_train)
                pred = pipe.predict(X_evaluation)
            st.session_state.image_pipe = pipe
            st.session_state.image_model_kind = "classical"
            st.session_state.image_trained_model_name = name
            st.session_state.image_size = size
            st.session_state.pop("image_cnn_model", None)
            st.session_state.pop("image_label_encoder", None)
            feature_description = ["flattened RGB pixels and color summaries"]

        image_metrics = {
            "Accuracy": accuracy_score(y_evaluation, pred),
            "F1-score": f1_score(
                y_evaluation, pred, average="weighted", zero_division=0
            ),
        }
        result = {
            "problem": "classification",
            "context": "computer vision",
            "model_name": name,
            "parameters": parameters,
            "features": feature_description,
            "target": "image class",
            "y_test": y_evaluation,
            "pred": pred,
            "metrics": image_metrics,
            "classes": sorted(np.unique(y).tolist()),
            "dataset_name": getattr(upload, "name", "Uploaded image folders"),
            "evaluation_method": plan["method"],
            "evaluation_label": plan["evaluation_label"],
            "evaluation_design": plan["design"],
            "training_count": len(train_indices),
            "evaluation_count": len(evaluation_indices),
            "known_group_overlap": len(
                set(dataset["group_keys"][train_indices])
                & set(dataset["group_keys"][evaluation_indices])
            ),
        }
        st.session_state.image_result = result
        image_result_display(result, result["classes"])

    st.divider()
    new = st.file_uploader(
        "Classify a new image", type=["png", "jpg", "jpeg"], key="img_new"
    )
    if new:
        img = Image.open(new).convert("RGB")
        st.image(img, width=350)
        model_kind = st.session_state.get("image_model_kind")
        trained_name = st.session_state.get("image_trained_model_name")
        if not model_kind:
            return st.info("Train the classifier first.")
        st.caption(f"Using the trained model: {trained_name}")

        if model_kind == "cnn":
            model = st.session_state.get("image_cnn_model")
            label_encoder = st.session_state.get("image_label_encoder")
            if model is None or label_encoder is None:
                return st.info("Train the convolutional neural network first.")
            arr = image_array(img, st.session_state.image_size).reshape(
                1, st.session_state.image_size, st.session_state.image_size, 3
            )
            probabilities = model.predict(arr, verbose=0)[0]
            predicted_code = int(np.argmax(probabilities))
            predicted_class = label_encoder.inverse_transform([predicted_code])[0]
            probability_table = pd.DataFrame({
                "Class": label_encoder.classes_,
                "Probability": probabilities,
            }).sort_values("Probability", ascending=False)
        else:
            pipe = st.session_state.get("image_pipe")
            if pipe is None:
                return st.info("Train the classifier first.")
            vector = image_vector(img, st.session_state.image_size).reshape(1, -1)
            predicted_class = pipe.predict(vector)[0]
            probability_table = None
            if hasattr(pipe, "predict_proba"):
                probability_table = pd.DataFrame({
                    "Class": pipe.classes_,
                    "Probability": pipe.predict_proba(vector)[0],
                }).sort_values("Probability", ascending=False)

        st.success(f"Predicted class: {predicted_class}")
        if probability_table is not None:
            st.dataframe(probability_table, use_container_width=True)


# -----------------------------------------------------------------------------
# Guided weekly teaching and learning layer
# -----------------------------------------------------------------------------
def week_status(project, week):
    entry = project.get("weeks", {}).get(week, {})
    completed = entry.get("completed_steps", [])
    return len(set(completed)), 5


def mark_guided_step(week, step):
    project = ensure_project_state()
    current = project.setdefault("current_plans", {}).setdefault(week, {})
    steps = set(current.get("completed_steps", []))
    steps.add(step)
    current["completed_steps"] = sorted(steps)


def render_week_header(week):
    project = ensure_project_state()
    lab = WEEKLY_LABS[week]
    completed, total = week_status(project, week)
    st.title(f"{week}: {lab['title']}")
    st.markdown(
        f'<div class="path-card"><strong>Today’s goal</strong><br>{lab["learn"]}</div>',
        unsafe_allow_html=True,
    )
    saved = project.get("weeks", {}).get(week)
    labels = []
    current_steps = project.get("current_plans", {}).get(week, {}).get("completed_steps", [])
    for i, label in enumerate(["Plan", "Learn", "Analyze", "Explain", "Save"], 1):
        done = label in current_steps or (saved is not None and label == "Save")
        labels.append(f"{'✓' if done else '○'} {i}. {label}")
    st.markdown(" ".join(f'<span class="progress-pill">{label}</span>' for label in labels), unsafe_allow_html=True)


def instructor_setup_page(week):
    df = get_df()
    lab = WEEKLY_LABS[week]
    brief = get_lab_brief(week, df)
    st.title("Instructor Setup")
    st.markdown(
        '<div class="simple-note"><strong>Use this page before class.</strong><br>'
        'Set the question, variables, instructions, and amount of student choice. The analytical tools remain unchanged.</div>',
        unsafe_allow_html=True,
    )
    st.subheader(f"{week}: {lab['title']}")
    targets = suitable_targets(week, df)
    with st.form(f"brief_form_{week}"):
        question = st.text_area(
            "Question we will answer today",
            value=brief.get("research_question", ""),
            placeholder="Example: How well can the selected predictors explain or predict the chosen target?",
        )
        choice_mode = st.radio(
            "How much freedom should students have?",
            ["Use the instructor's exact variables", "Choose from instructor-approved variables", "Choose any suitable variables"],
            index=["Use the instructor's exact variables", "Choose from instructor-approved variables", "Choose any suitable variables"].index(brief.get("choice_mode", "Use the instructor's exact variables")),
        )
        selected_targets = st.multiselect(
            "Assigned or approved target variables",
            targets,
            default=[c for c in brief.get("targets", []) if c in targets],
            help="For an exact class activity, select one target. For guided choice, select the targets students may choose from.",
        )
        predictor_options = [c for c in df.columns if c not in selected_targets]
        selected_predictors = st.multiselect(
            "Assigned or approved predictors",
            predictor_options,
            default=[c for c in brief.get("predictors", []) if c in predictor_options],
        )
        class_example = st.text_area(
            "Simple class example or opening question",
            value=brief.get("class_example", ""),
            placeholder="Example: Before fitting the model, do you expect the relationship to be positive or negative?",
        )
        instructions = st.text_area("Today's task", value=brief.get("instructions", lab["assignment"]))
        required_outputs = st.multiselect(
            "What students must report",
            sorted(set(lab["required"] + brief.get("required_outputs", []))),
            default=[x for x in brief.get("required_outputs", lab["required"]) if x in sorted(set(lab["required"] + brief.get("required_outputs", [])))],
        )
        duration = st.slider("Planned lab time in minutes", 20, 120, int(brief.get("duration", 60)), 5)
        submitted = st.form_submit_button("Save today's lab brief", type="primary")
    if submitted:
        st.session_state.lab_briefs[week] = {
            "week": week,
            "research_question": question.strip(),
            "targets": selected_targets,
            "predictors": selected_predictors,
            "choice_mode": choice_mode,
            "class_example": class_example.strip(),
            "instructions": instructions.strip(),
            "required_outputs": required_outputs,
            "duration": duration,
        }
        st.success("Today's lab brief was saved.")
        st.rerun()
    current = get_lab_brief(week, df)
    st.subheader("Student preview")
    st.markdown(f"**Question:** {current.get('research_question') or 'Students will write the question.'}")
    st.markdown(f"**Target choices:** {', '.join(map(humanize, current.get('targets', []))) or 'Open choice'}")
    st.markdown(f"**Predictor choices:** {', '.join(map(humanize, current.get('predictors', []))) or 'Open choice'}")
    st.markdown(f"**Task:** {current.get('instructions', '')}")
    st.download_button(
        "Download this class brief",
        json.dumps(json_safe(current), indent=2),
        f"MATH490_{week.replace(' ', '_')}_lab_brief.json",
        "application/json",
        key=f"brief_download_{week}",
    )
    imported = st.file_uploader("Import a saved class brief", type=["json"], key=f"brief_upload_{week}")
    if imported is not None and st.button("Import this brief", key=f"brief_import_{week}"):
        try:
            loaded = json.loads(imported.getvalue().decode("utf-8"))
            if not isinstance(loaded, dict): raise ValueError("Invalid brief")
            loaded["week"] = week
            st.session_state.lab_briefs[week] = loaded
            st.success("Brief imported.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not import the brief: {exc}")


def guided_plan_step(week):
    df = get_df()
    project = ensure_project_state()
    brief = get_lab_brief(week, df)
    lab = WEEKLY_LABS[week]
    st.subheader("1. Plan: What are we doing today?")
    st.markdown(
        '<div class="simple-note"><strong>Start with the question.</strong><br>'
        'Do not click a model until you can name the target and the information being used.</div>',
        unsafe_allow_html=True,
    )
    if brief.get("class_example"):
        st.info("Class opening question: " + brief["class_example"])
    choice_mode = brief.get("choice_mode", "Use the instructor's exact variables")
    target_options = suitable_targets(week, df)
    approved_targets = [c for c in brief.get("targets", []) if c in target_options]
    if choice_mode == "Use the instructor's exact variables" and approved_targets:
        available_targets = approved_targets[:1]
        target_disabled = True
    elif choice_mode == "Choose from instructor-approved variables" and approved_targets:
        available_targets = approved_targets
        target_disabled = False
    else:
        available_targets = target_options
        target_disabled = False
    if not available_targets:
        available_targets = list(df.columns)
    saved_plan = project.setdefault("current_plans", {}).setdefault(week, {})
    default_target = saved_plan.get("target") if saved_plan.get("target") in available_targets else available_targets[0]
    target = st.selectbox(
        "What are we trying to explain or predict?",
        available_targets,
        index=available_targets.index(default_target),
        format_func=humanize,
        disabled=target_disabled,
        key=f"guided_target_{week}",
    )
    all_predictors = [c for c in df.columns if c != target]
    approved_predictors = [c for c in brief.get("predictors", []) if c in all_predictors]
    if choice_mode == "Use the instructor's exact variables":
        predictor_options = approved_predictors or all_predictors
        default_predictors = approved_predictors
        predictor_disabled = bool(approved_predictors)
    elif choice_mode == "Choose from instructor-approved variables" and approved_predictors:
        predictor_options = approved_predictors
        default_predictors = [c for c in saved_plan.get("predictors", []) if c in predictor_options] or approved_predictors[: min(3, len(approved_predictors))]
        predictor_disabled = False
    else:
        predictor_options = all_predictors
        default_predictors = [c for c in saved_plan.get("predictors", []) if c in predictor_options] or default_predictors_for_target(df, target, 4)
        predictor_disabled = False
    predictors = st.multiselect(
        "Which information will we use?",
        predictor_options,
        default=default_predictors,
        format_func=humanize,
        disabled=predictor_disabled,
        key=f"guided_predictors_{week}",
    )
    class_question = brief.get("research_question", "").strip()
    use_class_question = st.checkbox(
        "Use the class research question",
        value=True,
        key=f"guided_use_question_{week}",
        disabled=choice_mode == "Use the instructor's exact variables" and bool(class_question),
    )
    question = st.text_area(
        "Research question",
        value=class_question if use_class_question else saved_plan.get("question", class_question),
        disabled=use_class_question and bool(class_question),
        key=f"guided_question_{week}",
    )
    st.markdown("**Today’s task:** " + (brief.get("instructions") or lab["assignment"]))
    if brief.get("required_outputs"):
        st.markdown("**You must report:** " + "; ".join(brief["required_outputs"]))
    if st.button("Confirm today's plan", type="primary", key=f"guided_confirm_{week}"):
        final_question = class_question if use_class_question and class_question else question.strip()
        if not final_question:
            st.warning("Write the question before continuing.")
        elif week not in ["Week 1", "Week 8", "Week 15", "Week 16"] and not target:
            st.warning("Choose a target.")
        else:
            plan = {
                "question": final_question,
                "target": target,
                "predictors": predictors,
                "dataset_name": st.session_state.get("dataset_name", ""),
                "completed_steps": sorted(set(saved_plan.get("completed_steps", [])) | {"Plan"}),
            }
            project["current_plans"][week] = plan
            if week == "Week 2":
                project["main_question"] = final_question
                if target in num_cols(df): project["continuous_target"] = target
                elif target in class_targets(df): project["classification_target"] = target
                project["candidate_predictors"] = predictors
            apply_plan_to_tool(week, plan, context="legacy_guided", force=True)
            st.success("Plan confirmed. Open Step 2: Learn.")
            st.rerun()


def guided_learn_step(week):
    lab = WEEKLY_LABS[week]
    st.subheader("2. Learn: The idea in simple words")
    st.markdown(f'<div class="simple-note"><strong>Key idea</strong><br>{lab["key_idea"]}</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.markdown("### Words to know")
        for term, meaning in lab["terms"].items():
            st.markdown(f"**{term}:** {meaning}")
    with c2:
        st.markdown("### What we will do")
        for i, item in enumerate(lab["steps"], 1):
            st.markdown(f"**{i}.** {item}")
    with st.expander("Why this matters", expanded=False):
        st.write(lab["learn"])
        st.write("The goal is to understand the question, use the correct method, and explain the result in ordinary language.")
    st.warning("Common mistake: " + lab["mistake"])
    st.markdown("### Quick check")
    check = st.radio(
        "Which sentence is always safest?",
        [
            "The model proves that the predictors cause the target.",
            "The result describes the fitted data or predictions, but it does not automatically prove causation.",
            "The largest number is always the best result.",
        ],
        index=None,
        key=f"guided_check_{week}",
    )
    if check:
        if check.startswith("The result describes"):
            st.success("Correct. Models describe patterns and predictions. Causal claims require stronger study designs.")
            mark_guided_step(week, "Learn")
        else:
            st.error("Try again. A model result does not automatically prove causation, and the meaning of a metric matters.")


def render_tool_page(page_name):
    pages = {
        "Home and Quick Start": page_home,
        "Data and Research Questions": page_data,
        "Probability and Uncertainty": page_probability,
        "Visualization and Descriptive Statistics": page_visualization,
        "Relationships and Association": page_association,
        "Simple and Multiple Regression": page_linear_regression,
        "Machine Learning for Regression": page_ml_regression,
        "Logistic Regression": page_logistic,
        "Machine Learning for Classification": page_ml_classification,
        "Predictor Selection": page_predictor_selection,
        "Model Explanations": page_explanations,
        "Model Evaluation and Comparison": page_comparison,
        "Cross-Validation and Model Selection": page_cv,
        "Bootstrap and Uncertainty": page_bootstrap,
        "Time Series Forecasting": page_forecasting,
        "Computer Vision": page_computer_vision,
        "Three-Slide Mini-Report Builder": page_report,
    }
    if page_name in pages:
        pages[page_name]()


def guided_analyze_step(week):
    project = ensure_project_state()
    lab = WEEKLY_LABS[week]
    plan = project.get("current_plans", {}).get(week, {})
    st.subheader("3. Analyze: Follow the steps")
    if not plan.get("question") and week not in ["Week 1", "Week 8", "Week 16"]:
        st.warning("Complete Step 1: Plan before running the analysis.")
        return
    if plan:
        st.markdown(
            f'<div class="success-note"><strong>Question:</strong> {plan.get("question", "")}<br>'
            f'<strong>Target:</strong> {humanize(plan.get("target", ""))}<br>'
            f'<strong>Predictors:</strong> {", ".join(map(humanize, plan.get("predictors", []))) or "Not required for this activity"}</div>',
            unsafe_allow_html=True,
        )
        apply_plan_to_tool(week, plan, context="legacy_guided")
    st.info("Use the preselected values for today. Change them only when your instructor asks you to explore.")
    if lab.get("tool"):
        render_tool_page(lab["tool"])
        if st.button("I completed today's analysis", key=f"guided_analysis_done_{week}"):
            mark_guided_step(week, "Analyze")
            st.success("Analysis marked complete. Open Step 4: Explain.")
    else:
        st.markdown("### Review checklist")
        for item in lab["steps"]:
            st.checkbox(item, key=f"review_{week}_{item}")
        if st.button("I completed the review", key=f"guided_review_done_{week}"):
            mark_guided_step(week, "Analyze")
            st.success("Review marked complete.")


def guided_explain_step(week):
    project = ensure_project_state()
    plan = project.get("current_plans", {}).get(week, {})
    lab = WEEKLY_LABS[week]
    result = week_result(week)
    st.subheader("4. Explain: What does the result mean?")
    st.markdown(
        '<div class="simple-note"><strong>Use this sentence pattern:</strong><br>'
        'We used [method] to answer [question]. The main result was [number or pattern]. This means [plain-language meaning]. One limitation is [limitation].</div>',
        unsafe_allow_html=True,
    )
    notes = simple_result_notes(result)
    if notes:
        st.markdown("### Guided interpretation")
        for note in notes:
            st.markdown("- " + note)
    elif week == "Week 12" and isinstance(result, dict):
        scores = np.asarray(result.get("scores", []), dtype=float)
        if len(scores):
            st.markdown(f"- The average {result.get('metric', 'score').lower()} is **{scores.mean():.3f}**.")
            st.markdown(f"- The score changes by about **{scores.std(ddof=1) if len(scores)>1 else 0:.3f}** across resamples.")
    elif week == "Week 13" and isinstance(result, dict):
        values = np.asarray(result.get("values", []), dtype=float)
        if len(values):
            lo, hi = np.percentile(values, [2.5, 97.5])
            st.markdown(f"- The original estimate is **{result.get('original', float('nan')):.3f}**.")
            st.markdown(f"- The bootstrap 95% interval is approximately **[{lo:.3f}, {hi:.3f}]**.")
    else:
        st.info("Run the analysis first. Then return here for a guided explanation.")
    result_text = st.text_area(
        "My key result",
        value=project.get("current_plans", {}).get(week, {}).get("key_result", ""),
        placeholder="Example: The held-out mean absolute error was 6.28 points.",
        key=f"guided_result_text_{week}",
    )
    interpretation = st.text_area(
        "My explanation in ordinary language",
        value=project.get("current_plans", {}).get(week, {}).get("interpretation", ""),
        placeholder="Explain what the result says about the research question.",
        key=f"guided_interpretation_{week}",
    )
    limitation = st.text_area(
        "One limitation",
        value=project.get("current_plans", {}).get(week, {}).get("limitation", ""),
        placeholder="Example: The model uses one dataset and does not establish cause and effect.",
        key=f"guided_limitation_{week}",
    )
    if st.button("Save my explanation", type="primary", key=f"guided_explain_save_{week}"):
        if not result_text.strip() or not interpretation.strip() or not limitation.strip():
            st.warning("Complete the result, explanation, and limitation.")
        else:
            current = project.setdefault("current_plans", {}).setdefault(week, {})
            current.update({"key_result": result_text.strip(), "interpretation": interpretation.strip(), "limitation": limitation.strip()})
            mark_guided_step(week, "Explain")
            st.success("Explanation saved. Open Step 5: Save.")


def weekly_report_text(week, entry):
    """Create preparation notes; students still design and present the final slides."""
    lab = WEEKLY_LABS[week]
    predictors = ", ".join(map(humanize, entry.get("predictors", []))) or "Not applicable"
    target = humanize(entry.get("target", "")) or "Not applicable"
    return f"""# MATH 490 Slide-Preparation Notes

**These are analysis notes, not a finished presentation. The student must create and present the three slides.**

**Module:** {week}: {lab['title']}

**Weekly presentation assignment:** {lab['assignment']}

## Student's Independent Analysis
**Research question:** {entry.get('question', '')}

**Dataset:** {entry.get('dataset_name', '')}

**Target or outcome:** {target}

**Predictors or variables examined:** {predictors}

**Method used:** {entry.get('method', '')}

**Key result:** {entry.get('key_result', '')}

**Interpretation:** {entry.get('interpretation', '')}

**Limitation:** {entry.get('limitation', '')}

**Next step:** {entry.get('next_step', '')}

---
## Use These Notes to Prepare Your Own Three Slides

### Slide 1 — Question and Data
State your research question, identify the dataset and variables, and include one suitable introductory visual.

### Slide 2 — Method and Evidence
Name the method, report the key numerical result or pattern, and include one result figure or table.

### Slide 3 — Meaning and Limitation
Explain what the result means, state one limitation, and identify one appropriate next step.

**Communication reminder:** Explain what the analysis supports without claiming causation unless the study design justifies it.
"""

def complete_notebook_markdown():
    """Export student assignment work separately from instructor-led class work."""
    project = ensure_project_state()
    parts = [
        "# MATH 490 Applied AI Lab Studio Notebook",
        "",
        f"**Student:** {project.get('student_name') or 'Not entered'}",
        "",
        "This notebook separates the student's independent assignment analysis from the instructor-led class analysis and practical record.",
        "",
        "# Part A — Student Independent Assignment Analysis",
        "",
        "Use this section as evidence when preparing the required weekly three-slide presentations.",
        "",
    ]
    for week, lab in WEEKLY_LABS.items():
        entry = project.get("weeks", {}).get(week)
        parts.extend([f"## {week}: {lab['title']}", "", f"**Weekly presentation assignment:** {lab['assignment']}", ""])
        if not entry:
            parts.extend(["No independent assignment analysis has been saved for this week.", ""])
            continue
        predictors = ", ".join(map(humanize, entry.get("predictors", []))) or "Not applicable"
        target = humanize(entry.get("target", "")) or "Not applicable"
        parts.extend([
            f"**Student dataset:** {entry.get('dataset_name', '')}",
            "",
            f"**Student research question:** {entry.get('question', '')}",
            "",
            f"**Target or outcome:** {target}",
            "",
            f"**Predictors or variables examined:** {predictors}",
            "",
            f"**Method used:** {entry.get('method', '')}",
            "",
            f"**Key result:** {entry.get('key_result', '')}",
            "",
            f"**Student interpretation:** {entry.get('interpretation', '')}",
            "",
            f"**Limitation:** {entry.get('limitation', '')}",
            "",
            f"**Next step:** {entry.get('next_step', '')}",
            "",
        ])

    parts.extend([
        "# Part B — Instructor-Led Class Analysis and Practical Record",
        "",
        "This section records the class question, instructor-selected starting variables, practical activity, the student's class reflection, and Wrap-Up performance. It is separate from the student's independent assignment analysis above.",
        "",
    ])
    briefs = st.session_state.get("lab_briefs", {})
    any_class_record = False
    for week, lab in WEEKLY_LABS.items():
        brief = briefs.get(week, {})
        plan = project.get("practice_plans", {}).get(week, {})
        progress = project.get("practical_progress", {}).get(week, {})
        wrap = project.get("wrap_up_progress", {}).get(week, {})
        if not any([brief, plan, progress, wrap]):
            continue
        any_class_record = True
        targets = brief.get("targets", []) or ([plan.get("target")] if plan.get("target") else [])
        predictors = brief.get("predictors", []) or plan.get("predictors", [])
        result_notes = progress.get("result_notes", [])
        parts.extend([
            f"## {week}: {lab['title']}",
            "",
            f"**Instructor class dataset:** {brief.get('dataset_name') or plan.get('dataset_name', '')}",
            "",
            f"**Instructor class research question:** {brief.get('research_question') or plan.get('question', '')}",
            "",
            f"**Instructor-selected target or outcome:** {', '.join(map(humanize, targets)) or 'Not applicable'}",
            "",
            f"**Instructor-selected starting predictors or variables:** {', '.join(map(humanize, predictors)) or 'Not applicable'}",
            "",
            f"**Class practical task:** {brief.get('instructions', lab['assignment'])}",
            "",
            f"**Required class outputs:** {'; '.join(brief.get('required_outputs', lab['required']))}",
            "",
            f"**Student practical reflection:** {progress.get('reflection', 'Not yet recorded')}",
            "",
        ])
        if result_notes:
            parts.extend(["**Class result-reading notes:**", ""])
            parts.extend([f"- {note}" for note in result_notes])
            parts.append("")
        if wrap:
            parts.extend([
                f"**Wrap-Up latest score:** {wrap.get('score', 0)}/{wrap.get('total', 0)}",
                "",
                f"**Wrap-Up best score:** {wrap.get('best_score', 0)}/{wrap.get('total', 0)}",
                "",
            ])
    if not any_class_record:
        parts.append("No instructor-led class record is available in this session.")

    parts.extend([
        "# Part C — Independent Exam Practice Summary",
        "",
        "This section records topic-level practice scores. Detailed question reports can also be downloaded from Exam Practice after each attempt.",
        "",
    ])
    practice_records = project.get("exam_practice", {})
    if not practice_records:
        parts.append("No independent exam-practice attempt has been completed.")
    else:
        for topic, record in sorted(practice_records.items()):
            parts.extend([
                f"## {topic}",
                "",
                f"**Attempts:** {record.get('attempts', 0)}",
                "",
                f"**Latest score:** {record.get('latest_score', 0)}/{record.get('latest_total', 0)}",
                "",
                f"**Best score:** {record.get('best_score', 0)}/{record.get('best_total', 0)}",
                "",
            ])
    return "\n".join(parts).strip() + "\n"

def guided_save_step(week):
    project = ensure_project_state()
    lab = WEEKLY_LABS[week]
    plan = project.get("current_plans", {}).get(week, {})
    result = week_result(week)
    st.subheader("5. Save: Build the weekly mini-report")
    if not plan.get("question") and week not in ["Week 1", "Week 8", "Week 16"]:
        st.warning("Complete the earlier steps first.")
        return
    method_default = result.get("model_name", "") if isinstance(result, dict) else lab["title"]
    method = st.text_input("Method used", value=plan.get("method", method_default), key=f"guided_method_{week}")
    key_result = st.text_area("Key result", value=plan.get("key_result", ""), key=f"guided_save_result_{week}")
    interpretation = st.text_area("Interpretation", value=plan.get("interpretation", ""), key=f"guided_save_interpretation_{week}")
    limitation = st.text_area("Limitation", value=plan.get("limitation", ""), key=f"guided_save_limitation_{week}")
    next_step = st.text_area("Next step", value=plan.get("next_step", ""), key=f"guided_next_{week}")
    if st.button("Save this week to My Lab Notebook", type="primary", key=f"guided_week_save_{week}"):
        required = [key_result.strip(), interpretation.strip(), limitation.strip(), next_step.strip()]
        if not all(required):
            st.warning("Complete the key result, interpretation, limitation, and next step.")
        else:
            entry = {
                "week": week,
                "title": lab["title"],
                "question": plan.get("question", ""),
                "dataset_name": plan.get("dataset_name", st.session_state.get("dataset_name", "")),
                "target": plan.get("target", ""),
                "predictors": plan.get("predictors", []),
                "method": method.strip(),
                "key_result": key_result.strip(),
                "interpretation": interpretation.strip(),
                "limitation": limitation.strip(),
                "next_step": next_step.strip(),
                "completed_steps": ["Plan", "Learn", "Analyze", "Explain", "Save"],
            }
            project.setdefault("weeks", {})[week] = entry
            project.setdefault("current_plans", {})[week] = {**plan, **entry}
            st.success("Saved to My Lab Notebook.")
            st.rerun()
    entry = project.get("weeks", {}).get(week)
    if entry:
        st.markdown('<div class="success-note"><strong>This week is saved.</strong><br>You can edit it, download the report, or continue next week.</div>', unsafe_allow_html=True)
        report = weekly_report_text(week, entry)
        st.download_button(
            "Download this week's three-slide outline",
            report,
            f"MATH490_{week.replace(' ', '_')}_three_slide_report.md",
            "text/markdown",
            key=f"guided_week_report_{week}",
        )
        with st.expander("Preview the three-slide outline", expanded=False):
            st.markdown(report)


def guided_lab_page(week):
    render_week_header(week)
    step = st.radio(
        "Today's steps",
        ["1. Plan", "2. Learn", "3. Analyze", "4. Explain", "5. Save"],
        horizontal=True,
        key=f"guided_step_{week}",
    )
    st.divider()
    if step == "1. Plan": guided_plan_step(week)
    elif step == "2. Learn": guided_learn_step(week)
    elif step == "3. Analyze": guided_analyze_step(week)
    elif step == "4. Explain": guided_explain_step(week)
    else: guided_save_step(week)


def notebook_page():
    project = ensure_project_state()
    st.title("📘 My Lab Notebook")
    st.markdown(
        '<div class="simple-note"><strong>Your work grows each week.</strong><br>'
        'Open any saved week to review the question, variables, result, explanation, and next step.</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Saved weeks", len(project.get("weeks", {})))
    c2.metric("Assignment dataset", st.session_state.get("dataset_name", "Not selected"))
    c3.metric("Student", project.get("student_name") or "Not entered")
    st.download_button("Download complete notebook", project_json(), "MATH490_lab_notebook.json", "application/json", key="notebook_download")
    for week in WEEKLY_LABS:
        entry = project.get("weeks", {}).get(week)
        title = WEEKLY_LABS[week]["title"]
        status = "Saved" if entry else "Not saved"
        with st.expander(f"{'✅' if entry else '○'} {week}: {title} — {status}", expanded=False):
            if not entry:
                st.write("Complete the guided lab and save the weekly entry.")
                continue
            st.markdown(f"**Question:** {entry.get('question', '')}")
            st.markdown(f"**Target:** {humanize(entry.get('target', ''))}")
            st.markdown(f"**Predictors:** {', '.join(map(humanize, entry.get('predictors', []))) or 'Not applicable'}")
            st.markdown(f"**Method:** {entry.get('method', '')}")
            st.markdown(f"**Key result:** {entry.get('key_result', '')}")
            st.markdown(f"**Interpretation:** {entry.get('interpretation', '')}")
            st.markdown(f"**Limitation:** {entry.get('limitation', '')}")
            st.markdown(f"**Next step:** {entry.get('next_step', '')}")
            st.download_button(
                "Download this week's report",
                weekly_report_text(week, entry),
                f"MATH490_{week.replace(' ', '_')}_report.md",
                "text/markdown",
                key=f"notebook_report_{week}",
            )


# -----------------------------------------------------------------------------
# Simplified teaching spaces
# -----------------------------------------------------------------------------
WEEK_OUTPUT_GUIDES = {
    "Week 1": [
        ("Theoretical probability", "The probability expected from the rules of the chance experiment."),
        ("Experimental probability", "The observed proportion after the experiment is repeated."),
        ("Number of trials", "Larger samples usually show less random fluctuation in the observed proportion."),
    ],
    "Week 2": [
        ("Research question", "A clear, answerable question that names the outcome and relevant variables."),
        ("Target", "The outcome to explain or predict."),
        ("Predictors", "Information that may help answer the question without revealing the answer directly."),
    ],
    "Week 3": [
        ("Center", "A typical value, often summarized by the mean or median."),
        ("Spread", "How much the values vary."),
        ("Plots", "Pictures that reveal patterns, unusual values, and group differences."),
    ],
    "Week 4": [
        ("Association value", "The strength of the relationship or group difference."),
        ("Direction", "Whether larger values tend to occur with larger or smaller values."),
        ("Caution", "Association does not automatically mean causation."),
    ],
    "Week 5": [
        ("Slope", "The predicted change in the target for a one-unit increase in the predictor."),
        ("Intercept", "The predicted target when the predictor equals zero."),
        ("Prediction error", "How far predictions are from observed values on unseen rows."),
    ],
    "Week 6": [
        ("Coefficients", "The direction and size of each linear relationship while other predictors are held fixed."),
        ("Held-out error", "How well the model predicts rows it did not use for fitting."),
        ("Added value", "Whether the extra predictors improve prediction enough to be useful."),
    ],
    "Week 7": [
        ("Fair comparison", "Every model must use the same data split, target, predictors, and metric."),
        ("MAE and RMSE", "Smaller values mean smaller prediction errors."),
        ("Recommendation", "Choose complexity only when it produces a useful improvement on unseen data."),
    ],
    "Week 8": [
        ("Method choice", "Match the method to the type of question and variables."),
        ("Metric meaning", "Explain what the number measures rather than memorizing it."),
        ("Limitation", "State what the analysis cannot establish."),
    ],
    "Week 9": [
        ("Probability", "The model's estimated chance of the positive class."),
        ("Threshold", "The cutoff used to turn a probability into a class."),
        ("Confusion matrix", "A count of correct and incorrect class predictions."),
    ],
    "Week 10": [
        ("Accuracy", "The fraction of all held-out predictions that are correct."),
        ("Precision and recall", "Precision checks predicted positives; recall checks actual positives."),
        ("F1-score", "A balance between precision and recall."),
    ],
    "Week 11": [
        ("Primary metric", "The score chosen before comparing models."),
        ("Prediction model", "The model recommended for held-out performance."),
        ("Explanation model", "A model that may be preferred because it is easier to understand."),
    ],
    "Week 12": [
        ("Fold", "One part of the data used for validation while the others train the model."),
        ("Average score", "The model's typical performance across folds."),
        ("Stability", "How much performance and ranking change across folds."),
    ],
    "Week 13": [
        ("Bootstrap sample", "A new sample formed by drawing rows with replacement."),
        ("Bootstrap distribution", "The many estimates produced by repeated resampling."),
        ("Interval", "A range showing plausible uncertainty around the estimate."),
    ],
    "Week 14": [
        ("Time order", "Training must use information available before the forecasted time."),
        ("Lag", "A previous value used as a predictor."),
        ("Baseline", "A simple forecast that a more complex model should try to beat."),
    ],
    "Week 15": [
        ("Pixels", "Numbers that describe image brightness and color."),
        ("Class label", "The category the model is trained to recognize."),
        ("Unseen images", "Images not used for training and needed for honest evaluation."),
    ],
    'Week 16': [('Review scope',
  'Connect classification, model evaluation, cross-validation, uncertainty, forecasting, neural networks, and computer '
  'vision.'),
 ('Method choice', 'Match the model to the target type, data structure, and intended decision.'),
 ('Evaluation', 'Use held-out evidence and the metric that reflects the important error.'),
 ('Responsible interpretation', 'State uncertainty, limitations, fairness concerns, and the need for human oversight.')],
}

QUICK_CHECKS = {
    "Week 1": ("What usually happens as the number of independent trials increases?", ["Experimental probability often moves closer to theoretical probability", "The sample space disappears", "The next result becomes guaranteed"], "Experimental probability often moves closer to theoretical probability"),
    "Week 2": ("Which question is clearest?", ["Can selected variables help explain or predict the chosen outcome?", "What is the biggest number?", "Can the model prove causation?"], "Can selected variables help explain or predict the chosen outcome?"),
    "Week 3": ("What should happen before modeling?", ["Look at summaries and plots", "Choose the most complex model", "Delete every unusual value"], "Look at summaries and plots"),
    "Week 4": ("What does association prove?", ["A relationship in the data, not automatic causation", "That one variable causes the other", "That the model will forecast the future"], "A relationship in the data, not automatic causation"),
    "Week 5": ("What does the slope describe?", ["Predicted target change for a one-unit predictor increase", "The number of rows", "The classification threshold"], "Predicted target change for a one-unit predictor increase"),
    "Week 6": ("Does adding predictors always improve unseen prediction?", ["No", "Yes", "Only when the training error becomes zero"], "No"),
    "Week 7": ("Which regression model is better for prediction?", ["The one with lower held-out error", "The one with more settings", "The one with the best training score only"], "The one with lower held-out error"),
    "Week 8": ("What is the main goal of review?", ["Explain what methods and outputs mean", "Memorize every button", "Use every model at once"], "Explain what methods and outputs mean"),
    "Week 9": ("What turns a predicted probability into a class?", ["A threshold", "A residual", "A slope"], "A threshold"),
    "Week 10": ("What does recall focus on?", ["Actual positive cases correctly found", "Only predicted negatives", "Average regression error"], "Actual positive cases correctly found"),
    "Week 11": ("How should models be compared fairly?", ["Use the same rows, predictors, split, and metric", "Use different test sets", "Choose the largest training score"], "Use the same rows, predictors, split, and metric"),
    "Week 12": ("Why use cross-validation?", ["To see whether performance is stable across splits", "To prove causation", "To remove the target"], "To see whether performance is stable across splits"),
    "Week 13": ("What does bootstrap show?", ["How an estimate changes across resampled datasets", "Only the training accuracy", "The exact future"], "How an estimate changes across resampled datasets"),
    "Week 14": ("Why must time order be preserved?", ["Future observations must not leak into training", "To make the graph colorful", "To increase the number of classes"], "Future observations must not leak into training"),
    "Week 15": ("Where should final image accuracy be checked?", ["On unseen images", "Only on training images", "On one hand-picked image"], "On unseen images"),
    'Week 16': ('What is the main goal of the final review?',
 ['Connect methods, evaluation, uncertainty, and limitations into defensible decisions',
  'Memorize every button without interpreting results',
  'Choose the most complex model for every problem'],
 'Connect methods, evaluation, uncertainty, and limitations into defensible decisions'),
}


WEEK_WRAP_UP_QUESTIONS = {
    "Week 1": [
        {"question": "Which statement best describes probability?", "options": ["A number from 0 to 1 describing likelihood", "A guarantee of the next outcome", "The number of columns in a dataset"], "answer": "A number from 0 to 1 describing likelihood", "explanation": "Probability measures likelihood. It does not guarantee the result of one trial."},
        {"question": "What is experimental probability?", "options": ["The observed proportion after repeated trials", "A fixed rule that never uses data", "The largest outcome in the sample space"], "answer": "The observed proportion after repeated trials", "explanation": "Experimental probability is calculated from what actually occurred in the simulation or experiment."},
        {"question": "What usually happens when the number of independent trials becomes large?", "options": ["Experimental probability often moves closer to theoretical probability", "Every outcome becomes equally likely", "The sample space disappears"], "answer": "Experimental probability often moves closer to theoretical probability", "explanation": "More trials usually reduce random fluctuation in the observed proportion."},
        {"question": "What is a sample space?", "options": ["The complete set of possible outcomes", "Only the most likely outcome", "The final probability estimate"], "answer": "The complete set of possible outcomes", "explanation": "The sample space lists every possible outcome of the experiment."},
    ],
    "Week 2": [
        {"question": "What makes a research question useful for data analysis?", "options": ["It clearly identifies what will be studied", "It asks the model to prove causation", "It avoids naming any variables"], "answer": "It clearly identifies what will be studied", "explanation": "A useful question is specific, answerable with the available data, and connected to measurable variables."},
        {"question": "What is the target?", "options": ["The outcome to explain or predict", "Every column except the identifier", "The name of the dataset"], "answer": "The outcome to explain or predict", "explanation": "The target is the main outcome of interest."},
        {"question": "What is a predictor?", "options": ["Information used to help explain or predict the target", "A variable that must contain the answer", "The final model score"], "answer": "Information used to help explain or predict the target", "explanation": "Predictors provide information that may help answer the research question."},
        {"question": "What is target leakage?", "options": ["Using information that reveals the answer or would not be available at prediction time", "Having a numerical target", "Using more than one predictor"], "answer": "Using information that reveals the answer or would not be available at prediction time", "explanation": "Leakage makes performance look unrealistically good because the model has access to information it should not have."},
    ],
    "Week 3": [
        {"question": "What should normally happen before fitting a model?", "options": ["Inspect summaries and visualizations", "Choose the most complex model", "Remove every unusual value"], "answer": "Inspect summaries and visualizations", "explanation": "Exploration helps reveal distributions, missing values, unusual observations, and possible relationships."},
        {"question": "Which plot is useful for viewing the distribution of one numerical variable?", "options": ["Histogram", "Confusion matrix", "ROC curve"], "answer": "Histogram", "explanation": "A histogram shows how numerical values are distributed across intervals."},
        {"question": "What does spread describe?", "options": ["How much the values vary", "The name of the target", "Whether a model is causal"], "answer": "How much the values vary", "explanation": "Spread describes how tightly clustered or widely dispersed the values are."},
        {"question": "What should you do with an apparent outlier?", "options": ["Investigate it before deciding what to do", "Always delete it", "Always replace it with the mean"], "answer": "Investigate it before deciding what to do", "explanation": "An unusual value may be an error or a meaningful observation, so it should be checked rather than automatically removed."},
    ],
    "Week 4": [
        {"question": "What does a positive correlation mean?", "options": ["Larger values of one variable tend to occur with larger values of the other", "One variable definitely causes the other", "The variables are categorical"], "answer": "Larger values of one variable tend to occur with larger values of the other", "explanation": "Positive correlation describes a tendency for the variables to increase together."},
        {"question": "What does correlation establish?", "options": ["Association, not automatic causation", "Definite causation", "Perfect future prediction"], "answer": "Association, not automatic causation", "explanation": "Observed association can arise from many mechanisms and does not by itself establish cause and effect."},
        {"question": "When is eta useful?", "options": ["For association between a categorical variable and a numerical variable", "For two image files", "Only for time-series forecasting"], "answer": "For association between a categorical variable and a numerical variable", "explanation": "Eta summarizes how strongly a numerical outcome differs across categories."},
        {"question": "What does partial correlation do?", "options": ["Examines a relationship after accounting for selected controls", "Converts a numerical target into classes", "Guarantees no confounding"], "answer": "Examines a relationship after accounting for selected controls", "explanation": "Partial correlation adjusts for selected variables, although it cannot guarantee that every confounder has been controlled."},
    ],
    "Week 5": [
        {"question": "When is simple linear regression appropriate?", "options": ["A numerical target and one numerical predictor", "A binary target and image pixels", "Two categorical targets"], "answer": "A numerical target and one numerical predictor", "explanation": "Simple linear regression models a numerical outcome using one predictor."},
        {"question": "What does the slope represent?", "options": ["The predicted target change for a one-unit predictor increase", "The number of observations", "The classification cutoff"], "answer": "The predicted target change for a one-unit predictor increase", "explanation": "The slope translates a one-unit predictor change into the model's expected target change."},
        {"question": "What is a residual?", "options": ["Observed value minus predicted value", "Predicted value minus the sample size", "The model intercept"], "answer": "Observed value minus predicted value", "explanation": "A residual is the prediction error for one observation."},
        {"question": "Which model has better held-out MAE?", "options": ["The model with the smaller MAE", "The model with the larger MAE", "The model with more coefficients"], "answer": "The model with the smaller MAE", "explanation": "A smaller mean absolute error means predictions are closer to observed values on average."},
    ],
    "Week 6": [
        {"question": "What distinguishes multiple regression from simple regression?", "options": ["It uses two or more predictors", "It always uses a categorical target", "It does not make predictions"], "answer": "It uses two or more predictors", "explanation": "Multiple regression estimates a numerical target using several predictors at the same time."},
        {"question": "How is one coefficient interpreted in multiple regression?", "options": ["As the predicted target change while the other included predictors are held fixed", "As proof of causation", "As the number of missing values"], "answer": "As the predicted target change while the other included predictors are held fixed", "explanation": "Each coefficient is conditional on the other predictors included in the model."},
        {"question": "Does adding predictors always improve unseen prediction?", "options": ["No", "Yes", "Only when the training error is zero"], "answer": "No", "explanation": "Extra predictors can add noise, redundancy, or overfitting and may not improve held-out performance."},
        {"question": "What can strong multicollinearity make difficult?", "options": ["Separating the individual contributions of related predictors", "Loading a CSV file", "Creating a binary target"], "answer": "Separating the individual contributions of related predictors", "explanation": "Highly related predictors can make individual coefficient estimates unstable."},
    ],
    "Week 7": [
        {"question": "How should regression models be compared fairly?", "options": ["Use the same data split, predictors, target, and metric", "Give each model a different test set", "Compare only training error"], "answer": "Use the same data split, predictors, target, and metric", "explanation": "A fair comparison changes the model while keeping the evaluation conditions the same."},
        {"question": "Which regression model is preferred for prediction?", "options": ["The model with meaningfully lower held-out error", "The model with the longest name", "The model with the most settings"], "answer": "The model with meaningfully lower held-out error", "explanation": "Prediction recommendations should be based on performance on unseen data, not complexity alone."},
        {"question": "What is overfitting?", "options": ["Learning the training data too closely and performing poorly on new data", "Having too few columns to display", "Using a simple baseline"], "answer": "Learning the training data too closely and performing poorly on new data", "explanation": "An overfit model captures noise or training-specific patterns that do not transfer."},
        {"question": "Why keep a simple baseline?", "options": ["To check whether complexity produces a real improvement", "To guarantee the complex model wins", "To remove the target"], "answer": "To check whether complexity produces a real improvement", "explanation": "A complex model is useful only when it improves meaningfully over a reasonable simpler alternative."},
    ],
    "Week 8": [
        {"question": "What should determine the analytical method?", "options": ["The research question and variable types", "The most colorful chart", "The model with the most controls"], "answer": "The research question and variable types", "explanation": "Method choice should follow the question, outcome type, predictors, and study design."},
        {"question": "Why explain a metric in ordinary language?", "options": ["So readers know what the number measures", "So the score becomes larger", "So causation is established"], "answer": "So readers know what the number measures", "explanation": "A metric is useful only when its practical meaning is understood."},
        {"question": "Why state a limitation?", "options": ["To communicate what the analysis cannot establish", "To invalidate all results", "To increase sample size"], "answer": "To communicate what the analysis cannot establish", "explanation": "Responsible reporting distinguishes supported conclusions from claims the analysis cannot justify."},
        {"question": "Which target type usually suggests classification?", "options": ["A categorical class outcome", "A continuous numerical outcome", "A row identifier"], "answer": "A categorical class outcome", "explanation": "Classification predicts categories or class labels."},
    ],
    "Week 9": [
        {"question": "What kind of outcome is logistic regression designed for in this course?", "options": ["A binary class target", "A continuous numerical target", "An unlabeled image folder"], "answer": "A binary class target", "explanation": "Binary logistic regression estimates the probability of one of two classes."},
        {"question": "What does logistic regression first produce?", "options": ["A predicted probability", "A residual plot only", "A forecast date"], "answer": "A predicted probability", "explanation": "A threshold is then used to turn the probability into a class prediction."},
        {"question": "What does a classification threshold do?", "options": ["Converts a predicted probability into a class", "Calculates a regression slope", "Selects the dataset rows"], "answer": "Converts a predicted probability into a class", "explanation": "Changing the threshold changes which cases are labeled positive or negative."},
        {"question": "Why might the default 0.50 threshold be changed?", "options": ["Different false-positive and false-negative costs may matter", "To make every model perfect", "Because probabilities cannot exceed 0.50"], "answer": "Different false-positive and false-negative costs may matter", "explanation": "The useful threshold depends on the practical consequences of the two error types."},
    ],
    "Week 10": [
        {"question": "What do machine-learning classifiers predict?", "options": ["Class labels or class probabilities", "Only numerical slopes", "Only future dates"], "answer": "Class labels or class probabilities", "explanation": "Classification models assign cases to categories, often through estimated class probabilities."},
        {"question": "What does precision ask?", "options": ["Of the predicted positives, how many were correct?", "Of all actual positives, how many were found?", "How large were regression residuals?"], "answer": "Of the predicted positives, how many were correct?", "explanation": "Precision evaluates the reliability of positive predictions."},
        {"question": "What does recall ask?", "options": ["Of all actual positives, how many were found?", "Of the predicted positives, how many were correct?", "How many predictors were selected?"], "answer": "Of all actual positives, how many were found?", "explanation": "Recall measures how completely the model identifies the positive class."},
        {"question": "Why can accuracy be misleading with a rare positive class?", "options": ["A model can predict the common class almost every time and still appear accurate", "Accuracy always equals recall", "Rare classes cannot be modeled"], "answer": "A model can predict the common class almost every time and still appear accurate", "explanation": "Class imbalance can produce high accuracy while the rare class is poorly detected."},
    ],
    "Week 11": [
        {"question": "What makes a model comparison fair?", "options": ["The same held-out rows and primary metric", "Different targets for every model", "Choosing the winner from training performance"], "answer": "The same held-out rows and primary metric", "explanation": "Models must be evaluated under the same conditions."},
        {"question": "When should the primary evaluation metric be chosen?", "options": ["Before selecting the winning model", "After seeing which metric favors a preferred model", "Only after publication"], "answer": "Before selecting the winning model", "explanation": "Choosing the metric in advance reduces cherry-picking."},
        {"question": "What does a confusion matrix summarize?", "options": ["Correct and incorrect class predictions", "Regression coefficients", "Bootstrap samples"], "answer": "Correct and incorrect class predictions", "explanation": "It separates true positives, true negatives, false positives, and false negatives."},
        {"question": "Can the best prediction model differ from the easiest model to explain?", "options": ["Yes", "No", "Only for image data"], "answer": "Yes", "explanation": "A simpler model may be easier to communicate even when another model predicts slightly better."},
    ],
    "Week 12": [
        {"question": "What is a validation fold?", "options": ["One part held out while the remaining folds train the model", "A duplicated target column", "A probability threshold"], "answer": "One part held out while the remaining folds train the model", "explanation": "Cross-validation rotates which fold is held out for evaluation."},
        {"question": "Why average performance across folds?", "options": ["To estimate typical performance across several splits", "To guarantee perfect accuracy", "To remove all uncertainty"], "answer": "To estimate typical performance across several splits", "explanation": "One split can be unusually easy or difficult, so several folds provide a broader view."},
        {"question": "What does large variation across folds suggest?", "options": ["Performance may be unstable across samples", "The target is definitely causal", "The model has no predictors"], "answer": "Performance may be unstable across samples", "explanation": "Changing validation results indicate sensitivity to which observations are used for training and testing."},
        {"question": "Where must preprocessing be fitted during cross-validation?", "options": ["Inside each training fold", "Once on the complete dataset before folding", "Only on the validation fold"], "answer": "Inside each training fold", "explanation": "Fitting preprocessing on all data would leak information from validation folds into training."},
    ],
    "Week 13": [
        {"question": "How is a bootstrap sample created?", "options": ["Draw rows with replacement", "Sort rows by the target", "Keep only unusual rows"], "answer": "Draw rows with replacement", "explanation": "Sampling with replacement means some rows may appear more than once while others may be absent."},
        {"question": "What is a bootstrap distribution?", "options": ["The collection of estimates from many resampled datasets", "The original dataset sorted by value", "A list of model names"], "answer": "The collection of estimates from many resampled datasets", "explanation": "The variation among bootstrap estimates helps describe sampling uncertainty."},
        {"question": "What does a bootstrap interval communicate?", "options": ["A plausible range for an estimate under the resampling procedure", "A guaranteed range for every future value", "The number of predictors"], "answer": "A plausible range for an estimate under the resampling procedure", "explanation": "The interval summarizes uncertainty in an estimated quantity, not certainty about every future observation."},
        {"question": "Why repeat the bootstrap many times?", "options": ["To obtain a stable picture of the estimate's variation", "To prove the result is causal", "To avoid using data"], "answer": "To obtain a stable picture of the estimate's variation", "explanation": "Many resamples provide a clearer approximation of the sampling distribution."},
    ],
    "Week 14": [
        {"question": "Why must time order be preserved in forecasting?", "options": ["Future observations must not leak into training", "Time order makes every model linear", "It removes the need for evaluation"], "answer": "Future observations must not leak into training", "explanation": "A forecast must be built only from information that would have been available at the forecast origin."},
        {"question": "What is a lagged predictor?", "options": ["A previous value used to predict a later value", "A categorical class label", "A shuffled future observation"], "answer": "A previous value used to predict a later value", "explanation": "Lags represent historical information available before the forecasted time."},
        {"question": "Why compare against a baseline forecast?", "options": ["To check whether the complex method adds useful skill", "To force the baseline to win", "To remove time order"], "answer": "To check whether the complex method adds useful skill", "explanation": "A forecasting model should improve on a simple and credible benchmark."},
        {"question": "Which split is inappropriate for ordinary time-series forecasting?", "options": ["Randomly shuffling past and future rows", "Training on earlier dates and testing on later dates", "Rolling-origin evaluation"], "answer": "Randomly shuffling past and future rows", "explanation": "Random shuffling can place future information in the training set and create leakage."},
    ],
    "Week 15": [
        {"question": "What information does a digital image provide to a model?", "options": ["Pixel values", "Regression slopes only", "Calendar events"], "answer": "Pixel values", "explanation": "Image models learn patterns from numerical pixel brightness and color values."},
        {"question": "What is a class label in image classification?", "options": ["The category assigned to an image", "The image width only", "The number of training epochs"], "answer": "The category assigned to an image", "explanation": "Labels tell the model which category each training image represents."},
        {"question": "Where should final image-classification performance be evaluated?", "options": ["On unseen images", "Only on training images", "On one hand-picked image"], "answer": "On unseen images", "explanation": "Unseen images provide a more honest test of whether the model generalizes."},
        {"question": "What is a warning sign of overfitting in image classification?", "options": ["Very high training accuracy but much lower validation accuracy", "Similar training and validation results", "Using labeled images"], "answer": "Very high training accuracy but much lower validation accuracy", "explanation": "A large training-validation gap suggests the model learned training-specific details that do not transfer."},
    ],
}

WEEK_WRAP_UP_EXTRA_QUESTIONS = {'Week 1': [{'question': 'What is theoretical probability?',
             'options': ['The probability expected from the rules or model of the experiment',
                         'The result observed in one trial',
                         'The number of mistakes in the experiment'],
             'answer': 'The probability expected from the rules or model of the experiment',
             'explanation': 'Theoretical probability comes from the structure or assumptions of the experiment, such as a fair coin having '
                            'probability 0.5 of heads.'},
            {'question': 'What is an event in probability?',
             'options': ['One outcome or a collection of outcomes of interest', 'The number of trials completed', 'A graph of all observations'],
             'answer': 'One outcome or a collection of outcomes of interest',
             'explanation': 'An event is the outcome or group of outcomes whose probability we want to study.'},
            {'question': 'What probability describes an impossible event?',
             'options': ['0', '0.5', '1'],
             'answer': '0',
             'explanation': 'An impossible event cannot occur, so its probability is 0.'},
            {'question': 'What probability describes a certain event?',
             'options': ['1', '0', '0.25'],
             'answer': '1',
             'explanation': 'A certain event must occur, so its probability is 1.'},
            {'question': 'What does it mean for repeated trials to be independent?',
             'options': ['The result of one trial does not change the probabilities for the next trial',
                         'Every trial must produce a different result',
                         'The experiment has only one possible outcome'],
             'answer': 'The result of one trial does not change the probabilities for the next trial',
             'explanation': 'For independent trials, an earlier result does not alter the probability distribution of the next trial.'},
            {'question': 'If a fair coin is tossed 20 times, must exactly 10 tosses be heads?',
             'options': ['No, 10 heads is expected on average but not guaranteed',
                         'Yes, probability forces exactly 10 heads',
                         'Yes, unless the sample space changes'],
             'answer': 'No, 10 heads is expected on average but not guaranteed',
             'explanation': 'Probability describes long-run behavior and likelihood. A small set of trials can differ from the expected '
                            'proportion.'}],
 'Week 2': [{'question': 'Which target type is normally used for a regression question?',
             'options': ['A numerical outcome', 'A folder of images only', 'A column containing unique identification numbers'],
             'answer': 'A numerical outcome',
             'explanation': 'Regression is used when the target is numerical, such as temperature, price, or score.'},
            {'question': 'Which target type is normally used for a classification question?',
             'options': ['A category or class', 'A continuous numerical measurement only', 'The row number'],
             'answer': 'A category or class',
             'explanation': 'Classification predicts categories such as approved/not approved or disease/no disease.'},
            {'question': 'Why is a unique student or record ID usually a poor predictor?',
             'options': ['It normally identifies a row rather than describing a meaningful characteristic',
                         'It is always categorical',
                         'It always contains missing values'],
             'answer': 'It normally identifies a row rather than describing a meaningful characteristic',
             'explanation': 'Identifiers usually label observations and do not represent a generalizable relationship with the target.'},
            {'question': 'When should a predictor be available for a real prediction?',
             'options': ['At or before the time the prediction is made', 'Only after the target is already known', 'Only after model evaluation'],
             'answer': 'At or before the time the prediction is made',
             'explanation': 'A usable predictor must be available when the model is expected to make its prediction.'},
            {'question': 'Why should the research question be written before selecting a model?',
             'options': ['The question determines the target, variables, and suitable method',
                         'The model name automatically creates the research goal',
                         'It guarantees a high score'],
             'answer': 'The question determines the target, variables, and suitable method',
             'explanation': 'The analysis should serve the research question, not the other way around.'},
            {'question': 'What should be checked when first inspecting a dataset?',
             'options': ['Column meanings, variable types, missing values, and possible errors',
                         'Only the file name',
                         'Only the largest numerical value'],
             'answer': 'Column meanings, variable types, missing values, and possible errors',
             'explanation': 'Understanding the available data is necessary before choosing a valid target and predictors.'}],
 'Week 3': [{'question': 'Which measure of center is usually less affected by extreme values?',
             'options': ['Median', 'Mean', 'Range'],
             'answer': 'Median',
             'explanation': 'The median depends on the ordered middle value and is usually more resistant to extreme observations than the mean.'},
            {'question': 'Which plot is suitable for showing counts across categories?',
             'options': ['Bar chart', 'Scatterplot', 'Residual plot'],
             'answer': 'Bar chart',
             'explanation': 'A bar chart compares the number or proportion of observations in each category.'},
            {'question': 'Which plot can compare a numerical variable across several groups?',
             'options': ['Boxplot', 'ROC curve', 'Autocorrelation plot only'],
             'answer': 'Boxplot',
             'explanation': 'A boxplot summarizes the center, spread, and possible unusual values of a numerical variable for each group.'},
            {'question': 'Why should missing values be checked before analysis?',
             'options': ['They can change sample size and affect summaries or models',
                         'They always prove the data are unusable',
                         'They automatically become zeros'],
             'answer': 'They can change sample size and affect summaries or models',
             'explanation': 'Missing data may reduce the usable observations or require a justified handling method.'},
            {'question': 'Why are axis labels and units important?',
             'options': ['They tell the reader what each value represents', 'They increase model accuracy', 'They remove outliers'],
             'answer': 'They tell the reader what each value represents',
             'explanation': 'A chart cannot be interpreted correctly without knowing the variables and measurement units.'},
            {'question': 'What can a visualization do before modeling?',
             'options': ['Reveal patterns that should be investigated',
                         'Prove that one variable causes another',
                         'Guarantee that a model will generalize'],
             'answer': 'Reveal patterns that should be investigated',
             'explanation': 'Plots help us notice patterns and problems, but they do not by themselves establish causation or future performance.'}],
 'Week 4': [{'question': 'What does a negative correlation mean?',
             'options': ['Larger values of one variable tend to occur with smaller values of the other',
                         'Both variables always increase together',
                         'The variables have no observations'],
             'answer': 'Larger values of one variable tend to occur with smaller values of the other',
             'explanation': 'A negative direction means the variables tend to move in opposite directions.'},
            {'question': 'What is the usual range of a correlation coefficient?',
             'options': ['From -1 to 1', 'From 0 to 100', 'From negative infinity to positive infinity'],
             'answer': 'From -1 to 1',
             'explanation': 'Correlation values near -1 or 1 indicate stronger linear relationships, while values near 0 indicate weak linear '
                            'association.'},
            {'question': 'Can two variables have a nonlinear relationship even when linear correlation is near zero?',
             'options': ['Yes', 'No', 'Only when both are categories'],
             'answer': 'Yes',
             'explanation': 'A near-zero linear correlation does not rule out a curved or otherwise nonlinear relationship.'},
            {'question': 'Which plot is especially useful for examining two numerical variables?',
             'options': ['Scatterplot', 'Pie chart', 'Confusion matrix'],
             'answer': 'Scatterplot',
             'explanation': 'A scatterplot shows the direction, form, and possible unusual observations in a two-numerical-variable relationship.'},
            {'question': 'Why might controls be added in a partial correlation?',
             'options': ['To examine the relationship after accounting for selected other variables',
                         'To turn both variables into categories',
                         'To guarantee causation'],
             'answer': 'To examine the relationship after accounting for selected other variables',
             'explanation': 'Partial correlation removes the linear contribution of the selected controls before measuring the remaining '
                            'relationship.'},
            {'question': 'What does eta mainly communicate?',
             'options': ['The strength of association between a categorical grouping variable and a numerical variable',
                         'The direction of a time-series forecast',
                         'The number of model coefficients'],
             'answer': 'The strength of association between a categorical grouping variable and a numerical variable',
             'explanation': 'Eta summarizes how strongly numerical values differ across categories; it does not have a positive or negative '
                            'direction like correlation.'}],
 'Week 5': [{'question': 'What does the intercept represent?',
             'options': ['The predicted target when the predictor equals zero',
                         'The number of observations',
                         'The average classification probability'],
             'answer': 'The predicted target when the predictor equals zero',
             'explanation': 'The intercept is the fitted value of the target at predictor value zero, although that value may not always be '
                            'meaningful in context.'},
            {'question': 'What does R-squared summarize?',
             'options': ['The proportion of target variation explained by the fitted model',
                         'The number of predictors',
                         'The probability that the model is causal'],
             'answer': 'The proportion of target variation explained by the fitted model',
             'explanation': 'R-squared describes model fit for the observed data but does not establish causation.'},
            {'question': 'What pattern supports using a straight-line model?',
             'options': ['The scatterplot shows an approximately linear relationship',
                         'The target contains only categories',
                         'All observations have the same predictor value'],
             'answer': 'The scatterplot shows an approximately linear relationship',
             'explanation': 'Simple linear regression is most suitable when the mean relationship is reasonably represented by a straight line.'},
            {'question': 'Why is prediction far outside the observed predictor range risky?',
             'options': ['The fitted relationship may not continue beyond the observed data',
                         'The slope always becomes zero',
                         'The sample space disappears'],
             'answer': 'The fitted relationship may not continue beyond the observed data',
             'explanation': 'Extrapolation assumes the observed pattern continues where the model has little or no supporting data.'},
            {'question': 'Does a statistically or visually strong slope prove that the predictor causes the target?',
             'options': ['No', 'Yes', 'Only when R-squared exceeds 0.50'],
             'answer': 'No',
             'explanation': 'Regression can describe association, but causal conclusions require stronger study design and assumptions.'},
            {'question': 'Why should the fitted line be evaluated on held-out data?',
             'options': ['To estimate how well it predicts observations not used to fit the line',
                         'To make the training error smaller',
                         'To remove the target from the dataset'],
             'answer': 'To estimate how well it predicts observations not used to fit the line',
             'explanation': 'Held-out evaluation provides a more honest assessment of generalization.'}],
 'Week 6': [{'question': 'What can a residual plot reveal?',
             'options': ['Patterns the linear model failed to capture', 'The exact causal effect of every predictor', 'The class labels of a target'],
             'answer': 'Patterns the linear model failed to capture',
             'explanation': 'Curvature, changing spread, or other structure in residuals can show that linear-model assumptions are inadequate.'},
            {'question': 'Why might categorical predictors require encoding?',
             'options': ['Models need a numerical representation of the categories',
                         'Categories must always be deleted',
                         'Encoding guarantees better prediction'],
             'answer': 'Models need a numerical representation of the categories',
             'explanation': 'Indicator or one-hot variables allow a regression model to represent group differences.'},
            {'question': 'Why compare simple and multiple regression on the same held-out rows?',
             'options': ['To make the performance comparison fair', 'To guarantee that multiple regression wins', 'To eliminate all uncertainty'],
             'answer': 'To make the performance comparison fair',
             'explanation': 'Both models must face the same observations and metric for a meaningful comparison.'},
            {'question': 'What can happen when too many weak predictors are added?',
             'options': ['The model may overfit and perform worse on new data',
                         'The target automatically becomes categorical',
                         'Every coefficient becomes causal'],
             'answer': 'The model may overfit and perform worse on new data',
             'explanation': 'Additional predictors can fit noise rather than useful general patterns.'},
            {'question': 'Why can a coefficient change after another predictor is added?',
             'options': ['The coefficient is now estimated while holding the added predictor constant',
                         'The data type of the target always changes',
                         'The intercept is removed'],
             'answer': 'The coefficient is now estimated while holding the added predictor constant',
             'explanation': 'Shared information among predictors can alter the conditional relationship represented by each coefficient.'},
            {'question': 'What is the purpose of checking assumptions?',
             'options': ["To judge whether the linear model's interpretation and predictions are trustworthy",
                         'To force every residual to equal zero',
                         'To prove the predictors are independent in the population'],
             'answer': "To judge whether the linear model's interpretation and predictions are trustworthy",
             'explanation': 'Assumption checks help identify when a linear model may be misspecified or its uncertainty misleading.'}],
 'Week 7': [{'question': 'Which result should normally be used to choose among regression models?',
             'options': ['Held-out prediction error', 'Training error alone', 'The number of settings shown'],
             'answer': 'Held-out prediction error',
             'explanation': 'The purpose is to estimate performance on new data rather than reward memorization of the training set.'},
            {'question': 'What does a lower root mean squared error indicate?',
             'options': ['Predictions are closer to observed values under that metric', 'The model has more predictors', 'The target is categorical'],
             'answer': 'Predictions are closer to observed values under that metric',
             'explanation': 'For error metrics such as root mean squared error, smaller values are better.'},
            {'question': 'Does the most complex regression model always predict best?',
             'options': ['No', 'Yes', 'Only when it is a neural network'],
             'answer': 'No',
             'explanation': 'Complexity can help capture patterns, but it can also increase overfitting and instability.'},
            {'question': 'What is a model hyperparameter?',
             'options': ['A setting chosen before or during training that controls model behavior',
                         'The observed target value',
                         'A final prediction error'],
             'answer': 'A setting chosen before or during training that controls model behavior',
             'explanation': 'Examples include tree depth, number of trees, and regularization strength.'},
            {'question': 'Why should hyperparameters not be selected using the final test set?',
             'options': ['It would leak test information into model selection',
                         'It would make the test set larger',
                         'It would turn regression into classification'],
             'answer': 'It would leak test information into model selection',
             'explanation': 'The test set should remain independent until the final evaluation.'},
            {'question': 'Why may a slightly less accurate simple model still be useful?',
             'options': ['It may be easier to explain, check, and deploy', 'It always has zero error', 'It proves causation'],
             'answer': 'It may be easier to explain, check, and deploy',
             'explanation': 'Prediction performance is important, but interpretability and practical use can also matter.'}],
 'Week 8': [{'question': 'Which task normally uses a numerical target?',
             'options': ['Regression', 'Classification', 'Image labeling only'],
             'answer': 'Regression',
             'explanation': 'Regression estimates a numerical outcome.'},
            {'question': 'Which task normally uses a categorical target?',
             'options': ['Classification', 'Regression', 'Descriptive statistics only'],
             'answer': 'Classification',
             'explanation': 'Classification predicts a class or category.'},
            {'question': 'What is the purpose of a held-out test set?',
             'options': ['To evaluate performance on observations not used for fitting',
                         'To choose every model setting repeatedly',
                         'To make the training data larger'],
             'answer': 'To evaluate performance on observations not used for fitting',
             'explanation': 'Held-out data provide an honest check of generalization.'},
            {'question': 'What does overfitting mean?',
             'options': ['Learning training-specific noise that does not transfer well',
                         'Using too few decimal places',
                         'Having a clearly stated target'],
             'answer': 'Learning training-specific noise that does not transfer well',
             'explanation': 'An overfit model performs much better on training data than on new observations.'},
            {'question': 'What should an interpretation include?',
             'options': ['The result, its meaning in context, and an important limitation', 'Only the model name', 'Only the largest metric'],
             'answer': 'The result, its meaning in context, and an important limitation',
             'explanation': 'A complete interpretation connects the numerical output to the research question without overstating the evidence.'},
            {'question': 'Does an observed association automatically prove causation?',
             'options': ['No', 'Yes', 'Only for large datasets'],
             'answer': 'No',
             'explanation': 'Causation requires appropriate design and assumptions beyond an observed relationship.'}],
 'Week 9': [{'question': 'What is the positive class?',
             'options': ['The class treated as the event of interest', 'The class with the longest name', 'Always the most common class'],
             'answer': 'The class treated as the event of interest',
             'explanation': 'Metrics such as precision and recall are interpreted relative to the chosen positive class.'},
            {'question': 'What is a false positive?',
             'options': ['The model predicts positive when the observed class is negative',
                         'The model predicts negative when the observed class is positive',
                         'The model predicts a probability of exactly zero'],
             'answer': 'The model predicts positive when the observed class is negative',
             'explanation': 'A false positive is a false alarm.'},
            {'question': 'What is a false negative?',
             'options': ['The model predicts negative when the observed class is positive',
                         'The model predicts positive when the observed class is negative',
                         'The model uses a numerical predictor'],
             'answer': 'The model predicts negative when the observed class is positive',
             'explanation': 'A false negative is a missed positive case.'},
            {'question': 'What usually happens when the classification threshold is lowered?',
             'options': ['More observations are predicted positive',
                         'No observations can be predicted positive',
                         'The model becomes a regression model'],
             'answer': 'More observations are predicted positive',
             'explanation': 'A lower cutoff makes it easier for a predicted probability to be classified as positive.'},
            {'question': 'Is a predicted probability of 0.80 a guarantee that the positive class will occur?',
             'options': ['No', 'Yes', 'Only when the threshold is 0.50'],
             'answer': 'No',
             'explanation': 'A probability expresses uncertainty; individual outcomes can differ from the most likely class.'},
            {'question': 'What does a confusion matrix help identify?',
             'options': ['The types of correct and incorrect class predictions', 'The slope of a regression line', 'The number of bootstrap samples'],
             'answer': 'The types of correct and incorrect class predictions',
             'explanation': 'It separates true positives, true negatives, false positives, and false negatives.'}],
 'Week 10': [{'question': 'What does the F1-score combine?',
              'options': ['Precision and recall', 'Mean absolute error and R-squared', 'Slope and intercept'],
              'answer': 'Precision and recall',
              'explanation': 'F1 is a harmonic balance of precision and recall.'},
             {'question': 'Why should classifiers use the same train-test split when compared?',
              'options': ['So each model is evaluated under the same conditions',
                          'So every model obtains the same predictions',
                          'So the target becomes numerical'],
              'answer': 'So each model is evaluated under the same conditions',
              'explanation': 'A fair comparison requires the same observations, predictors, and evaluation rules.'},
             {'question': 'Why is training accuracy alone insufficient?',
              'options': ['A model can memorize training patterns and fail on new data',
                          'Training accuracy cannot be calculated',
                          'It is always lower than test accuracy'],
              'answer': 'A model can memorize training patterns and fail on new data',
              'explanation': 'Held-out performance is needed to assess generalization.'},
             {'question': 'Can a probability-producing classifier use different thresholds?',
              'options': ['Yes', 'No', 'Only logistic regression can produce probabilities'],
              'answer': 'Yes',
              'explanation': 'Many classifiers can produce class probabilities, and the threshold can be adjusted to match the decision costs.'},
             {'question': 'Does a more complex classifier automatically have better held-out performance?',
              'options': ['No', 'Yes', 'Only when the data are imbalanced'],
              'answer': 'No',
              'explanation': 'Complex methods may capture useful patterns or may overfit; evaluation must decide.'},
             {'question': 'What can class weighting help address?',
              'options': ['Unequal importance or frequency of classes', 'A numerical regression target', 'Incorrect axis labels'],
              'answer': 'Unequal importance or frequency of classes',
              'explanation': 'Class weights can make errors on a rare or important class count more during training.'}],
 'Week 11': [{'question': 'For mean absolute error and root mean squared error, which direction is better?',
              'options': ['Lower', 'Higher', 'Exactly 1'],
              'answer': 'Lower',
              'explanation': 'These metrics measure prediction error, so smaller values indicate closer predictions.'},
             {'question': 'For accuracy, recall, and F1-score, which direction is usually better?',
              'options': ['Higher', 'Lower', 'Exactly 0'],
              'answer': 'Higher',
              'explanation': 'These classification metrics describe successful predictions, so higher values are usually preferred.'},
             {'question': 'Why should the final test set remain untouched during model selection?',
              'options': ['To preserve an unbiased final evaluation', 'To increase training accuracy', 'To ensure every model wins'],
              'answer': 'To preserve an unbiased final evaluation',
              'explanation': 'Repeatedly using the test set to make choices turns it into part of the training process.'},
             {'question': 'Which metric type belongs to regression?',
              'options': ['Mean absolute error', 'Recall', 'Confusion-matrix specificity only'],
              'answer': 'Mean absolute error',
              'explanation': 'Mean absolute error measures the distance between numerical predictions and numerical observations.'},
             {'question': 'Which metric may be especially important when missing positive cases is costly?',
              'options': ['Recall', 'R-squared', 'Regression slope'],
              'answer': 'Recall',
              'explanation': 'Recall measures the fraction of actual positive cases that the model successfully finds.'},
             {'question': 'Why include a baseline model in comparison?',
              'options': ['To judge whether more complex models provide meaningful improvement',
                          'To guarantee the baseline is selected',
                          'To remove the need for a test set'],
              'answer': 'To judge whether more complex models provide meaningful improvement',
              'explanation': 'A complex method should outperform a credible simple alternative enough to justify its extra complexity.'}],
 'Week 12': [{'question': 'In 5-fold cross-validation, how many parts is the dataset divided into?',
              'options': ['5', '2', '10'],
              'answer': '5',
              'explanation': 'Five-fold cross-validation creates five parts and uses each part once for validation.'},
             {'question': 'How many times is each observation normally used for validation in ordinary k-fold cross-validation?',
              'options': ['Once', 'Never', 'In every fold'],
              'answer': 'Once',
              'explanation': 'Each observation belongs to one validation fold and to the training portion of the other folds.'},
             {'question': 'Why report both the mean score and variation across folds?',
              'options': ['They describe typical performance and stability', 'They prove causation', 'They replace the research question'],
              'answer': 'They describe typical performance and stability',
              'explanation': 'Two models can have similar averages but very different consistency across samples.'},
             {'question': 'Should ordinary shuffled k-fold cross-validation be used for forecasting?',
              'options': ['No, time-aware validation is needed', 'Yes, always', 'Only when the target is categorical'],
              'answer': 'No, time-aware validation is needed',
              'explanation': 'Forecasting must preserve time order so future observations do not enter training.'},
             {'question': 'What is data leakage during cross-validation?',
              'options': ['Information from validation folds influences model fitting or preprocessing',
                          'A model has fewer than five predictors',
                          'A fold has a low score'],
              'answer': 'Information from validation folds influences model fitting or preprocessing',
              'explanation': 'Leakage makes validation scores overly optimistic.'},
             {'question': 'Why might repeated cross-validation be useful?',
              'options': ['It examines performance across more than one random fold arrangement',
                          'It guarantees perfect prediction',
                          'It removes all computation'],
              'answer': 'It examines performance across more than one random fold arrangement',
              'explanation': 'Repeating the split provides additional evidence about performance stability.'}],
 'Week 13': [{'question': 'What does sampling with replacement allow?',
              'options': ['A row may appear more than once in a bootstrap sample',
                          'Every row must appear exactly once',
                          'Only missing rows are selected'],
              'answer': 'A row may appear more than once in a bootstrap sample',
              'explanation': 'Replacement means a selected row is returned to the pool and can be selected again.'},
             {'question': 'Will every original row appear in every bootstrap sample?',
              'options': ['No', 'Yes', 'Only when the sample is numerical'],
              'answer': 'No',
              'explanation': 'Some rows are repeated and some are absent in each bootstrap sample.'},
             {'question': 'What does a larger bootstrap standard error indicate?',
              'options': ['The estimate varies more across resamples', 'The estimate is automatically unbiased', 'The dataset has no target'],
              'answer': 'The estimate varies more across resamples',
              'explanation': 'A larger standard error signals greater sampling uncertainty.'},
             {'question': 'What usually happens to an uncertainty interval when the estimate is less stable?',
              'options': ['It tends to become wider', 'It must become zero', 'It changes the target to a category'],
              'answer': 'It tends to become wider',
              'explanation': 'Greater variation in the bootstrap distribution generally produces a wider interval.'},
             {'question': 'Can bootstrap resampling repair a badly biased or unrepresentative original dataset?',
              'options': ['No', 'Yes', 'Only with 100 resamples'],
              'answer': 'No',
              'explanation': 'Bootstrap describes uncertainty relative to the observed data; it cannot create missing populations or correct all '
                             'systematic bias.'},
             {'question': 'What should usually be resampled together in a row-based dataset?',
              'options': ['Complete rows', 'Each column independently', 'Only the target values'],
              'answer': 'Complete rows',
              'explanation': 'Resampling complete rows preserves the relationships among variables within each observation.'}],
 'Week 14': [{'question': 'What is the forecast horizon?',
              'options': ['How far into the future the prediction is made',
                          'The number of target classes',
                          'The width of a confidence interval only'],
              'answer': 'How far into the future the prediction is made',
              'explanation': 'The horizon may be one step ahead, one day ahead, or another stated future distance.'},
             {'question': 'What is an LSTM lookback window?',
              'options': ['The sequence of past time steps supplied to the model', 'The final test score', 'A random train-test split'],
              'answer': 'The sequence of past time steps supplied to the model',
              'explanation': 'The lookback controls how much recent history the recurrent model receives for each prediction.'},
             {'question': 'When can an external predictor be used in a real forecast?',
              'options': ['When its value is known or can itself be forecast at the forecast origin',
                          'Only after the target future value is observed',
                          'Whenever it improves training error'],
              'answer': 'When its value is known or can itself be forecast at the forecast origin',
              'explanation': 'A predictor unavailable at decision time creates future-information leakage.'},
             {'question': 'What does seasonality mean?',
              'options': ['A pattern that repeats at regular time intervals', 'A random shuffle of observations', 'A categorical target'],
              'answer': 'A pattern that repeats at regular time intervals',
              'explanation': 'Examples include daily, weekly, or yearly repeating behavior.'},
             {'question': 'Which forecast is better under root mean squared error?',
              'options': ['The one with lower root mean squared error',
                          'The one with more lags regardless of error',
                          'The one trained on future rows'],
              'answer': 'The one with lower root mean squared error',
              'explanation': 'Lower root mean squared error means smaller prediction errors under that metric.'},
             {'question': 'Why can multi-step forecasting be harder than one-step forecasting?',
              'options': ['Errors and uncertainty can accumulate farther into the future',
                          'The target becomes categorical',
                          'Time order no longer matters'],
              'answer': 'Errors and uncertainty can accumulate farther into the future',
              'explanation': 'Longer horizons provide less direct information and may depend on earlier forecasted values.'}],
 'Week 15': [{'question': 'What are image channels?',
              'options': ['Separate numerical layers such as red, green, and blue', "The model's class labels", 'The number of validation folds'],
              'answer': 'Separate numerical layers such as red, green, and blue',
              'explanation': 'A color image is commonly represented by multiple channels containing pixel intensities.'},
             {'question': 'Why are images often resized before training?',
              'options': ['The model expects a consistent input shape', 'Resizing guarantees perfect recognition', 'It converts every image to text'],
              'answer': 'The model expects a consistent input shape',
              'explanation': 'A batch of images must usually have the same height, width, and channel count.'},
             {'question': 'What is data augmentation?',
              'options': ['Creating realistic transformed training examples such as flips or small rotations',
                          'Copying the validation labels into training',
                          'Deleting all difficult images'],
              'answer': 'Creating realistic transformed training examples such as flips or small rotations',
              'explanation': 'Augmentation can expose the model to reasonable variation and reduce overfitting.'},
             {'question': 'What can an image confusion matrix show?',
              'options': ['Which classes are being confused with one another', 'Only image dimensions', 'The regression slope of pixels'],
              'answer': 'Which classes are being confused with one another',
              'explanation': 'The off-diagonal cells identify common misclassification pairs.'},
             {'question': 'What is transfer learning?',
              'options': ['Starting from a model pretrained on another large image dataset',
                          'Moving images between folders without labels',
                          'Using the test set to train the model'],
              'answer': 'Starting from a model pretrained on another large image dataset',
              'explanation': 'Pretrained features can reduce the amount of task-specific training data and computation needed.'},
             {'question': 'Why should every class be represented in validation data?',
              'options': ['So performance can be checked for each class',
                          'So training accuracy becomes 100 percent',
                          'So labels are no longer needed'],
              'answer': 'So performance can be checked for each class',
              'explanation': 'A class absent from validation cannot be honestly evaluated.'}]}




REVIEW_WRAP_UP_QUESTIONS = {'Week 8': [{'question': 'Which statement best defines artificial intelligence (AI)?',
             'options': ['The broad field of building systems that perform tasks associated with human intelligence',
                         'A single algorithm used only for regression',
                         'Any spreadsheet containing numerical data',
                         'A rule that guarantees a computer is conscious'],
             'answer': 'The broad field of building systems that perform tasks associated with human intelligence',
             'explanation': 'AI is the broad field; it includes learning, reasoning, perception, language, planning, and other intelligent '
                            'behavior.'},
            {'question': 'Which statement best defines machine learning?',
             'options': ['A part of AI in which systems learn patterns from data to make predictions or decisions',
                         'A method for manually writing every decision rule',
                         'A database management system',
                         'A guarantee that a model will be unbiased'],
             'answer': 'A part of AI in which systems learn patterns from data to make predictions or decisions',
             'explanation': 'Machine learning is a subset of AI that learns relationships from examples rather than relying only on hand-written '
                            'rules.'},
            {'question': 'How are AI and machine learning related?',
             'options': ['Machine learning is one approach within the broader field of AI',
                         'AI is a small subset of machine learning',
                         'They are unrelated fields',
                         'They mean exactly the same thing in every context'],
             'answer': 'Machine learning is one approach within the broader field of AI',
             'explanation': 'AI is broader. Machine learning is one major way of creating AI systems.'},
            {'question': 'Which task is supervised learning?',
             'options': ['Learning from rows that include known target values',
                         'Grouping unlabeled observations only',
                         'Storing a file without analyzing it',
                         'Randomly deleting predictors'],
             'answer': 'Learning from rows that include known target values',
             'explanation': 'Supervised learning uses labeled examples: predictors are paired with a known target.'},
            {'question': 'Which problem is a regression problem?',
             'options': ['Predicting a numerical house price',
                         'Predicting whether an email is spam',
                         'Grouping customers without labels',
                         'Recognizing the language of a document as one of five classes'],
             'answer': 'Predicting a numerical house price',
             'explanation': 'Regression predicts a numerical target.'},
            {'question': 'Which problem is a classification problem?',
             'options': ['Predicting whether a student passes or does not pass',
                         "Predicting a student's exact score",
                         'Estimating average rainfall in millimeters',
                         'Calculating the median of a column'],
             'answer': 'Predicting whether a student passes or does not pass',
             'explanation': 'Classification predicts a category or class.'},
            {'question': 'What is a neural network?',
             'options': ['A layered model that combines weighted inputs and nonlinear transformations to learn complex patterns',
                         'A table that stores only target values',
                         'A rule that always produces a straight line',
                         'A chart used only for categorical variables'],
             'answer': 'A layered model that combines weighted inputs and nonlinear transformations to learn complex patterns',
             'explanation': 'Neural networks learn through connected layers of weighted calculations and activation functions.'},
            {'question': 'What is one important difference between a neural network and a support vector machine (SVM)?',
             'options': ['A neural network learns through multiple connected layers, while an SVM seeks a separating boundary with a large margin',
                         'An SVM has biological neurons, while a neural network does not',
                         'A neural network can only classify, while an SVM can only regress',
                         'An SVM never uses numerical predictors'],
             'answer': 'A neural network learns through multiple connected layers, while an SVM seeks a separating boundary with a large margin',
             'explanation': 'The models use different learning structures: layered representations versus a margin-based decision boundary.'},
            {'question': 'What does an SVM try to do in a basic classification problem?',
             'options': ['Find a decision boundary that separates classes with the largest possible margin',
                         'Average all target values',
                         'Create a histogram of every predictor',
                         'Select the class with the most missing values'],
             'answer': 'Find a decision boundary that separates classes with the largest possible margin',
             'explanation': 'A support vector machine focuses on a boundary and the observations closest to it.'},
            {'question': 'Why is feature scaling often important for SVMs and neural networks?',
             'options': ['Variables with very different numerical scales can affect optimization and distance-based calculations unevenly',
                         'Scaling changes a categorical target into a numerical target',
                         'Scaling guarantees perfect test accuracy',
                         'Scaling removes the need for a test set'],
             'answer': 'Variables with very different numerical scales can affect optimization and distance-based calculations unevenly',
             'explanation': 'Standardizing features can make optimization more stable and prevent large-unit variables from dominating.'},
            {'question': 'What is a hyperparameter?',
             'options': ['A setting chosen before or during training that controls how a model learns',
                         'A target value observed after prediction',
                         'A residual from the test set',
                         'A column name that must contain the word hyper'],
             'answer': 'A setting chosen before or during training that controls how a model learns',
             'explanation': 'Examples include tree depth, learning rate, regularization strength, and number of hidden units.'},
            {'question': 'What is hyperparameter tuning?',
             'options': ['Systematically comparing candidate settings using validation evidence',
                         'Changing settings until training accuracy reaches 100 percent',
                         'Using the test set repeatedly to choose the best model',
                         'Renaming predictors after training'],
             'answer': 'Systematically comparing candidate settings using validation evidence',
             'explanation': 'Tuning should use training and validation data, while the final test set remains untouched.'},
            {'question': 'What is regularization?',
             'options': ['A method that discourages excessive model complexity to reduce overfitting',
                         'A way to copy test rows into training',
                         'A rule that increases every coefficient',
                         'A requirement that all predictors be categorical'],
             'answer': 'A method that discourages excessive model complexity to reduce overfitting',
             'explanation': 'Regularization adds constraints or penalties so the model does not fit noise too aggressively.'},
            {'question': 'What is overfitting?',
             'options': ['Performing very well on training data but poorly on new data',
                         'Performing equally well on training and test data',
                         'Using too few observations to calculate a mean',
                         'Having a categorical target'],
             'answer': 'Performing very well on training data but poorly on new data',
             'explanation': 'An overfit model memorizes training-specific patterns that do not generalize.'},
            {'question': 'What is underfitting?',
             'options': ['A model is too simple to capture important structure in both training and test data',
                         'A model has too many hidden layers',
                         'A model uses a large training set',
                         'A model has a low test error'],
             'answer': 'A model is too simple to capture important structure in both training and test data',
             'explanation': 'Underfitting means the model has not learned enough of the real pattern.'},
            {'question': 'Why do we keep a test set separate from model training?',
             'options': ['To estimate performance on unseen observations',
                         'To increase the training sample secretly',
                         'To choose every hyperparameter',
                         'To guarantee causation'],
             'answer': 'To estimate performance on unseen observations',
             'explanation': 'The test set provides a more honest estimate of generalization when it is not used for fitting or tuning.'},
            {'question': 'What is target leakage?',
             'options': ['A predictor contains information about the target that would not legitimately be available when predictions are made',
                         'A target has a few missing values',
                         'A model uses fewer than five predictors',
                         'A chart has an unlabeled axis'],
             'answer': 'A predictor contains information about the target that would not legitimately be available when predictions are made',
             'explanation': 'Leakage gives the model unfair access to the answer and produces overly optimistic performance.'},
            {'question': 'What is a predictor?',
             'options': ['A variable used as input to help explain or predict the target',
                         'The final score used to grade a student',
                         'The same thing as a residual',
                         'A value observed only after deployment'],
             'answer': 'A variable used as input to help explain or predict the target',
             'explanation': 'Predictors are the input information supplied to a model.'},
            {'question': 'Which research question is most suitable for data analysis?',
             'options': ['How well do study hours and attendance predict mathematics score in this dataset?',
                         'Why is education important?',
                         'Can data answer every possible question?',
                         'Which model is always best?'],
             'answer': 'How well do study hours and attendance predict mathematics score in this dataset?',
             'explanation': 'A strong research question identifies an outcome, relevant information, and a question the available data can address.'},
            {'question': 'A fair coin is tossed once. What is the sample space?',
             'options': ['{Heads, Tails}', '{Heads}', '{0.5}', '{Heads, Tails, Edge, Missing}'],
             'answer': '{Heads, Tails}',
             'explanation': 'The sample space is the complete set of possible outcomes.'},
            {'question': 'Which value cannot be a probability?',
             'options': ['1.25', '0', '0.40', '1'],
             'answer': '1.25',
             'explanation': 'Probabilities must lie between 0 and 1 inclusive.'},
            {'question': 'If P(A) = 0.35, what is P(not A)?',
             'options': ['0.65', '0.35', '1.35', '0.1225'],
             'answer': '0.65',
             'explanation': 'The complement rule gives P(not A) = 1 - P(A) = 0.65.'},
            {'question': 'Events A and B are mutually exclusive, with P(A) = 0.20 and P(B) = 0.30. What is P(A or B)?',
             'options': ['0.50', '0.06', '0.10', '0.70'],
             'answer': '0.50',
             'explanation': 'For mutually exclusive events, add the probabilities: 0.20 + 0.30 = 0.50.'},
            {'question': 'Two independent events have probabilities 0.50 and 0.40. What is the probability that both occur?',
             'options': ['0.20', '0.90', '0.10', '0.45'],
             'answer': '0.20',
             'explanation': 'For independent events, multiply: 0.50 × 0.40 = 0.20.'},
            {'question': 'In a class, 20 students studied, and 15 of those students passed. What is P(Pass | Studied)?',
             'options': ['0.75', '0.25', '1.33', '0.15'],
             'answer': '0.75',
             'explanation': 'Conditional probability within the studied group is 15 divided by 20, which is 0.75.'},
            {'question': 'A simulation produces 18 heads in 30 tosses. What is the experimental probability of heads?',
             'options': ['0.60', '0.40', '0.18', '1.67'],
             'answer': '0.60',
             'explanation': 'Experimental probability is observed successes divided by trials: 18/30 = 0.60.'},
            {'question': 'A biased coin has P(Heads) = 0.60. How many heads are expected on average in 50 tosses?',
             'options': ['30', '20', '50', '0.60'],
             'answer': '30',
             'explanation': 'The expected count is n × p = 50 × 0.60 = 30.'},
            {'question': 'A game pays $10 with probability 0.20 and $0 otherwise. What is the expected payout?',
             'options': ['$2', '$8', '$10', '$0.20'],
             'answer': '$2',
             'explanation': 'Expected value is 10 × 0.20 + 0 × 0.80 = $2.'},
            {'question': 'What does variance describe?',
             'options': ['How spread out values are around their mean',
                         'The largest observed value only',
                         'The number of categories',
                         'Whether one variable causes another'],
             'answer': 'How spread out values are around their mean',
             'explanation': 'Variance summarizes average squared deviation from the mean.'},
            {'question': 'What usually happens as the number of independent simulation trials becomes very large?',
             'options': ['The experimental proportion often moves closer to the theoretical probability',
                         'Every short run becomes exactly equal to the theoretical probability',
                         'The sample space becomes smaller',
                         'Probability values become greater than 1'],
             'answer': 'The experimental proportion often moves closer to the theoretical probability',
             'explanation': 'This is the long-run stabilization described by the law of large numbers.'},
            {'question': 'Which measure of center is usually more resistant to extreme outliers?',
             'options': ['Median', 'Mean', 'Range', 'Variance'],
             'answer': 'Median',
             'explanation': 'The median depends on order rather than the magnitude of the most extreme values.'},
            {'question': 'What does standard deviation measure?',
             'options': ["Typical spread of values around the mean in the variable's units",
                         'The number of missing rows',
                         'The direction of causation',
                         'The test-set sample size only'],
             'answer': "Typical spread of values around the mean in the variable's units",
             'explanation': 'Standard deviation is a spread measure expressed in the original units.'},
            {'question': 'Which plot is most suitable for showing the distribution of one numerical variable?',
             'options': ['Histogram', 'Confusion matrix', 'Line graph of model epochs only', 'Pie chart of residuals'],
             'answer': 'Histogram',
             'explanation': "A histogram groups numerical values into intervals to show the distribution's shape."},
            {'question': 'Which plot is especially useful for comparing a numerical variable across categories?',
             'options': ['Boxplot', 'Single-value metric card', 'Confusion matrix', 'Network diagram'],
             'answer': 'Boxplot',
             'explanation': 'Side-by-side boxplots display center, spread, and potential outliers for each group.'},
            {'question': 'Which plot is most suitable for examining the relationship between two numerical variables?',
             'options': ['Scatterplot', 'Bar chart of one category count only', 'Confusion matrix', 'Image montage'],
             'answer': 'Scatterplot',
             'explanation': 'A scatterplot shows paired numerical values and reveals direction, form, and unusual points.'},
            {'question': 'A correlation of -0.82 indicates what?',
             'options': ['A strong negative linear association',
                         'A strong positive linear association',
                         'No relationship of any kind',
                         'Proof that one variable causes the other'],
             'answer': 'A strong negative linear association',
             'explanation': 'The negative sign gives direction and the magnitude near 1 indicates a strong linear association.'},
            {'question': 'Which correlation is strongest in magnitude?',
             'options': ['-0.90', '0.65', '-0.20', '0.05'],
             'answer': '-0.90',
             'explanation': 'Strength is based on absolute value; |-0.90| is largest.'},
            {'question': 'Why does correlation not establish causation?',
             'options': ['A third variable, reverse direction, or study design may explain the association',
                         'Correlation can never be calculated',
                         'Causation requires a negative coefficient',
                         'Only categorical variables can be causal'],
             'answer': 'A third variable, reverse direction, or study design may explain the association',
             'explanation': 'Observational association alone does not isolate a causal effect.'},
            {'question': 'What does partial correlation examine?',
             'options': ['The relationship between two numerical variables after accounting for selected control variables',
                         'Only the largest correlation in a table',
                         'A relationship between two images',
                         'The probability of a classification threshold'],
             'answer': 'The relationship between two numerical variables after accounting for selected control variables',
             'explanation': 'Partial correlation removes the linear contribution of specified controls before measuring association.'},
            {'question': 'In simple linear regression, what does the slope represent?',
             'options': ['The predicted change in the target for a one-unit increase in the predictor',
                         'The target value for every observation',
                         'The average prediction error',
                         'The number of rows in the test set'],
             'answer': 'The predicted change in the target for a one-unit increase in the predictor',
             'explanation': 'The slope connects a one-unit predictor increase to the fitted change in the target.'},
            {'question': 'When can a regression intercept have little practical meaning?',
             'options': ['When predictor value zero is impossible or far outside the observed data',
                         'Whenever R-squared is positive',
                         'Whenever the slope is negative',
                         'When the test set has observations'],
             'answer': 'When predictor value zero is impossible or far outside the observed data',
             'explanation': 'The intercept is the prediction at predictor zero, which may be outside the meaningful range.'},
            {'question': 'What is a residual?',
             'options': ['Observed target minus predicted target',
                         'Predicted target plus observed target',
                         'The average of all predictors',
                         'The probability of the positive class'],
             'answer': 'Observed target minus predicted target',
             'explanation': 'A residual records the signed prediction error for one observation.'},
            {'question': 'A model has mean absolute error of 6 score points. What does this mean?',
             'options': ['Its predictions miss the observed scores by about 6 points on average',
                         'It explains exactly 6 percent of the variation',
                         'Every prediction is wrong by exactly 6 points',
                         'Its accuracy is 94 percent'],
             'answer': 'Its predictions miss the observed scores by about 6 points on average',
             'explanation': 'Mean absolute error is the average absolute prediction mistake in target units.'},
            {'question': 'Why can root mean squared error be larger than mean absolute error?',
             'options': ['Squaring gives extra weight to larger prediction errors',
                         'Root mean squared error ignores all large errors',
                         'Mean absolute error uses only training rows',
                         'Root mean squared error is a classification metric'],
             'answer': 'Squaring gives extra weight to larger prediction errors',
             'explanation': 'Large residuals contribute disproportionately to root mean squared error.'},
            {'question': 'What does an R-squared of 0.64 mean on the evaluated data?',
             'options': ['The model explains about 64 percent of the observed target variation under that evaluation',
                         'The model is 64 percent causal',
                         'Every prediction is 64 percent correct',
                         'The mean absolute error is 0.64 target units'],
             'answer': 'The model explains about 64 percent of the observed target variation under that evaluation',
             'explanation': 'R-squared is a proportion of variation explained, not proof of causation or a classification accuracy.'},
            {'question': 'In multiple linear regression, how is one coefficient interpreted?',
             'options': ['As the estimated target change for a one-unit increase in that predictor while other included predictors are held fixed',
                         'As the total change caused by all predictors together',
                         'As the test error',
                         'As the probability of a class'],
             'answer': 'As the estimated target change for a one-unit increase in that predictor while other included predictors are held fixed',
             'explanation': 'Multiple-regression coefficients are conditional on the other included predictors.'},
            {'question': 'What is the safest way to decide whether adding predictors improved a regression model?',
             'options': ['Compare held-out performance using the same evaluation rows',
                         'Choose the model with the most predictors',
                         'Choose the model with the highest training R-squared only',
                         'Use different test sets for each model'],
             'answer': 'Compare held-out performance using the same evaluation rows',
             'explanation': 'New predictors are useful only if they improve generalization under a fair comparison.'},
            {'question': 'What does a decision tree learn?',
             'options': ['A sequence of feature-based splits that divide observations into prediction regions',
                         'One fixed straight line only',
                         'A probability table with no predictors',
                         'A random test-set assignment'],
             'answer': 'A sequence of feature-based splits that divide observations into prediction regions',
             'explanation': 'Trees repeatedly split the feature space to create groups with similar targets.'},
            {'question': 'Why can a random forest generalize better than one deep decision tree?',
             'options': ['It averages many varied trees, reducing sensitivity to one training sample',
                         'It uses the test labels during training',
                         'It removes all randomness',
                         'It guarantees zero error'],
             'answer': 'It averages many varied trees, reducing sensitivity to one training sample',
             'explanation': 'Combining many decorrelated trees usually reduces variance compared with a single tree.'},
            {'question': 'How does gradient boosting build a model?',
             'options': ['It adds weak learners sequentially, with later learners correcting earlier errors',
                         'It trains unrelated models and chooses one at random',
                         'It fits only an intercept',
                         'It removes difficult observations from the dataset'],
             'answer': 'It adds weak learners sequentially, with later learners correcting earlier errors',
             'explanation': 'Boosting constructs an additive model in stages, focusing each stage on remaining error.'}],
 'Week 16': [{'question': 'What does logistic regression estimate before a class label is assigned?',
              'options': ['A probability for the positive class',
                          'A continuous target with no upper bound',
                          'A cluster number with no labels',
                          'A bootstrap interval only'],
              'answer': 'A probability for the positive class',
              'explanation': 'Logistic regression first estimates a value between 0 and 1, which can then be converted to a class using a '
                             'threshold.'},
             {'question': 'What does a classification threshold do?',
              'options': ['Converts a predicted probability into a class decision',
                          'Selects the train-test split',
                          'Standardizes every predictor',
                          'Calculates a regression slope'],
              'answer': 'Converts a predicted probability into a class decision',
              'explanation': 'Probabilities at or above the threshold are typically assigned to the positive class.'},
             {'question': 'What usually happens when the positive-class threshold is lowered?',
              'options': ['More cases are predicted positive, often increasing recall and false positives',
                          'Fewer cases are predicted positive, always increasing precision',
                          'The target becomes numerical',
                          'The model no longer produces probabilities'],
              'answer': 'More cases are predicted positive, often increasing recall and false positives',
              'explanation': 'A lower threshold makes positive predictions easier, so sensitivity often rises while false positives may also rise.'},
             {'question': 'What is a true positive?',
              'options': ['The model predicts positive and the observed class is positive',
                          'The model predicts positive and the observed class is negative',
                          'The model predicts negative and the observed class is positive',
                          'The model predicts negative and the observed class is negative'],
              'answer': 'The model predicts positive and the observed class is positive',
              'explanation': 'A true positive is a correctly identified positive case.'},
             {'question': 'What is a false negative?',
              'options': ['The model predicts negative even though the observed class is positive',
                          'The model predicts positive even though the observed class is negative',
                          'The model predicts positive and the observed class is positive',
                          'The model predicts negative and the observed class is negative'],
              'answer': 'The model predicts negative even though the observed class is positive',
              'explanation': 'A false negative is a missed positive case.'},
             {'question': 'A classifier makes 90 correct predictions out of 100. What is its accuracy?',
              'options': ['0.90', '0.10', '9.0', '90.0'],
              'answer': '0.90',
              'explanation': 'Accuracy is correct predictions divided by all predictions: 90/100 = 0.90.'},
             {'question': 'A classifier has 30 true positives and 10 false positives. What is precision?',
              'options': ['0.75', '0.60', '0.30', '0.25'],
              'answer': '0.75',
              'explanation': 'Precision = TP/(TP+FP) = 30/(30+10) = 0.75.'},
             {'question': 'A classifier has 30 true positives and 20 false negatives. What is recall?',
              'options': ['0.60', '0.75', '0.40', '0.50'],
              'answer': '0.60',
              'explanation': 'Recall = TP/(TP+FN) = 30/(30+20) = 0.60.'},
             {'question': 'What does the F1-score balance?',
              'options': ['Precision and recall', 'Training time and file size', 'Mean and median', 'R-squared and root mean squared error'],
              'answer': 'Precision and recall',
              'explanation': 'F1 is the harmonic mean of precision and recall.'},
             {'question': 'Why can accuracy be misleading for an imbalanced classification problem?',
              'options': ['A model can predict the majority class most of the time and still obtain high accuracy',
                          'Accuracy is always lower than recall',
                          'Accuracy cannot be calculated from a confusion matrix',
                          'Imbalanced data automatically create leakage'],
              'answer': 'A model can predict the majority class most of the time and still obtain high accuracy',
              'explanation': 'When one class dominates, high accuracy may hide poor performance on the minority class.'},
             {'question': 'What does an area under the ROC curve near 1 indicate?',
              'options': ['Strong ability to rank positive cases above negative cases across thresholds',
                          'A perfect regression slope',
                          'No discrimination between classes',
                          'A very wide confidence interval'],
              'answer': 'Strong ability to rank positive cases above negative cases across thresholds',
              'explanation': 'ROC AUC summarizes ranking discrimination over many thresholds.'},
             {'question': 'What does a positive logistic-regression coefficient generally indicate?',
              'options': ['Higher predictor values are associated with higher log-odds of the positive class, holding other predictors fixed',
                          'Higher predictor values guarantee the positive class',
                          'The predictor has no relationship with the class',
                          'The threshold must equal zero'],
              'answer': 'Higher predictor values are associated with higher log-odds of the positive class, holding other predictors fixed',
              'explanation': 'The sign gives the direction of association on the log-odds scale, not a guarantee or causal claim.'},
             {'question': 'How does a decision tree classify an observation?',
              'options': ['It follows feature-based split rules from the root to a terminal leaf',
                          'It averages all target labels',
                          'It uses one global straight-line equation only',
                          'It calculates a bootstrap confidence interval first'],
              'answer': 'It follows feature-based split rules from the root to a terminal leaf',
              'explanation': 'Each split sends the observation down a branch until it reaches a prediction leaf.'},
             {'question': 'Why does a random forest use many trees?',
              'options': ['Averaging varied trees usually reduces variance and improves generalization',
                          'Every tree is trained on the full test set',
                          'Many trees guarantee causal inference',
                          'It prevents all predictors from being used'],
              'answer': 'Averaging varied trees usually reduces variance and improves generalization',
              'explanation': 'Random forests combine bootstrap samples and random feature subsets to create diverse trees.'},
             {'question': 'What distinguishes gradient boosting from a random forest?',
              'options': ['Boosting builds learners sequentially to correct remaining errors, while a random forest builds trees more independently '
                          'and averages them',
                          'Gradient boosting cannot use trees',
                          'A random forest is always a neural network',
                          'Gradient boosting uses no target values'],
              'answer': 'Boosting builds learners sequentially to correct remaining errors, while a random forest builds trees more independently '
                        'and averages them',
              'explanation': 'The key distinction is sequential error correction versus parallel-like averaging of varied trees.'},
             {'question': 'What is the central idea of a support vector machine classifier?',
              'options': ['Find a separating boundary with a large margin between classes',
                          'Estimate the mean target value',
                          'Create random bootstrap intervals',
                          'Use only categorical predictors'],
              'answer': 'Find a separating boundary with a large margin between classes',
              'explanation': 'Support vectors near the boundary determine the maximum-margin separator.'},
             {'question': 'What is a feedforward neural network?',
              'options': ['A network in which information moves from input layers through hidden layers to output without recurrent loops',
                          'A model that can only analyze time series',
                          'A single decision-tree split',
                          'A probability simulator'],
              'answer': 'A network in which information moves from input layers through hidden layers to output without recurrent loops',
              'explanation': 'Feedforward networks transform inputs through one or more hidden layers to produce an output.'},
             {'question': 'Why are activation functions used in neural networks?',
              'options': ['They introduce nonlinearity so the network can learn more than a single linear transformation',
                          'They split the test set',
                          'They calculate the sample mean',
                          'They guarantee calibrated probabilities'],
              'answer': 'They introduce nonlinearity so the network can learn more than a single linear transformation',
              'explanation': 'Without nonlinear activations, stacked layers would collapse into an overall linear mapping.'},
             {'question': 'What is an epoch in neural-network training?',
              'options': ['One complete pass through the training data',
                          'One predictor column',
                          'One class in the target',
                          'One bootstrap confidence limit'],
              'answer': 'One complete pass through the training data',
              'explanation': 'Training commonly uses many epochs, with parameters updated repeatedly.'},
             {'question': 'What can happen when the learning rate is far too large?',
              'options': ['Optimization may overshoot good solutions and become unstable',
                          'The model always becomes perfectly calibrated',
                          'The test set becomes part of training',
                          'The number of classes decreases'],
              'answer': 'Optimization may overshoot good solutions and become unstable',
              'explanation': 'A learning rate that is too large can make loss oscillate or diverge.'},
             {'question': 'How can dropout help a neural network?',
              'options': ['It randomly disables some units during training to reduce reliance on specific pathways',
                          'It removes the target variable permanently',
                          'It guarantees zero validation error',
                          'It replaces the activation function with a confusion matrix'],
              'answer': 'It randomly disables some units during training to reduce reliance on specific pathways',
              'explanation': 'Dropout is a regularization method intended to improve generalization.'},
             {'question': 'What is the purpose of L1 or L2 regularization?',
              'options': ['Penalize large or numerous parameter values to control model complexity',
                          'Increase every coefficient without limit',
                          'Use the test set for tuning',
                          'Convert regression into clustering'],
              'answer': 'Penalize large or numerous parameter values to control model complexity',
              'explanation': 'Regularization discourages overly complex fits; L1 can also drive some coefficients toward zero.'},
             {'question': 'Where should hyperparameters normally be selected?',
              'options': ['Using training and validation evidence, not the final test set',
                          'Using the final test set repeatedly',
                          'Using only the largest training score',
                          'After seeing the true future labels in deployment'],
              'answer': 'Using training and validation evidence, not the final test set',
              'explanation': 'The final test set should be reserved for the last unbiased evaluation.'},
             {'question': 'What makes a model comparison fair?',
              'options': ['Models use the same outcome, data split, preprocessing rules, and evaluation metric',
                          'Each model uses a different test set',
                          'The most complex model receives more favorable rows',
                          'Training accuracy is compared with test accuracy from another model'],
              'answer': 'Models use the same outcome, data split, preprocessing rules, and evaluation metric',
              'explanation': 'Holding the evaluation conditions constant isolates differences between models.'},
             {'question': 'What is the main purpose of cross-validation?',
              'options': ['Estimate how performance changes across several training-validation splits',
                          'Train on the final test labels',
                          'Guarantee the model will work in every population',
                          'Remove all uncertainty'],
              'answer': 'Estimate how performance changes across several training-validation splits',
              'explanation': 'Cross-validation provides repeated evidence about average performance and stability.'},
             {'question': 'In five-fold cross-validation, how many times is each fold used for validation?',
              'options': ['Once', 'Five times', 'Never', 'Only when its score is highest'],
              'answer': 'Once',
              'explanation': 'Each of the five folds serves as validation once while the other four are used for training.'},
             {'question': 'Why must scaling, imputation, and feature selection be fitted inside each cross-validation training fold?',
              'options': ['To prevent information from the validation fold leaking into model preparation',
                          'To make every fold identical',
                          'To increase the number of target classes',
                          'To avoid calculating validation metrics'],
              'answer': 'To prevent information from the validation fold leaking into model preparation',
              'explanation': 'Preprocessing learned from all rows would reveal validation information and inflate the score.'},
             {'question': 'What does large variation in cross-validation scores suggest?',
              'options': ['Model performance is sensitive to which observations form the training and validation sets',
                          'The model is guaranteed to generalize',
                          'The target has no variation',
                          'The test error is exactly zero'],
              'answer': 'Model performance is sensitive to which observations form the training and validation sets',
              'explanation': 'High fold-to-fold variation is a warning about instability.'},
             {'question': 'Why is stratified cross-validation useful for classification?',
              'options': ['It helps preserve class proportions in each fold',
                          'It sorts rows by predictor magnitude',
                          'It converts probabilities to regression targets',
                          'It removes minority-class cases'],
              'answer': 'It helps preserve class proportions in each fold',
              'explanation': 'Stratification reduces the chance that a fold has an unrepresentative class balance.'},
             {'question': 'What is nested cross-validation designed to separate?',
              'options': ['Hyperparameter selection from outer performance evaluation',
                          'Regression from classification',
                          'Training rows from predictor columns',
                          'Images from color channels'],
              'answer': 'Hyperparameter selection from outer performance evaluation',
              'explanation': 'An inner loop tunes settings, while an outer loop estimates generalization more honestly.'},
             {'question': 'What does bootstrap sampling with replacement mean?',
              'options': ['A selected row is returned to the pool and may be selected again',
                          'Every row appears exactly once',
                          'Only missing rows are sampled',
                          'Rows are selected in time order'],
              'answer': 'A selected row is returned to the pool and may be selected again',
              'explanation': 'A bootstrap sample can contain repeated rows and omit others.'},
             {'question': 'What does a bootstrap standard error summarize?',
              'options': ['How much an estimate varies across bootstrap resamples',
                          'The number of predictors selected',
                          'The classification threshold',
                          'The image resolution'],
              'answer': 'How much an estimate varies across bootstrap resamples',
              'explanation': 'The spread of bootstrap estimates approximates sampling uncertainty.'},
             {'question': 'Which interpretation of a 95 percent confidence interval is most appropriate?',
              'options': ['It gives a range produced by a method intended to cover the true parameter in about 95 percent of repeated samples',
                          'There is a 95 percent chance that every individual observation lies inside it',
                          'The model has 95 percent accuracy',
                          'The parameter changes randomly after the interval is calculated'],
              'answer': 'It gives a range produced by a method intended to cover the true parameter in about 95 percent of repeated samples',
              'explanation': 'The repeated-sampling interpretation concerns the interval procedure, not 95 percent of individual observations.'},
             {'question': 'What important problem cannot bootstrap resampling automatically fix?',
              'options': ['A biased or unrepresentative original dataset',
                          'Sampling variability',
                          'Repeated rows in resamples',
                          'The need to summarize uncertainty'],
              'answer': 'A biased or unrepresentative original dataset',
              'explanation': 'Bootstrap resamples the observed data; it cannot create populations or information absent from that data.'},
             {'question': 'What is a lag in time-series forecasting?',
              'options': ['A past value used as a predictor of a later value',
                          'A future target used during training',
                          'A random class label',
                          'A bootstrap sample size'],
              'answer': 'A past value used as a predictor of a later value',
              'explanation': 'Lagged values provide historical information available before the forecast time.'},
             {'question': 'What is a forecast horizon?',
              'options': ['How far into the future the prediction is made',
                          'The number of classes in the target',
                          'The width of the training table',
                          'The number of image channels'],
              'answer': 'How far into the future the prediction is made',
              'explanation': 'Examples include one hour ahead, one day ahead, or twelve months ahead.'},
             {'question': 'Why should forecasting use a time-ordered split?',
              'options': ['Training must use earlier observations and evaluation must use later observations',
                          'Random order always improves realism',
                          'The test period should occur before training',
                          'Time order matters only for images'],
              'answer': 'Training must use earlier observations and evaluation must use later observations',
              'explanation': 'A time-ordered split reproduces the real direction of prediction.'},
             {'question': 'Which situation is time-series leakage?',
              'options': ['Using a future target value or future-derived feature to predict an earlier time',
                          "Using yesterday's value to predict today",
                          'Comparing with a naïve forecast',
                          'Reporting a forecast horizon'],
              'answer': 'Using a future target value or future-derived feature to predict an earlier time',
              'explanation': 'Future information would not be available at the forecast origin and makes evaluation unrealistic.'},
             {'question': 'Why should a forecasting model be compared with a naïve baseline?',
              'options': ['A complex model is useful only if it improves on a simple realistic forecast',
                          'The baseline guarantees perfect accuracy',
                          'The baseline removes seasonality from the data',
                          'A baseline is required only for classification'],
              'answer': 'A complex model is useful only if it improves on a simple realistic forecast',
              'explanation': 'A naïve forecast establishes the minimum useful standard.'},
             {'question': 'What is seasonality?',
              'options': ['A pattern that repeats at regular time intervals',
                          'A random train-test split',
                          'A class imbalance problem',
                          'A coefficient penalty'],
              'answer': 'A pattern that repeats at regular time intervals',
              'explanation': 'Daily, weekly, and yearly cycles are common seasonal patterns.'},
             {'question': 'What is the lookback window in an LSTM forecast?',
              'options': ['The number of past time steps supplied as the input sequence',
                          'The number of future labels used for training',
                          'The number of classes',
                          'The image width'],
              'answer': 'The number of past time steps supplied as the input sequence',
              'explanation': 'The lookback controls how much history the recurrent network sees for each prediction.'},
             {'question': 'Why do forecast intervals often widen at longer horizons?',
              'options': ['Less direct information is available and uncertainty can accumulate farther into the future',
                          'The target automatically becomes categorical',
                          'Longer horizons use more test labels',
                          'Root mean squared error becomes zero'],
              'answer': 'Less direct information is available and uncertainty can accumulate farther into the future',
              'explanation': 'Farther-ahead predictions usually face greater uncertainty.'},
             {'question': 'What is a pixel?',
              'options': ['A numerical image element at a particular row and column',
                          'A complete image class',
                          'A regression coefficient',
                          'A cross-validation fold'],
              'answer': 'A numerical image element at a particular row and column',
              'explanation': 'Images are arrays of pixel values.'},
             {'question': 'What are red, green, and blue channels?',
              'options': ['Separate numerical layers that combine to represent color',
                          'Three target classes required for every image problem',
                          'Three validation folds',
                          'Three regression residuals'],
              'answer': 'Separate numerical layers that combine to represent color',
              'explanation': 'A color image commonly stores intensity values in three channels.'},
             {'question': 'What does a convolutional neural network learn especially well?',
              'options': ['Local visual patterns such as edges, textures, and shapes',
                          'Only global averages with no spatial information',
                          'Spreadsheet formulas',
                          'Bootstrap sampling rules'],
              'answer': 'Local visual patterns such as edges, textures, and shapes',
              'explanation': 'Convolutional filters exploit the spatial structure of images.'},
             {'question': 'What is transfer learning in computer vision?',
              'options': ['Starting from a model pretrained on a large dataset and adapting it to a new task',
                          'Moving test images into the training folder',
                          'Copying class labels between projects',
                          'Training every model from random weights only'],
              'answer': 'Starting from a model pretrained on a large dataset and adapting it to a new task',
              'explanation': 'Pretrained visual features can reduce the data and computation required for a new image task.'},
             {'question': 'What is data augmentation?',
              'options': ['Creating realistic transformed training images, such as small rotations or flips',
                          'Adding test labels to training',
                          'Deleting difficult classes',
                          'Increasing image confidence manually'],
              'answer': 'Creating realistic transformed training images, such as small rotations or flips',
              'explanation': 'Augmentation exposes the model to plausible variation and can reduce overfitting.'},
             {'question': 'What can an image-classification confusion matrix reveal?',
              'options': ['Which image classes are most often confused with one another',
                          'Only the number of pixels',
                          'The learning rate',
                          'The forecast horizon'],
              'answer': 'Which image classes are most often confused with one another',
              'explanation': 'Off-diagonal cells show specific class-to-class errors.'},
             {'question': 'What is the best evidence that an AI model generalizes?',
              'options': ['Strong performance on genuinely unseen, representative data',
                          'Perfect training accuracy alone',
                          'A large number of parameters',
                          'A visually attractive interface'],
              'answer': 'Strong performance on genuinely unseen, representative data',
              'explanation': 'Generalization concerns new data from the intended use setting.'},
             {'question': 'What is responsible AI practice?',
              'options': ['Evaluate performance, bias, limitations, privacy, and human consequences before and after deployment',
                          'Deploy the model whenever training accuracy is high',
                          'Hide uncertainty from users',
                          'Assume one model works equally well for every group'],
              'answer': 'Evaluate performance, bias, limitations, privacy, and human consequences before and after deployment',
              'explanation': 'Responsible use requires technical evaluation together with attention to people, context, and oversight.'}]}


def wrap_question_limit(week):
    """Return the intended question count for each weekly Wrap-Up."""
    return 50 if week in {"Week 8", "Week 16"} else 10


def week_wrap_questions(week, brief=None, df=None):
    """Return the weekly question set, including 50-question review banks."""
    if week in REVIEW_WRAP_UP_QUESTIONS:
        questions = [dict(item) for item in REVIEW_WRAP_UP_QUESTIONS[week]]
    else:
        questions = [dict(item) for item in WEEK_WRAP_UP_QUESTIONS.get(week, [])]
        questions.extend(dict(item) for item in WEEK_WRAP_UP_EXTRA_QUESTIONS.get(week, []))
        brief = brief or {}
        targets = [v for v in brief.get("targets", []) if v]
        predictors = [v for v in brief.get("predictors", []) if v]
        dataset_question = None
        if targets:
            target = targets[0]
            distractors = [v for v in predictors if v != target]
            if df is not None:
                distractors += [c for c in df.columns if c != target and c not in distractors]
            raw_options = [target] + distractors[:2]
            display_options = [humanize(v) for v in raw_options]
            if len(set(display_options)) >= 2:
                dataset_question = {
                    "question": "Which variable is the target or main outcome in today's class plan?",
                    "options": display_options,
                    "answer": humanize(target),
                    "explanation": f"{humanize(target)} is the outcome selected in the instructor's class brief.",
                }
        if dataset_question:
            questions = questions[:9] + [dataset_question]
        else:
            questions = questions[:10]

    expected_total = wrap_question_limit(week)
    questions = questions[:expected_total]

    # Shuffle option positions deterministically so the correct answer is not always first.
    for item in questions:
        options = list(dict.fromkeys(item.get("options", [])))
        seed = int(hashlib.sha256(f"{week}|{item.get('question', '')}".encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        if len(options) > 1:
            options = [options[i] for i in rng.permutation(len(options))]
        item["options"] = options
    return questions





def variable_language(week):
    if week in {"Week 3", "Week 4"}:
        return "Main variable or outcome", "Other variables to examine"
    if week in {"Week 5", "Week 6", "Week 7"}:
        return "Numerical target", "Predictors"
    if week in {"Week 9", "Week 10"}:
        return "Class target", "Predictors"
    if week == "Week 14":
        return "Forecast target", "Lagged or additional predictors"
    if week in {"Week 11", "Week 12", "Week 13"}:
        return "Target or quantity of interest", "Predictors or variables"
    return "Target or outcome", "Variables or possible predictors"


def suggested_research_question(week, target="", predictors=None):
    predictors = [humanize(v).lower() for v in (predictors or []) if v and v != target]
    target_text = humanize(target).lower() if target else "the selected outcome"
    predictor_text = ", ".join(predictors[:-1]) + (" and " + predictors[-1] if len(predictors) > 1 else (predictors[0] if predictors else "the selected information"))
    if week == "Week 1":
        return "How does experimental probability compare with theoretical probability as the number of trials increases?"
    if week == "Week 2":
        return f"How might {predictor_text} help us understand or predict {target_text}?"
    if week == "Week 3":
        return f"What patterns, unusual values, and group differences can we observe in {target_text} and the selected variables?"
    if week == "Week 4":
        first = predictors[0] if predictors else "the selected variable"
        return f"What is the relationship between {first} and {target_text}?"
    if week in {"Week 5", "Week 6", "Week 7"}:
        return f"How well can {predictor_text} explain or predict {target_text}?"
    if week in {"Week 9", "Week 10"}:
        return f"How well can {predictor_text} predict the class of {target_text}?"
    if week == "Week 11":
        return f"Which model gives the most useful held-out performance for {target_text}?"
    if week == "Week 12":
        return f"How stable is model performance for {target_text} across cross-validation folds?"
    if week == "Week 13":
        return f"How uncertain is the selected estimate or prediction for {target_text}?"
    if week == "Week 14":
        extras = f" and {predictor_text}" if predictors else ""
        return f"How accurately can previous values of {target_text}{extras} forecast future {target_text}?"
    if week == "Week 15":
        return "How accurately can an image classifier identify the assigned image classes?"
    if week == "Week 8":
        return "How well can we explain and apply the main ideas from Weeks 1 to 7?"
    if week == "Week 16":
        return "How well can we connect classification, validation, uncertainty, forecasting, neural networks, and computer vision?"
    return "How can the selected method help answer the assigned data question?"


def read_uploaded_table(uploaded):
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)


def dataset_manager(scope, title, expanded=False):
    frame, current_name = scope_dataset(scope)
    with st.expander(title, expanded=expanded):
        st.caption(f"Current dataset: {current_name} · {len(frame):,} rows · {frame.shape[1]} columns")
        source = st.radio("Choose data source", ["Demo dataset", "Upload CSV or Excel"], horizontal=True, key=f"{scope}_manager_source")
        if source == "Demo dataset":
            name = st.selectbox("Demo dataset", list(DEMO_DATASETS), key=f"{scope}_manager_demo")
            if st.button("Use this demo", key=f"{scope}_manager_demo_button"):
                set_scope_dataset(scope, DEMO_DATASETS[name](), name)
                st.success(f"Loaded {name}.")
                st.rerun()
        else:
            uploaded = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"], key=f"{scope}_manager_upload")
            if uploaded is not None and st.button("Use this uploaded dataset", key=f"{scope}_manager_upload_button"):
                try:
                    set_scope_dataset(scope, read_uploaded_table(uploaded), uploaded.name)
                    st.success(f"Loaded {uploaded.name}.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not load the dataset: {exc}")
    return scope_dataset(scope)


def brief_problem(brief, df):
    non_tabular_week = brief.get("week") in {"Week 1", "Week 8", "Week 15", "Week 16"}
    missing = list(brief.get("missing_targets", [])) + list(brief.get("missing_predictors", []))
    if missing and not non_tabular_week:
        return "The saved class brief uses variables that are not in the current class dataset: " + ", ".join(map(humanize, missing))
    if not brief.get("dataset_matches", True) and not non_tabular_week:
        return "The class dataset changed after this brief was prepared. Open Instructor Setup and confirm the variables again."
    if not brief.get("research_question", "").strip():
        return "The instructor has not yet saved today's research question."
    return ""



def render_weekly_journey(week, active_stage):
    """Display the same five-stage rhythm everywhere in the guided course."""
    project = ensure_project_state()
    setup_done = week in st.session_state.get("lab_briefs", {}) and bool(st.session_state.lab_briefs.get(week, {}).get("research_question", "").strip())
    statuses = {
        "Instructor Setup": setup_done,
        "Today's Lab": bool(project.get("today_progress", {}).get(week)),
        "Practical Studio": bool(project.get("practical_progress", {}).get(week, {}).get("completed")),
        "Wrap-Up": bool(project.get("wrap_up_progress", {}).get(week, {}).get("completed")),
        "My Notebook": week in project.get("weeks", {}),
    }
    st.caption("Weekly rhythm: instructor prepares → students learn → practise → reflect → apply independently")
    columns = st.columns(5)
    labels = [
        ("Instructor Setup", "Prepare"),
        ("Today's Lab", "Learn"),
        ("Practical Studio", "Practise"),
        ("Wrap-Up", "Reflect"),
        ("My Notebook", "Homework"),
    ]
    for column, (stage, action) in zip(columns, labels):
        symbol = "✅" if statuses[stage] else ("▶️" if stage == active_stage else "○")
        emphasis = "**" if stage == active_stage else ""
        column.markdown(f"{symbol} {emphasis}{stage}{emphasis}<br><small>{action}</small>", unsafe_allow_html=True)
    st.caption("The scheduled class ends after Wrap-Up. My Notebook is the independent assignment.")


def week1_probability_practice(context="practice"):
    """A deliberately small Week 1 simulator that also teaches students how to use the app."""
    prefix = f"v66_week1_probability_{context}"
    st.markdown("### Probability experiment: repeated coin tosses")
    st.markdown(
        '<div class="simple-note"><strong>Question</strong><br>'
        'Does the observed proportion of heads move closer to the theoretical probability when we increase the number of trials?</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    probability = c1.slider("Theoretical probability of heads", 0.05, 0.95, 0.50, 0.05, key=f"{prefix}_p")
    trials = c2.slider("Number of tosses", 10, 2000, 100, 10, key=f"{prefix}_n")
    seed = c3.number_input("Simulation seed", min_value=0, max_value=99999, value=42, step=1, key=f"{prefix}_seed")
    rng = np.random.default_rng(int(seed))
    outcomes = rng.binomial(1, probability, int(trials))
    cumulative = np.cumsum(outcomes) / np.arange(1, int(trials) + 1)
    observed = float(cumulative[-1])
    heads = int(outcomes.sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Heads observed", heads)
    c2.metric("Experimental probability", f"{observed:.3f}")
    c3.metric("Theoretical probability", f"{probability:.3f}")
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(np.arange(1, int(trials) + 1), cumulative, label="Experimental probability")
    ax.axhline(probability, linestyle="--", label="Theoretical probability")
    ax.set_xlabel("Number of tosses")
    ax.set_ylabel("Probability of heads")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    difference = abs(observed - probability)
    st.markdown(
        f'<div class="success-note"><strong>Read the result</strong><br>'
        f'After {trials} tosses, the observed proportion was {observed:.3f}. '
        f'It was {difference:.3f} away from the theoretical probability of {probability:.3f}. '
        'Run the experiment with different trial counts and notice that small samples usually fluctuate more.</div>',
        unsafe_allow_html=True,
    )
    st.session_state[f"{prefix}_result"] = {
        "trials": int(trials),
        "probability": float(probability),
        "observed": observed,
        "heads": heads,
        "difference": difference,
    }
    return st.session_state[f"{prefix}_result"]

def instructor_setup_page_v65(week):
    activate_dataset_scope("class")
    st.title("🧑🏾‍🏫 Instructor Setup")
    render_weekly_journey(week, "Instructor Setup")
    st.markdown('<div class="simple-note"><strong>Prepare one clear lesson.</strong><br>Choose the class data, research question, variables, student freedom, practical task, and required outputs.</div>', unsafe_allow_html=True)
    lab = WEEKLY_LABS[week]
    if week == "Week 1":
        st.info("Week 1 uses the built-in probability simulator, so no dataset upload is required for the class practical.")
    else:
        dataset_manager("class", "1. Choose the class dataset", expanded=True)
    df = get_df()
    old = st.session_state.lab_briefs.get(week, default_brief(week, df))
    st.subheader(f"2. Prepare {week}: {lab['title']}")
    target_label, predictor_label = variable_language(week)
    if week in {"Week 1", "Week 8", "Week 15", "Week 16"}:
        target = ""
        predictors = []
        if week == "Week 1":
            st.info("Week 1 uses a probability simulation rather than a tabular target-and-predictor model. Students learn the app rhythm while comparing theoretical and experimental probability.")
        else:
            st.info("This week does not require a tabular target-and-predictor setup. The class dataset may still be used for orientation or review.")
    else:
        targets = suitable_targets(week, df)
        target_choices = [""] + targets
        old_target = next((v for v in old.get("targets", []) if v in targets), "")
        target = st.selectbox(target_label, target_choices, index=target_choices.index(old_target), format_func=lambda v: "— Select —" if v == "" else humanize(v), key=f"v65_inst_target_{week}")
        predictor_options = [c for c in df.columns if c != target]
        predictors = st.multiselect(
            predictor_label,
            predictor_options,
            default=[c for c in old.get("predictors", []) if c in predictor_options],
            format_func=humanize,
            key=f"v65_inst_predictors_{week}",
        )
    suggestion = suggested_research_question(week, target, predictors)
    question_key = f"v65_inst_question_{week}"
    if question_key not in st.session_state:
        st.session_state[question_key] = old.get("research_question", "")
    if st.button("Use the suggested research question", key=f"v65_use_suggestion_{week}"):
        st.session_state[question_key] = suggestion
        st.rerun()
    st.caption("Suggested question: " + suggestion)
    question = st.text_area("Question we will answer today", key=question_key, placeholder="Write or generate one clear question using the current dataset.")
    choice_options = ["Use the instructor's exact variables", "Choose from instructor-approved variables", "Choose any suitable variables"]
    choice_mode = st.radio(
        "How much choice should students have in the practical?",
        choice_options,
        index=choice_options.index(old.get("choice_mode", choice_options[0])) if old.get("choice_mode", choice_options[0]) in choice_options else 0,
        key=f"v65_inst_choice_{week}",
    )
    class_example = st.text_area(
        "Opening question to ask the class",
        value=old.get("class_example", ""),
        placeholder="Example: Before we calculate anything, what pattern do you expect and why?",
        key=f"v65_inst_opening_{week}",
    )
    instructions = st.text_area("Today's practical task", value=old.get("instructions", lab["assignment"]), key=f"v65_inst_task_{week}")
    output_options = sorted(set(lab["required"] + old.get("required_outputs", [])))
    required_outputs = st.multiselect("Students must report", output_options, default=[v for v in old.get("required_outputs", lab["required"]) if v in output_options], key=f"v65_inst_outputs_{week}")
    duration = st.slider("Class time in minutes", 20, 120, int(old.get("duration", 60)), 5, key=f"v65_inst_duration_{week}")
    if st.button("Save and publish today's class brief", type="primary", key=f"v65_inst_save_{week}"):
        needs_target = week not in {"Week 1", "Week 8", "Week 15", "Week 16"}
        if not question.strip():
            st.warning("Write or generate the research question.")
        elif needs_target and not target:
            st.warning("Choose a target or main outcome for this week.")
        elif week in {"Week 5", "Week 6", "Week 7", "Week 9", "Week 10", "Week 11", "Week 12"} and not predictors:
            st.warning("Choose at least one predictor.")
        else:
            st.session_state.lab_briefs[week] = {
                "week": week,
                "research_question": question.strip(),
                "targets": [target] if target else [],
                "predictors": predictors,
                "choice_mode": choice_mode,
                "class_example": class_example.strip(),
                "instructions": instructions.strip(),
                "required_outputs": required_outputs,
                "duration": duration,
                "dataset_name": active_dataset_name(),
                "dataset_signature": dataframe_signature(df),
            }
            course_state = ensure_project_state()
            course_state["practice_plans"].pop(week, None)
            course_state["today_progress"].pop(week, None)
            course_state["practical_progress"].pop(week, None)
            course_state["wrap_up_progress"].pop(week, None)
            course_state["wrap_up_attempts"].pop(week, None)
            st.success("Today's Lab, Practical Studio, and the new Wrap-Up are ready.")
            st.rerun()
    current = get_lab_brief(week, df)
    st.subheader("Student preview")
    st.markdown(f'<div class="question-card"><span class="tiny-label">Research question</span><br><strong>{current.get("research_question") or "Not yet configured"}</strong></div>', unsafe_allow_html=True)
    st.write(f"**Class dataset:** {active_dataset_name()}")
    st.write(f"**{target_label}:** {', '.join(map(humanize, current.get('targets', []))) or 'Not selected'}")
    st.write(f"**{predictor_label}:** {', '.join(map(humanize, current.get('predictors', []))) or 'Not selected or not required'}")
    st.write(f"**Practical task:** {current.get('instructions', '')}")
    with st.expander("Preview the automatic Wrap-Up questions", expanded=False):
        preview_questions = week_wrap_questions(week, current, df)
        st.caption(
            f"Students will answer {len(preview_questions)} questions one at a time. "
            "Answer choices and correct answers remain hidden here so the quiz still requires independent thinking."
        )
        for number, item in enumerate(preview_questions, 1):
            st.markdown(f"**{number}. {item['question']}**")
    st.download_button("Download this class brief", json.dumps(json_safe(current), indent=2), f"MATH490_{week.replace(' ', '_')}_class_brief.json", "application/json", key=f"v65_brief_download_{week}")
    imported = st.file_uploader("Import a saved class brief", type=["json"], key=f"v65_brief_upload_{week}")
    if imported is not None and st.button("Import class brief", key=f"v65_brief_import_{week}"):
        try:
            loaded = json.loads(imported.getvalue().decode("utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("Invalid brief")
            loaded["week"] = week
            st.session_state.lab_briefs[week] = loaded
            st.success("Class brief imported. Confirm that its variables match the current dataset.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not import the brief: {exc}")


def todays_lab_page(week):
    activate_dataset_scope("class")
    df = get_df()
    lab = WEEKLY_LABS[week]
    brief = get_lab_brief(week, df)
    st.title(f"🎓 Today's Lab — {week}")
    render_weekly_journey(week, "Today's Lab")
    st.markdown(f"### {lab['title']}")
    problem = brief_problem(brief, df)
    if problem:
        st.markdown(f'<div class="warning-card"><strong>Today’s lesson is not ready.</strong><br>{problem}</div>', unsafe_allow_html=True)
        st.info("Open Instructor Setup, choose the current dataset and variables, then save the class brief.")
        return
    target_label, predictor_label = variable_language(week)
    target = brief.get("targets", [""])[0] if brief.get("targets") else ""
    predictors = brief.get("predictors", [])
    st.markdown(f'<div class="question-card"><span class="tiny-label">Today’s research question</span><br><strong>{brief["research_question"]}</strong></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Activity" if week == "Week 1" else "Class dataset", "Probability simulation" if week == "Week 1" else brief.get("dataset_name", active_dataset_name()))
    c2.metric(target_label, humanize(target) if target else "Not required")
    c3.metric("Class time", f"{brief.get('duration', 60)} minutes")
    step = st.radio("Lesson", ["1. Learn the idea", "2. See the research plan", "3. Understand the outputs", "4. Check your understanding"], horizontal=True, key=f"v65_today_step_{week}")
    st.divider()
    if step.startswith("1"):
        st.markdown(f'<div class="simple-note"><strong>Key idea</strong><br>{lab["key_idea"]}</div>', unsafe_allow_html=True)
        st.markdown("### Words to know")
        for term, meaning in lab["terms"].items():
            st.markdown(f"**{term}:** {meaning}")
        st.warning("Common mistake: " + lab["mistake"])
        if brief.get("class_example"):
            st.info("Think first: " + brief["class_example"])
    elif step.startswith("2"):
        st.markdown("### Our plan for today")
        st.write(f"**Dataset:** {brief.get('dataset_name', active_dataset_name())}")
        st.write(f"**{target_label}:** {humanize(target) if target else 'Not required'}")
        st.write(f"**{predictor_label}:** {', '.join(map(humanize, predictors)) or 'Not required for this activity'}")
        st.write(f"**Method:** {lab['title']}")
        st.write(f"**Practical task:** {brief.get('instructions', lab['assignment'])}")
        st.markdown("### We will do this in order")
        for index, item in enumerate(lab["steps"], 1):
            st.markdown(f'<div class="lesson-card"><span class="lesson-number">{index}</span><strong>{item}</strong></div>', unsafe_allow_html=True)
    elif step.startswith("3"):
        st.markdown("### How to read today’s results")
        for name, meaning in WEEK_OUTPUT_GUIDES[week]:
            st.markdown(f'<div class="lesson-card"><strong>{name}</strong><br>{meaning}</div>', unsafe_allow_html=True)
        if brief.get("required_outputs"):
            st.markdown("### What students must report")
            for item in brief["required_outputs"]:
                st.markdown(f"- {item}")
        st.caption("The app will explain the actual numbers after the analysis. Students must still write the meaning in their own words.")
    else:
        question, options, correct = QUICK_CHECKS[week]
        answer = st.radio(question, options, index=None, key=f"v65_today_check_{week}")
        if answer:
            if answer == correct:
                st.success("Correct. You are ready for the practical.")
                ensure_project_state()["today_progress"][week] = True
            else:
                st.error("Not quite. Re-read the key idea and try again.")
        if ensure_project_state()["today_progress"].get(week):
            st.markdown('<div class="success-note"><strong>Today’s lesson is complete.</strong><br>Next, open Practical Studio and follow the class activity.</div>', unsafe_allow_html=True)


def choose_plan_variables(week, df, brief, storage_key, prefix):
    project = ensure_project_state()
    storage = project.setdefault(storage_key, {})
    saved = storage.get(week, {})
    if week in {"Week 1", "Week 8", "Week 15", "Week 16"}:
        question = brief.get("research_question", "").strip() or suggested_research_question(week)
        st.text_area("Research question or learning focus", value=question, disabled=True, key=f"{prefix}_question_locked_{week}")
        return question, "", []

    choice_mode = brief.get("choice_mode", "Use the instructor's exact variables")
    exploratory_week = week in {"Week 3", "Week 4"}
    target_options = suitable_targets(week, df)
    approved_targets = [c for c in brief.get("targets", []) if c in target_options]

    if choice_mode == "Use the instructor's exact variables" and approved_targets and not exploratory_week:
        available_targets = approved_targets[:1]
        target_disabled = True
    elif choice_mode == "Choose from instructor-approved variables" and approved_targets and not exploratory_week:
        available_targets = approved_targets
        target_disabled = False
    else:
        available_targets = target_options
        target_disabled = False
    if not available_targets:
        available_targets = list(df.columns)

    preferred_target = saved.get("target")
    if preferred_target not in available_targets:
        preferred_target = approved_targets[0] if approved_targets and approved_targets[0] in available_targets else (available_targets[0] if available_targets else "")
    target_label, predictor_label = variable_language(week)
    target = (
        st.selectbox(
            target_label,
            available_targets,
            index=available_targets.index(preferred_target),
            format_func=humanize,
            disabled=target_disabled,
            key=f"{prefix}_target_{week}",
        )
        if available_targets else ""
    )

    all_predictors = [c for c in df.columns if c != target]
    approved_predictors = [c for c in brief.get("predictors", []) if c in all_predictors]
    if choice_mode == "Use the instructor's exact variables" and not exploratory_week:
        predictor_options = approved_predictors or all_predictors
        defaults = approved_predictors
        disabled = bool(approved_predictors)
    elif choice_mode == "Choose from instructor-approved variables" and approved_predictors and not exploratory_week:
        predictor_options = approved_predictors
        defaults = [c for c in saved.get("predictors", []) if c in predictor_options] or approved_predictors[: min(3, len(approved_predictors))]
        disabled = False
    else:
        predictor_options = all_predictors
        defaults = [c for c in saved.get("predictors", []) if c in predictor_options]
        if not defaults:
            defaults = approved_predictors or default_predictors_for_target(df, target, 4)
        disabled = False

    predictors = st.multiselect(
        predictor_label,
        predictor_options,
        default=defaults,
        format_func=humanize,
        disabled=disabled,
        key=f"{prefix}_predictors_{week}",
    )
    if exploratory_week:
        st.caption("The instructor's variables are a starting point. You may freely change the variables and plots during this exploratory practical.")

    class_question = brief.get("research_question", "").strip()
    if choice_mode == "Use the instructor's exact variables" and class_question:
        question = class_question
        st.text_area("Research question", value=question, disabled=True, key=f"{prefix}_question_locked_{week}")
    else:
        suggestion = suggested_research_question(week, target, predictors)
        question = st.text_area("Research question", value=saved.get("question", suggestion), key=f"{prefix}_question_{week}")
        st.caption("Suggested from your current variables: " + suggestion)
    return question.strip(), target, predictors

def practical_studio_page(week):
    activate_dataset_scope("class")
    df = get_df()
    lab = WEEKLY_LABS[week]
    brief = get_lab_brief(week, df)
    st.title(f"🧪 Practical Studio — {week}")
    render_weekly_journey(week, "Practical Studio")
    st.markdown("**This is the hands-on class activity.** Follow the instructor's plan, run the method, and explain one result.")
    if not ensure_project_state().get("today_progress", {}).get(week):
        st.warning("Recommended sequence: complete Today's Lab first so the practical has a clear purpose.")
    problem = brief_problem(brief, df)
    if problem:
        st.markdown(f'<div class="warning-card"><strong>The practical is not ready.</strong><br>{problem}</div>', unsafe_allow_html=True)
        return
    st.markdown(f'<div class="question-card"><span class="tiny-label">Class research question</span><br><strong>{brief["research_question"]}</strong></div>', unsafe_allow_html=True)
    question, target, predictors = choose_plan_variables(week, df, brief, "practice_plans", "v68_practice")
    project = ensure_project_state()
    if st.button("Confirm the practical plan", type="primary", key=f"v68_practice_confirm_{week}"):
        project["practice_plans"][week] = {
            "question": question,
            "target": target,
            "predictors": predictors,
            "dataset_name": active_dataset_name(),
            "dataset_signature": dataframe_signature(df),
        }
        apply_plan_to_tool(week, project["practice_plans"][week], context="class_practical", force=True)
        st.success("Plan confirmed. The practical tool is ready below.")
        st.rerun()

    plan = project["practice_plans"].get(week)
    if not plan:
        st.info("Confirm the practical plan before opening the analytical tool.")
        return
    if plan.get("dataset_signature") != dataframe_signature(df):
        st.warning("The class dataset changed. Confirm the practical plan again.")
        return

    apply_plan_to_tool(week, plan, context="class_practical")
    st.divider()
    st.markdown("### Complete today’s analysis")
    if week in {"Week 3", "Week 4"}:
        st.info("The class plan supplies a starting point. Freely change the x-variable, y-variable, chart, grouping, or association controls; your choices will remain in place while you work.")
    else:
        st.info("The class plan loads the starting variables once. You may change any control that the instructor has left unlocked; the app will not force it back after each selection.")

    if week == "Week 1":
        week1_probability_practice("practice")
    elif lab.get("tool"):
        render_tool_page(lab["tool"])
    else:
        for item in lab["steps"]:
            st.checkbox(item, key=f"v68_practice_review_{week}_{item}")

    result = week_result(week)
    notes = simple_result_notes(result)
    if notes:
        with st.expander("Explain the result in simple words", expanded=True):
            for note in notes:
                st.markdown("- " + note)

    previous = project.get("practical_progress", {}).get(week, {})
    reflection = st.text_area(
        "One sentence I learned from the practical",
        value=previous.get("reflection", ""),
        placeholder="Write one result or lesson in your own words.",
        key=f"v68_practice_reflection_{week}",
    )
    if st.button("Mark the practical complete", key=f"v68_practical_complete_{week}"):
        if not reflection.strip():
            st.warning("Write one sentence about what you learned.")
        else:
            project["practical_progress"][week] = {
                "completed": True,
                "reflection": reflection.strip(),
                "question": plan.get("question", ""),
                "dataset_name": plan.get("dataset_name", active_dataset_name()),
                "target": plan.get("target", ""),
                "predictors": plan.get("predictors", []),
                "result_notes": notes,
            }
            st.success("Practical complete. Open Wrap-Up to finish today's class.")

def _new_wrap_attempt(week, questions, attempt_number=1):
    signature = hashlib.sha256(
        json.dumps([q.get("question", "") for q in questions], ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "attempt_number": int(attempt_number),
        "question_signature": signature,
        "index": 0,
        "score": 0,
        "responses": [],
        "feedback_pending": False,
        "complete": False,
    }


def _wrap_performance_message(score, total):
    if total <= 0:
        return "No score available."
    fraction = score / total
    if fraction >= 0.9:
        return "Excellent understanding."
    if fraction >= 0.7:
        return "Good understanding. Review the explanations for the questions you missed."
    return "More practice is recommended. Revisit Today's Lab and the Practical Studio, then try again."


def _go_to_guided_space(space):
    st.session_state["v65_space"] = space


def wrap_up_page(week):
    activate_dataset_scope("class")
    df = get_df()
    lab = WEEKLY_LABS[week]
    brief = get_lab_brief(week, df)
    project = ensure_project_state()
    st.title(f"✅ Wrap-Up — {week}")
    render_weekly_journey(week, "Wrap-Up")
    st.markdown(f"### Finish the class: {lab['title']}")
    intended_total = wrap_question_limit(week)
    review_wording = "comprehensive review questions" if intended_total > 10 else "questions"
    st.markdown(
        f'<div class="simple-note"><strong>Reflect before leaving class.</strong><br>'
        f'You will answer {intended_total} {review_wording}, one at a time. '
        'After each response, the app will tell you whether it is correct and explain why.</div>',
        unsafe_allow_html=True,
    )
    problem = brief_problem(brief, df)
    if problem:
        st.markdown(f'<div class="warning-card"><strong>The Wrap-Up is not ready.</strong><br>{problem}</div>', unsafe_allow_html=True)
        return
    practical = project.get("practical_progress", {}).get(week, {})
    if not practical.get("completed"):
        st.warning("The questions are locked because the Practical Studio has not yet been marked complete.")
        st.info("Open Practical Studio, complete the activity, write the one-sentence reflection, and select **Mark the practical complete**. Then return here.")
        st.button(
            "Return to Practical Studio",
            type="primary",
            on_click=_go_to_guided_space,
            args=("Practical Studio",),
            key=f"v67_return_practical_{week}",
        )
        return
    st.markdown(f'<div class="question-card"><span class="tiny-label">Question studied today</span><br><strong>{brief["research_question"]}</strong></div>', unsafe_allow_html=True)
    questions = week_wrap_questions(week, brief, df)
    total = len(questions)
    attempts = project.setdefault("wrap_up_attempts", {})
    saved = project.get("wrap_up_progress", {}).get(week)
    attempt = attempts.get(week)
    expected_signature = hashlib.sha256(
        json.dumps([q.get("question", "") for q in questions], ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    if attempt and attempt.get("question_signature") != expected_signature:
        attempt = None
        attempts.pop(week, None)
    if attempt is None and not saved:
        attempt = _new_wrap_attempt(week, questions, 1)
        attempts[week] = attempt

    # Completed attempt or imported completed record: show the score and review.
    completed_responses = None
    if attempt and attempt.get("complete"):
        completed_responses = attempt.get("responses", [])
    elif saved and not attempt:
        completed_responses = saved.get("responses", [])
    if completed_responses is not None:
        score = int((attempt or saved).get("score", 0))
        latest_total = int((attempt or saved).get("total", total))
        best_score = int((saved or {}).get("best_score", score))
        attempt_count = int((saved or {}).get("attempts", (attempt or {}).get("attempt_number", 1)))
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest score", f"{score}/{latest_total}")
        c2.metric("Best score", f"{best_score}/{latest_total}")
        c3.metric("Attempts", attempt_count)
        st.markdown(f'<div class="success-note"><strong>{_wrap_performance_message(score, latest_total)}</strong><br>Your class Wrap-Up attempt is complete.</div>', unsafe_allow_html=True)
        with st.expander("Review my answers and explanations", expanded=score < latest_total):
            if completed_responses:
                for number, response in enumerate(completed_responses, 1):
                    if response.get("correct"):
                        st.success(f"{number}. Correct — {response.get('selected', '')}")
                    else:
                        st.error(f"{number}. Your answer: {response.get('selected', 'Not recorded')}")
                        st.markdown(f"**Correct answer:** {response.get('answer', '')}")
                    st.caption(response.get("explanation", ""))
            else:
                st.info("This score came from an earlier app version, so detailed question-by-question responses are unavailable.")
        if st.button(f"Try the {latest_total} questions again", key=f"v67_wrap_retry_{week}"):
            next_number = int((attempt or {}).get("attempt_number", attempt_count)) + 1
            attempts[week] = _new_wrap_attempt(week, questions, next_number)
            st.rerun()
        st.markdown(
            '<div class="success-note"><strong>Today’s class is complete.</strong><br>'
            'My Notebook is the independent assignment. You can begin it now or complete it before the due date set in Canvas.</div>',
            unsafe_allow_html=True,
        )
        return

    attempt = attempts[week]
    index = int(attempt.get("index", 0))
    if index >= total:
        attempt["complete"] = True
        st.rerun()
    item = questions[index]
    st.progress(index / total if total else 0.0)
    answered_count = len(attempt.get("responses", []))
    score_text = f"Score so far: {int(attempt.get('score', 0))}/{answered_count}" if answered_count else "No questions answered yet"
    st.caption(f"Question {index + 1} of {total} · {score_text}")
    st.markdown(f"### {item['question']}")

    if not attempt.get("feedback_pending"):
        with st.form(f"v67_wrap_question_form_{week}_{attempt['attempt_number']}_{index}"):
            selected = st.radio(
                "Select one answer",
                item["options"],
                index=None,
                key=f"v67_wrap_choice_{week}_{attempt['attempt_number']}_{index}",
            )
            checked = st.form_submit_button("Check my answer", type="primary")
        if checked:
            if selected is None:
                st.warning("Select an answer before checking it.")
            else:
                correct = selected == item["answer"]
                response = {
                    "question": item["question"],
                    "selected": selected,
                    "answer": item["answer"],
                    "correct": bool(correct),
                    "explanation": item["explanation"],
                }
                attempt.setdefault("responses", []).append(response)
                attempt["score"] = int(attempt.get("score", 0)) + int(correct)
                attempt["feedback_pending"] = True
                st.rerun()
    else:
        response = attempt.get("responses", [])[-1]
        if response.get("correct"):
            st.success("Correct.")
        else:
            st.error("Not correct this time.")
            st.markdown(f"**Correct answer:** {response.get('answer', '')}")
        st.info(response.get("explanation", ""))
        next_label = "See my final score" if index + 1 >= total else "Next question"
        if st.button(next_label, type="primary", key=f"v67_wrap_next_{week}_{attempt['attempt_number']}_{index}"):
            attempt["feedback_pending"] = False
            attempt["index"] = index + 1
            if attempt["index"] >= total:
                attempt["complete"] = True
                previous = project.get("wrap_up_progress", {}).get(week, {})
                score = int(attempt.get("score", 0))
                best = max(score, int(previous.get("best_score", 0)))
                project.setdefault("wrap_up_progress", {})[week] = {
                    "completed": True,
                    "score": score,
                    "best_score": best,
                    "total": total,
                    "responses": list(attempt.get("responses", [])),
                    "attempts": int(previous.get("attempts", 0)) + 1,
                }
            st.rerun()


def assignment_plan_step(week, df):
    project = ensure_project_state()
    lab = WEEKLY_LABS[week]
    saved = project["assignment_plans"].get(week, {})
    st.markdown(
        f'<div class="simple-note"><strong>Your three-slide presentation assignment</strong><br>{lab["assignment"]}</div>',
        unsafe_allow_html=True,
    )
    st.caption("My Notebook stores the student's question, analysis, results, and interpretation. The student prepares and presents the final slides outside the app.")

    targets = suitable_targets(week, df)
    target_label, predictor_label = variable_language(week)
    if week in {"Week 1", "Week 8", "Week 15", "Week 16"}:
        target = ""
        predictors = []
    else:
        target_choices = [""] + targets
        old_target = saved.get("target") if saved.get("target") in targets else ""
        target = st.selectbox(
            target_label,
            target_choices,
            index=target_choices.index(old_target),
            format_func=lambda v: "— Select —" if not v else humanize(v),
            key=f"v68_assignment_target_{week}",
        )
        predictor_options = [c for c in df.columns if c != target]
        predictors = st.multiselect(
            predictor_label,
            predictor_options,
            default=[c for c in saved.get("predictors", []) if c in predictor_options],
            format_func=humanize,
            key=f"v68_assignment_predictors_{week}",
        )

    question = st.text_area(
        "Write your own research question",
        value=saved.get("question", ""),
        placeholder="Write one clear, answerable research question based on this week's assignment and the variables you selected.",
        key=f"v68_assignment_question_{week}",
    )
    st.caption("The app does not generate the assignment question. This is the student's own research decision.")

    if st.button("Save my assignment plan", type="primary", key=f"v68_assignment_plan_save_{week}"):
        needs_target = week not in {"Week 1", "Week 8", "Week 15", "Week 16"}
        if not question.strip():
            st.warning("Write your own research question.")
        elif needs_target and not target:
            st.warning("Choose the target or main outcome.")
        elif week in {"Week 5", "Week 6", "Week 7", "Week 9", "Week 10", "Week 11", "Week 12"} and not predictors:
            st.warning("Choose at least one predictor.")
        else:
            project["assignment_plans"][week] = {
                "question": question.strip(),
                "target": target,
                "predictors": predictors,
                "dataset_name": active_dataset_name(),
                "dataset_signature": dataframe_signature(df),
                "analyzed": False,
            }
            # Do not seed analytical widget keys while Step 1 is still on screen.
            # Streamlit would remove those unseen widget values at the end of the rerun.
            # Clearing the stamp makes Step 2 carry this saved target and predictors in
            # as defaults the next time the student opens the analysis.
            st.session_state.pop(_tool_plan_stamp_key(week, "student_assignment", "notebook"), None)
            st.success("Assignment plan saved. Open Step 2: Run My Analysis.")
            st.rerun()

def assignment_analyze_step(week, df):
    project = ensure_project_state()
    lab = WEEKLY_LABS[week]
    plan = project["assignment_plans"].get(week)
    if not plan:
        st.warning("Complete Step 1: Plan the Assignment first.")
        return
    if plan.get("dataset_signature") != dataframe_signature(df):
        st.warning("Your assignment dataset changed. Return to Plan and choose variables from the new dataset.")
        return
    st.markdown(f'<div class="question-card"><span class="tiny-label">My independent assignment question</span><br><strong>{plan["question"]}</strong></div>', unsafe_allow_html=True)
    apply_plan_to_tool(week, plan, context="student_assignment")
    st.info("Change the analytical controls as needed. Your variable selections will remain until you change them or replace the assignment plan.")
    if week == "Week 1":
        week1_probability_practice("assignment")
    elif lab.get("tool"):
        render_tool_page(lab["tool"])
    else:
        for item in lab["steps"]:
            st.checkbox(item, key=f"v68_assignment_review_{week}_{item}")
    if st.button("I completed my assignment analysis", key=f"v68_assignment_analyzed_{week}"):
        plan["analyzed"] = True
        st.success("Analysis complete. Open Step 3: Record Findings for Slides.")

def assignment_method_default(week, result, lab):
    """Return a student-facing method name that matches the weekly activity."""
    if week in {"Week 5", "Week 6"}:
        if isinstance(result, dict):
            feature_count = len(result.get("features", []) or [])
            if feature_count >= 2:
                return "Multiple linear regression"
            if feature_count == 1:
                return "Simple linear regression"
        return "Simple linear regression" if week == "Week 5" else "Multiple linear regression"

    fixed = {
        "Week 1": "Probability simulation",
        "Week 2": "Data inspection and research-question formulation",
        "Week 3": "Descriptive statistics and visualization",
        "Week 4": "Correlation and association",
        "Week 8": "Midterm review and interpretation",
        "Week 9": "Logistic regression",
        "Week 11": "Model evaluation and comparison",
        "Week 12": "Cross-validation",
        "Week 13": "Bootstrap resampling and uncertainty estimation",
        "Week 15": "Image classification",
        "Week 16": "Comprehensive final review",
    }
    if week in fixed:
        return fixed[week]
    if isinstance(result, dict):
        return result.get("display_model_name") or result.get("model_name") or lab["title"]
    return lab["title"]


def assignment_record_step(week, df):
    project = ensure_project_state()
    lab = WEEKLY_LABS[week]
    plan = project["assignment_plans"].get(week)
    if not plan:
        st.warning("Complete the assignment plan first.")
        return
    if not plan.get("analyzed"):
        st.info("Complete the analysis in Step 2 before recording the final evidence.")

    st.markdown("### Record the evidence you will use to prepare your slides")
    st.caption("This step saves analysis notes to My Notebook. It does not submit an assignment or create the final presentation.")
    result = week_result(week)
    notes = simple_result_notes(result)
    if notes:
        st.markdown("### Result-reading support")
        for note in notes:
            st.markdown("- " + note)
    else:
        st.info("Run the analysis first. Then return here to record the evidence you observed.")

    saved = project.get("weeks", {}).get(week, {})
    method_default = assignment_method_default(week, result, lab)
    saved_method = str(saved.get("method", "")).strip()
    # Migrate the old generic label so Week 5 and Week 6 describe the actual method.
    if week in {"Week 5", "Week 6"} and saved_method.lower() == "linear regression":
        saved_method = method_default
    method = st.text_input(
        "Method used",
        value=saved_method or method_default,
        key=f"v610_assignment_method_{week}",
    )
    key_result = st.text_area(
        "Key result for my slides",
        value=saved.get("key_result", ""),
        placeholder="Record one number, comparison, or pattern that directly answers your research question.",
        key=f"v68_assignment_result_{week}",
    )
    interpretation = st.text_area(
        "My interpretation in ordinary language",
        value=saved.get("interpretation", ""),
        key=f"v68_assignment_interpretation_{week}",
    )
    limitation = st.text_area("One limitation", value=saved.get("limitation", ""), key=f"v68_assignment_limitation_{week}")
    next_step = st.text_area("One next step", value=saved.get("next_step", ""), key=f"v68_assignment_next_{week}")

    if st.button("Save findings to My Notebook", type="primary", key=f"v68_assignment_record_{week}"):
        if not all(v.strip() for v in [method, key_result, interpretation, limitation, next_step]):
            st.warning("Complete the method, result, interpretation, limitation, and next step.")
        else:
            entry = {
                "week": week,
                "title": lab["title"],
                "question": plan.get("question", ""),
                "dataset_name": plan.get("dataset_name", active_dataset_name()),
                "target": plan.get("target", ""),
                "predictors": plan.get("predictors", []),
                "method": method.strip(),
                "key_result": key_result.strip(),
                "interpretation": interpretation.strip(),
                "limitation": limitation.strip(),
                "next_step": next_step.strip(),
                "completed_steps": ["Assignment plan", "Student analysis", "Findings recorded for slides"],
            }
            project["weeks"][week] = entry
            st.success("Findings saved. Use this notebook record to prepare and present the three slides required for this week's module.")
            st.rerun()

    entry = project.get("weeks", {}).get(week)
    if entry:
        report = weekly_report_text(week, entry)
        st.download_button(
            "Download this week’s slide-preparation notes",
            report,
            f"MATH490_{week.replace(' ', '_')}_slide_preparation_notes.md",
            "text/markdown",
            key=f"v68_assignment_notes_{week}",
        )
        with st.expander("Preview my slide-preparation notes", expanded=False):
            st.markdown(report)

def review_notebook():
    project = ensure_project_state()
    st.markdown("### Student assignment records")
    c1, c2 = st.columns(2)
    c1.metric("Saved assignment weeks", len(project.get("weeks", {})))
    c2.metric("Student", project.get("student_name") or "Not entered")
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download readable complete notebook",
        complete_notebook_markdown(),
        "MATH490_complete_notebook.md",
        "text/markdown",
        key="v68_notebook_markdown_download",
        use_container_width=True,
    )
    d2.download_button(
        "Download continuation backup",
        project_json(),
        "MATH490_notebook_backup.json",
        "application/json",
        key="v68_notebook_json_download",
        use_container_width=True,
    )

    st.markdown("#### Part A — My independent assignment analysis")
    for week in WEEKLY_LABS:
        entry = project.get("weeks", {}).get(week)
        with st.expander(f"{'✅' if entry else '○'} {week}: {WEEKLY_LABS[week]['title']}", expanded=False):
            st.write(f"**Weekly presentation assignment:** {WEEKLY_LABS[week]['assignment']}")
            if not entry:
                st.write("No student assignment analysis saved for this week.")
                continue
            st.write(f"**Student dataset:** {entry.get('dataset_name', '')}")
            st.write(f"**Student research question:** {entry.get('question', '')}")
            st.write(f"**Method:** {entry.get('method', '')}")
            st.write(f"**Key result:** {entry.get('key_result', '')}")
            st.write(f"**Student interpretation:** {entry.get('interpretation', '')}")
            st.write(f"**Limitation:** {entry.get('limitation', '')}")
            st.write(f"**Next step:** {entry.get('next_step', '')}")
            st.download_button(
                "Download slide-preparation notes",
                weekly_report_text(week, entry),
                f"MATH490_{week.replace(' ', '_')}_slide_preparation_notes.md",
                "text/markdown",
                key=f"v68_review_notes_{week}",
            )

    st.markdown("#### Part B — Instructor-led class analysis and practical record")
    briefs = st.session_state.get("lab_briefs", {})
    for week in WEEKLY_LABS:
        brief = briefs.get(week, {})
        plan = project.get("practice_plans", {}).get(week, {})
        progress = project.get("practical_progress", {}).get(week, {})
        wrap = project.get("wrap_up_progress", {}).get(week, {})
        if not any([brief, plan, progress, wrap]):
            continue
        with st.expander(f"Class record — {week}: {WEEKLY_LABS[week]['title']}", expanded=False):
            st.write(f"**Instructor class question:** {brief.get('research_question') or plan.get('question', '')}")
            st.write(f"**Class dataset:** {brief.get('dataset_name') or plan.get('dataset_name', '')}")
            st.write(f"**Class practical task:** {brief.get('instructions', WEEKLY_LABS[week]['assignment'])}")
            st.write(f"**Student practical reflection:** {progress.get('reflection', 'Not yet recorded')}")
            if progress.get("result_notes"):
                st.write("**Class result-reading notes:**")
                for note in progress["result_notes"]:
                    st.markdown("- " + note)
            if wrap:
                st.write(f"**Wrap-Up score:** {wrap.get('score', 0)}/{wrap.get('total', 0)} · Best: {wrap.get('best_score', 0)}/{wrap.get('total', 0)}")

def my_notebook_page(week):
    activate_dataset_scope("notebook")
    st.title(f"📘 My Notebook — {week}")
    render_weekly_journey(week, "My Notebook")
    st.markdown("**This is the student's independent assignment workspace after class.** Use the assigned dataset, make your own research decisions, record your analysis, and use the saved evidence to prepare the required presentation.")
    st.markdown(
        f'<div class="simple-note"><strong>This week’s presentation assignment</strong><br>{WEEKLY_LABS[week]["assignment"]}</div>',
        unsafe_allow_html=True,
    )
    st.caption("The final slides are created and presented outside the app. My Notebook is the analysis record that supports them.")

    wrap_saved = ensure_project_state().get("wrap_up_progress", {}).get(week, {})
    if not wrap_saved.get("completed"):
        st.info("The normal sequence is to finish Wrap-Up before starting the independent assignment.")
    project = ensure_project_state()
    name = st.text_input("Student name", value=project.get("student_name", ""), key="v68_notebook_student")
    project["student_name"] = name
    if week == "Week 1":
        st.info("Week 1 assignment uses the probability simulator. No CSV or Excel file is required.")
    else:
        dataset_manager("notebook", "Assignment dataset", expanded=False)
    df = get_df()

    view = st.radio(
        "Notebook",
        ["Do this week’s assignment", "Review saved work"],
        horizontal=True,
        key=f"v68_notebook_view_{week}",
    )
    if view == "Review saved work":
        review_notebook()
        return

    step = st.radio(
        "Assignment steps",
        ["1. Plan the Assignment", "2. Run My Analysis", "3. Record Findings for Slides"],
        horizontal=True,
        key=f"v68_assignment_step_{week}",
    )
    st.divider()
    if step.startswith("1"):
        assignment_plan_step(week, df)
    elif step.startswith("2"):
        assignment_analyze_step(week, df)
    else:
        assignment_record_step(week, df)

def full_studio_page(page):
    activate_dataset_scope("studio")
    st.markdown('<div class="simple-note"><strong>Optional Full Studio</strong><br>All analytical tools are available here for independent exploration. Beginners should normally use Today’s Lab, Practical Studio, Wrap-Up, and My Notebook.</div>', unsafe_allow_html=True)
    render_tool_page(page)


# -----------------------------------------------------------------------------
# Report builder and app entry point
# -----------------------------------------------------------------------------
def report_metric_summary(result):
    if not isinstance(result, dict):
        return ""
    preferred = (
        ["Mean absolute error", "Root mean squared error", "R-squared"]
        if result.get("problem") == "regression"
        else ["Accuracy", "Precision", "Recall", "F1-score", "ROC area under the curve"]
    )
    values = []
    for name in preferred:
        if name in result.get("metrics", {}):
            values.append(f"{name}: {result['metrics'][name]:.3f}")
    return "; ".join(values[:4])


def page_report():
    st.title("Three-Slide Mini-Report Builder")
    guide(
        "Turn your own research question, method, result, and interpretation into a concise three-slide report.",
        ["Write or refine your research question", "Enter the method used", "Summarize the key result"],
        ["The method matches the question", "The result uses correct metric language", "The conclusion avoids overclaiming"],
        ["Prepare three slides", "Use one key visual", "State a limitation and next step"],
        "Do not claim causation from observational prediction.",
    )
    st.info(
        "Suggested course workflow: use one main dataset and an evolving research question from data exploration through "
        "regression, classification, evaluation, cross-validation, and bootstrap. Forecasting and computer vision can use "
        "new datasets and new research questions."
    )
    result = st.session_state.get("latest_model_result")
    report_defaults = {
        "report_dataset": st.session_state.get("dataset_name", ""),
        "report_question": "",
        "report_target": "",
        "report_predictors": "",
        "report_method": "",
        "report_result": "",
        "report_interpretation": "",
        "report_limitation": "",
        "report_next_step": "",
    }
    for key, value in report_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    lab_options = [
        "Week 2: Data and Research Questions",
        "Week 3: Descriptive Statistics and Visualization",
        "Week 4: Correlation and Association",
        "Week 5: Simple Linear Regression",
        "Week 6: Multiple Linear Regression",
        "Week 7: Machine Learning Regression",
        "Week 9: Logistic Regression",
        "Week 10: Machine Learning Classification",
        "Week 11: Model Evaluation and Comparison",
        "Week 12: Cross-Validation",
        "Week 13: Bootstrap and Uncertainty",
        "Week 14: Time Series Forecasting",
        "Week 15: Computer Vision",
        "Week 16: Final Review",
    ]
    st.selectbox("Lab focus", lab_options, key="report_lab")

    if st.button("Use latest saved analysis", key="report_load_latest"):
        if isinstance(result, dict):
            st.session_state.report_dataset = result.get("dataset_name", st.session_state.get("dataset_name", ""))
            st.session_state.report_target = humanize(result.get("target", ""))
            st.session_state.report_predictors = ", ".join(map(humanize, result.get("features", [])))
            st.session_state.report_method = result.get("model_name", "")
            st.session_state.report_result = report_metric_summary(result)
        else:
            st.warning("Train or evaluate a model first, or enter the report details manually.")

    c1, c2 = st.columns(2)
    with c1:
        st.text_area("Research question", key="report_question", placeholder="Example: How well do study habits predict mathematics performance?")
        st.text_input("Dataset", key="report_dataset")
        st.text_input("Target or outcome", key="report_target")
        st.text_area("Predictors or variables examined", key="report_predictors")
    with c2:
        st.text_area("Method used", key="report_method", placeholder="Example: Multiple linear regression with a 75/25 train-test split")
        st.text_area("Key result", key="report_result", placeholder="Report only the most relevant coefficient, association, or performance metric.")
        st.text_area("Interpretation or conclusion", key="report_interpretation")
        st.text_area("Limitation", key="report_limitation")
        st.text_area("Next step", key="report_next_step")

    question = st.session_state.get("report_question", "").strip()
    dataset = st.session_state.get("report_dataset", "").strip()
    target = st.session_state.get("report_target", "").strip()
    predictors = st.session_state.get("report_predictors", "").strip()
    method = st.session_state.get("report_method", "").strip()
    key_result = st.session_state.get("report_result", "").strip()
    interpretation = st.session_state.get("report_interpretation", "").strip()
    limitation = st.session_state.get("report_limitation", "").strip()
    next_step = st.session_state.get("report_next_step", "").strip()
    lab_focus = st.session_state.get("report_lab", lab_options[0])

    report = f"""# MATH 490 Three-Slide Mini-Report

**Lab focus:** {lab_focus}

## Slide 1: Research Question and Data
**Research question:** {question or '[Enter your research question]'}

**Dataset:** {dataset or '[Enter the dataset]'}

**Target or outcome:** {target or '[Enter the target or outcome]'}

**Predictors or variables examined:** {predictors or '[Enter the predictors or variables]'}

**Suggested visual:** One descriptive chart that introduces the outcome or a key relationship.

---
## Slide 2: Method and Key Result
**Method used:** {method or '[Enter the method used]'}

**Key result:** {key_result or '[Enter the most important result]'}

**Suggested visual:** One model-performance, association, coefficient, or explanation figure directly connected to the result.

---
## Slide 3: Interpretation, Limitation, and Next Step
**Interpretation or conclusion:** {interpretation or '[Explain what the result means]'}

**Limitation:** {limitation or '[State one important limitation]'}

**Next step:** {next_step or '[State what should be tested or improved next]'}

**Communication reminder:** Describe what the analysis supports without overstating causation.
"""
    st.subheader("Report preview")
    st.markdown(report)
    st.download_button("Download mini-report outline", report, "MATH490_three_slide_report.md", "text/markdown", key="report_dl")

# -----------------------------------------------------------------------------
# Learning Library and independent Exam Practice
# -----------------------------------------------------------------------------
GLOSSARY_TSV = r"""
Artificial intelligence	AI and Machine Learning	The broad field of building computer systems that perform tasks associated with human intelligence, such as recognizing patterns, making predictions, or supporting decisions.	It is the umbrella idea for the course; machine learning is one approach within artificial intelligence.		machine learning, model, responsible AI
Machine learning	AI and Machine Learning	A branch of artificial intelligence in which a system learns patterns from data instead of relying only on hand-written rules.	It explains why model performance depends on the training data and evaluation design.		artificial intelligence, supervised learning, training
Supervised learning	AI and Machine Learning	Learning a relationship from examples that contain both predictors and a known target.	Regression and classification in this course are supervised-learning tasks.		regression, classification, target
Unsupervised learning	AI and Machine Learning	Finding structure in data without a supplied target, for example by clustering similar observations.	It distinguishes pattern discovery from supervised prediction.		clustering, supervised learning
Algorithm	AI and Machine Learning	A defined procedure used to solve a problem or fit a model.	The same data can be analyzed with different algorithms that make different assumptions.		model, training
Model	AI and Machine Learning	A fitted mathematical or computational representation that converts inputs into predictions or estimates.	A model is the learned result; an algorithm is the procedure used to fit it.		algorithm, parameter, prediction
Training	AI and Machine Learning	The process of using data to estimate a model's internal parameters.	Training performance alone does not show whether the model will generalize.		model, test set, generalization
Inference	AI and Machine Learning	Using a trained model to produce a prediction for new input data.	It is the stage at which a deployed model is actually used.		prediction, deployment
Feature	AI and Machine Learning	An input variable supplied to a model; also called a predictor.	Useful features contain information that may help explain or predict the target.		predictor, target
Predictor	Data and Research	A variable used to explain or predict an outcome.	Predictors must be available at the time the prediction would be made.		feature, target, leakage
Target	Data and Research	The outcome that the analysis is designed to explain or predict.	The target determines whether the task is regression or classification.		predictor, regression, classification
Parameter	AI and Machine Learning	An internal value learned from the training data, such as a regression slope or neural-network weight.	Parameters are learned during training rather than chosen directly by the user.		hyperparameter, coefficient
Hyperparameter	AI and Machine Learning	A model setting chosen before or during model development, such as tree depth or regularization strength.	Hyperparameters affect complexity and should be selected without using the final test set.		tuning, cross-validation, parameter
Hyperparameter tuning	AI and Machine Learning	The process of comparing candidate hyperparameter values using training and validation information.	Tuning can improve a model but can also overfit the validation process when done carelessly.		hyperparameter, validation set
Overfitting	AI and Machine Learning	A model learns the training data too closely and performs substantially worse on new data.	It is why held-out evaluation, regularization, and cross-validation are needed.		underfitting, regularization, generalization
Underfitting	AI and Machine Learning	A model is too simple to capture important patterns in either the training or new data.	Increasing useful complexity may improve both training and test performance.		overfitting, model complexity
Generalization	AI and Machine Learning	The ability of a model to perform well on genuinely unseen data from the intended setting.	Generalization, not memorization, is the goal of predictive modeling.		test set, leakage, overfitting
Regularization	AI and Machine Learning	A constraint or penalty that discourages an unnecessarily complex model.	It can reduce overfitting, although too much regularization can cause underfitting.		ridge regression, lasso regression, dropout
Bias-variance trade-off	AI and Machine Learning	The balance between a model being too rigid and a model changing too much with the training sample.	It helps explain why neither the simplest nor the most complex model is automatically best.		overfitting, underfitting
Baseline	Model Evaluation	A simple reference method that a more complex model should outperform to demonstrate added value.	Without a baseline, a sophisticated model may appear useful even when a simple rule works better.		majority baseline, naïve forecast
Research question	Data and Research	A precise, answerable statement describing what the analysis will investigate.	It should identify the outcome, relevant variables, population or context, and intended type of conclusion.		target, predictor
Observation	Data and Research	One row or case in a dataset, such as one person, location, image, or date.	Understanding what one row represents prevents incorrect interpretation and splitting.		variable, dataset
Variable	Data and Research	A recorded characteristic represented by a dataset column.	Its type determines which summaries, plots, and models are suitable.		numerical variable, categorical variable
Numerical variable	Data and Research	A variable whose values represent meaningful quantities or measurements.	Numerical outcomes can usually be modeled with regression.		categorical variable, regression
Categorical variable	Data and Research	A variable whose values represent groups or labels rather than measured amounts.	A categorical target creates a classification problem.		numerical variable, classification
Missing value	Data and Research	A value that was not observed, recorded, or available.	Missingness must be examined because deleting or filling values can change the analysis.		imputation, data quality
Imputation	Data and Research	Replacing missing values using a defined rule, such as the median or most frequent category.	Imputation should be learned from training data inside the modeling pipeline to avoid leakage.		missing value, preprocessing
Standardization	Data and Research	Transforming a numerical variable so it is centered near zero and measured in standard-deviation units.	Distance-based and penalty-based models often work better when predictors share a comparable scale.	z=(x-mean)/standard deviation	feature scaling
One-hot encoding	Data and Research	Representing each category with indicator columns so that a model can use categorical predictors numerically.	It allows many statistical and machine-learning models to use non-numerical inputs.		categorical variable, preprocessing
Preprocessing	Data and Research	The transformations applied before fitting a model, such as imputation, scaling, and encoding.	Preprocessing must be fitted using training data only during honest validation.		pipeline, leakage
Data leakage	Data and Research	Information unavailable at prediction time, or information from evaluation data, improperly enters model training.	Leakage can produce excellent-looking scores that fail in real use.		preprocessing leakage, target leakage, sequence leakage
Target leakage	Data and Research	A predictor directly or indirectly contains the answer represented by the target.	It makes evaluation unrealistic because the model receives information it would not truly have.		data leakage, predictor
Sequence leakage	Data and Research	Closely related frames or observations from the same sequence appear in both training and evaluation sets.	The model may recognize the sequence rather than generalize to a genuinely new case.		group-aware split, computer vision
Group-aware split	Validation and Uncertainty	A split that keeps every observation from the same person, site, sequence, or other group entirely on one side.	It provides a more honest estimate when observations within groups are related.		sequence leakage, cross-validation
Training set	Validation and Uncertainty	The observations used to fit model parameters.	The model is allowed to learn from these observations.		validation set, test set
Validation set	Validation and Uncertainty	Data used to compare settings, select models, or guide early stopping without fitting the final reported parameters directly.	Repeatedly inspecting validation results means the validation set is part of model development.		training set, test set
Test set	Validation and Uncertainty	An untouched set used for the final assessment after model choices have been completed.	It provides the strongest simple estimate of performance on unseen data.		validation set, held-out evaluation
Random seed	Validation and Uncertainty	A fixed starting value that makes a randomized procedure reproducible.	Changing the seed can reveal whether a result depends heavily on one random split.		reproducibility
Experiment	Probability and Statistics	A repeatable process that produces one of several possible outcomes.	It is the starting point for defining probability.		outcome, sample space
Outcome	Probability and Statistics	One possible result of a probability experiment.	Probabilities are assigned to outcomes or events.		experiment, event
Sample space	Probability and Statistics	The complete set of all possible outcomes of an experiment.	Probability calculations begin by identifying what can occur.	S	outcome, event
Event	Probability and Statistics	A set of one or more outcomes from the sample space.	An event is the statement whose probability is being calculated.	A	probability
Probability	Probability and Statistics	A number from zero to one that describes how likely an event is.	Probability supports uncertainty reasoning, classification probabilities, and simulation.	0 <= P(A) <= 1	event
Complement rule	Probability and Statistics	The probability that an event does not occur equals one minus the probability that it occurs.	It is often easier to calculate the opposite event first.	P(A^c)=1-P(A)	event
Addition rule	Probability and Statistics	A rule for the probability that event A or event B occurs, subtracting any overlap counted twice.	It is used for unions of events.	P(A union B)=P(A)+P(B)-P(A intersection B)	union, intersection
Multiplication rule	Probability and Statistics	A rule connecting joint and conditional probabilities.	It is used for the probability that two events occur together.	P(A intersection B)=P(A)P(B|A)	conditional probability
Conditional probability	Probability and Statistics	The probability of an event after learning that another event has occurred.	It formalizes belief updating when information changes.	P(A|B)=P(A intersection B)/P(B)	Bayes theorem
Independence	Probability and Statistics	Two events are independent when learning that one occurred does not change the probability of the other.	Independence determines when probabilities can be multiplied directly.	P(A intersection B)=P(A)P(B)	conditional probability
Bayes theorem	Probability and Statistics	A rule for updating the probability of a hypothesis after observing evidence.	It combines a prior probability with how likely the evidence is under competing explanations.	P(A|B)=P(B|A)P(A)/P(B)	conditional probability
Theoretical probability	Probability and Statistics	The probability expected from the mathematical rules of an experiment.	It can be compared with simulation or observed data.		experimental probability
Experimental probability	Probability and Statistics	The observed proportion of trials in which an event occurs.	It often approaches theoretical probability as the number of independent trials grows.	successes/trials	theoretical probability
Random variable	Probability and Statistics	A numerical value determined by the outcome of a random experiment.	Expected value and variance describe its long-run behavior.	X	expected value
Expected value	Probability and Statistics	The probability-weighted long-run average value of a random variable.	It summarizes the center of a probability distribution.	E(X)=sum xP(X=x)	variance
Variance	Probability and Statistics	The expected squared distance from the mean.	It measures spread and gives more weight to large deviations.	Var(X)=E[(X-E(X))^2]	standard deviation
Standard deviation	Probability and Statistics	The square root of variance, expressed in the original units of the variable.	It describes a typical scale of variation around the mean.	SD=sqrt(variance)	variance
Mean	Visualization and Association	The sum of values divided by the number of values.	It describes the center but can be influenced by extreme observations.	mean=sum(x)/n	median
Median	Visualization and Association	The middle value after observations are ordered.	It is often more resistant to extreme values than the mean.		mean
Distribution	Visualization and Association	The overall pattern of a variable's values, including center, spread, shape, and unusual observations.	Understanding the distribution guides summaries, transformations, and models.		histogram
Outlier	Visualization and Association	An observation that lies far from most of the data.	It may represent a real rare case, an error, or an influential observation requiring investigation.		boxplot, Huber regression
Histogram	Visualization and Association	A plot that groups a numerical variable into intervals and displays how many observations fall in each interval.	It reveals distribution shape, center, spread, and possible outliers.		distribution
Boxplot	Visualization and Association	A compact plot showing the median, quartiles, spread, and potential outliers of a numerical variable.	It is useful for comparing numerical distributions across groups.		median, outlier
Scatterplot	Visualization and Association	A plot of paired numerical values for two variables.	It helps reveal direction, form, clusters, outliers, and changing spread.		correlation
Bar chart	Visualization and Association	A chart comparing counts or summaries across categories.	Its y-axis must state whether the bars show counts, means, medians, or another quantity.		categorical variable
Correlation	Visualization and Association	A standardized measure of how two variables vary together.	Correlation describes association, not causation.	-1 <= r <= 1	Pearson correlation, Spearman correlation
Pearson correlation	Visualization and Association	A measure of the strength and direction of a linear relationship between two numerical variables.	It is most informative when the relationship is approximately linear and not dominated by outliers.	r	correlation
Spearman correlation	Visualization and Association	A rank-based measure of monotonic association between two variables.	It can capture steadily increasing or decreasing relationships that are not linear.	rho	correlation
Kendall correlation	Visualization and Association	A rank association based on concordant and discordant pairs.	It is interpretable and useful for ordinal or small datasets.	tau	correlation
Partial correlation	Visualization and Association	The association between two numerical variables after accounting for selected control variables.	It can reduce selected confounding but does not prove causation.		correlation
Eta	Visualization and Association	A measure of association between a categorical grouping variable and a numerical outcome.	It is useful when comparing how much numerical values differ across groups.	0 <= eta <= 1	association
Cramer's V	Visualization and Association	A standardized measure of association between two categorical variables based on a contingency table.	It summarizes the strength of categorical association.	0 <= V <= 1	chi-square
Causation	Visualization and Association	A claim that changing one factor produces a change in another.	Prediction and association from observational data do not by themselves establish causation.		correlation
Regression	Regression	A supervised-learning task that predicts a numerical target.	It is used when the outcome represents an amount, measurement, or continuous value.		classification
Linear regression	Regression	A model that predicts a numerical target using a weighted sum of predictors plus an intercept.	It provides a simple, interpretable baseline for numerical prediction.	y=b0+b1x1+...+bkxk	slope, intercept
Simple linear regression	Regression	Linear regression with one predictor.	It is useful for introducing slope, intercept, fitted values, and residuals.	y=b0+b1x	slope
Multiple linear regression	Regression	Linear regression with two or more predictors.	Each slope is interpreted while holding the other included predictors constant.	y=b0+b1x1+...+bkxk	multicollinearity
Slope	Regression	The estimated change in the predicted target for a one-unit increase in a predictor, holding other predictors constant when applicable.	It describes direction and magnitude in linear regression.	b1	coefficient
Intercept	Regression	The predicted target when all numerical predictors equal zero and categorical predictors are at reference levels.	It may or may not have a meaningful real-world interpretation.	b0	linear regression
Coefficient	Regression	A learned numerical weight describing how a predictor contributes to a model.	Its interpretation depends on the model, scaling, and coding.		parameter
Fitted value	Regression	The value predicted by a model for an observed predictor row.	It is compared with the observed target to calculate a residual.	y-hat	residual
Residual	Regression	The observed target minus the fitted or predicted value.	Residuals reveal model errors and patterns not captured by the model.	e=y-y-hat	fitted value
Mean absolute error	Model Evaluation	The average absolute difference between predictions and observed values.	It is expressed in target units and treats errors linearly.	MAE=(1/n)sum|y-y-hat|	root mean squared error
Root mean squared error	Model Evaluation	The square root of the average squared prediction error.	It is expressed in target units and gives extra weight to large mistakes.	RMSE=sqrt((1/n)sum(y-y-hat)^2)	mean absolute error
R-squared	Model Evaluation	The proportion of variation in a numerical target explained relative to predicting the mean.	It should be interpreted with held-out error and can be negative on test data.	R^2=1-SSE/SST	regression
Multicollinearity	Regression	Strong overlap among predictors in a multiple regression.	It can make individual coefficients unstable even when predictions remain useful.		variance inflation factor
Variance inflation factor	Regression	A diagnostic measuring how strongly one predictor is explained by the other predictors.	Large values indicate coefficient instability from multicollinearity.	VIF=1/(1-R_j^2)	multicollinearity
Ridge regression	Regression	Linear regression with an L2 penalty that shrinks coefficients toward zero.	It can stabilize prediction when predictors overlap, but usually retains every predictor.		regularization
Lasso regression	Regression	Linear regression with an L1 penalty that can shrink some coefficients exactly to zero.	It combines regularization with a form of predictor selection.		regularization
Elastic net	Regression	A linear model combining L1 and L2 regularization.	It balances lasso-style sparsity with ridge-style stability.		lasso regression, ridge regression
Huber regression	Regression	A robust linear model that reduces the influence of observations with large residuals.	It can be useful when outliers would dominate ordinary least squares.		outlier
Classification	Classification	A supervised-learning task that predicts a categorical target.	Examples include pass or fail, disease class, and image category.		regression
Logistic regression	Classification	A linear classification model that converts a weighted predictor score into class probabilities.	It is an interpretable baseline for binary or multiclass classification.	P(Y=1|X)=1/(1+e^-z)	odds ratio
Log-odds	Classification	The logarithm of the odds of an event.	Logistic regression models log-odds as a linear function of predictors.	log(p/(1-p))	odds
Odds	Classification	The probability of an event divided by the probability that it does not occur.	Odds are converted to probabilities and are used to interpret logistic regression.	odds=p/(1-p)	odds ratio
Odds ratio	Classification	The multiplicative change in odds associated with a one-unit predictor increase, holding other predictors constant.	Values above one increase estimated odds; values below one decrease them.	OR=e^coefficient	logistic regression
Decision threshold	Classification	The probability cutoff used to convert a predicted probability into a class label.	Changing it alters the balance between false positives and false negatives.		precision, recall
Confusion matrix	Classification	A table comparing observed classes with predicted classes.	It shows the types of errors hidden by a single overall score.		true positive, false positive
True positive	Classification	A positive case correctly predicted as positive.	It contributes to recall and precision.	TP	confusion matrix
True negative	Classification	A negative case correctly predicted as negative.	It contributes to accuracy and specificity.	TN	confusion matrix
False positive	Classification	A negative case incorrectly predicted as positive.	Its importance depends on the cost of a false alarm.	FP	precision
False negative	Classification	A positive case incorrectly predicted as negative.	Its importance depends on the cost of missing a true case.	FN	recall
Accuracy	Model Evaluation	The fraction of all predictions that are correct.	It can be misleading when one class is much more common than another.	(TP+TN)/(TP+TN+FP+FN)	balanced accuracy
Balanced accuracy	Model Evaluation	The average recall calculated across classes.	It gives each class equal influence and is useful for imbalanced data.	(sensitivity+specificity)/2	class imbalance
Precision	Model Evaluation	Among cases predicted as positive, the fraction that are truly positive.	It answers how trustworthy positive predictions are.	TP/(TP+FP)	recall
Recall	Model Evaluation	Among truly positive cases, the fraction correctly found by the model.	It answers how many actual positive cases were detected.	TP/(TP+FN)	precision
Specificity	Model Evaluation	Among truly negative cases, the fraction correctly predicted as negative.	It measures the ability to avoid false positives.	TN/(TN+FP)	recall
F1-score	Model Evaluation	The harmonic mean of precision and recall.	It is high only when both precision and recall are reasonably high.	F1=2PR/(P+R)	precision, recall
ROC curve	Model Evaluation	A curve of true-positive rate against false-positive rate across classification thresholds.	It shows ranking performance across many thresholds.		ROC area under the curve
ROC area under the curve	Model Evaluation	The probability that a randomly selected positive case receives a higher score than a randomly selected negative case.	It measures ranking discrimination but not probability calibration.	AUC	ROC curve
Average precision	Model Evaluation	A precision-recall summary that emphasizes performance on the positive class.	It is often informative when the positive class is rare.		precision-recall curve
Calibration	Model Evaluation	Agreement between predicted probabilities and observed event frequencies.	A model can rank cases well but still produce poorly calibrated probabilities.		Brier score
Brier score	Model Evaluation	The average squared difference between predicted probabilities and binary outcomes.	It evaluates probability accuracy and calibration together; lower is better.	(1/n)sum(p-y)^2	calibration
Class imbalance	Classification	A situation in which some target classes are much more common than others.	Accuracy may hide poor performance on the minority class.		balanced accuracy
Decision tree	Machine Learning Models	A model that repeatedly splits observations using predictor rules and makes a prediction in each final leaf.	It captures nonlinear relationships and interactions but can overfit when unrestricted.		tree depth
Random forest	Machine Learning Models	An ensemble of trees trained on resampled data and randomized predictor subsets, whose predictions are averaged or voted.	It usually improves stability over one tree.		number of trees, max features
Gradient boosting	Machine Learning Models	An ensemble that builds small trees sequentially, with each stage focusing on earlier errors.	It can be highly accurate but requires careful control of learning rate and complexity.		learning rate, boosting stages
K-nearest neighbors	Machine Learning Models	A model that predicts using the outcomes of the most similar training observations.	Its results depend strongly on scaling, the distance definition, and the number of neighbors.	K	feature scaling
Support vector machine	Machine Learning Models	A classifier that seeks a boundary with a wide margin between classes and may use a kernel for nonlinear separation.	It can work well in high-dimensional spaces but needs scaling and parameter tuning.		C, kernel, gamma
Support vector regression	Machine Learning Models	The regression counterpart of a support vector machine, fitting a function with an epsilon-insensitive error region.	It can capture nonlinear patterns through kernels.		C, epsilon, kernel
Linear discriminant analysis	Machine Learning Models	A classification model that assumes classes share a covariance structure and separates them using linear boundaries.	It can be efficient when its distributional assumptions are reasonable.		quadratic discriminant analysis
Quadratic discriminant analysis	Machine Learning Models	A classification model that allows each class to have its own covariance structure, producing curved boundaries.	It is more flexible than linear discriminant analysis but needs more data.		linear discriminant analysis
Feedforward neural network	Neural Networks	A neural network that sends one fixed input vector forward through hidden layers to an output.	It learns nonlinear combinations of predictors but has no recurrent memory or image-specific spatial filters.	FFNN	hidden layer, activation function
Neuron	Neural Networks	A computational unit that forms a weighted sum of inputs and applies an activation function.	Networks learn by adjusting the weights connecting many neurons.		hidden layer
Hidden layer	Neural Networks	A layer between the inputs and output that learns intermediate representations.	More layers and neurons increase flexibility but can also increase overfitting and computation.		neuron
Activation function	Neural Networks	A nonlinear transformation applied inside a neural network, such as ReLU, tanh, or logistic.	Without nonlinear activations, stacked layers behave like one linear transformation.	ReLU(x)=max(0,x)	neural network
Learning rate	Neural Networks	The step size used when updating model weights during optimization.	Too large can make training unstable; too small can make learning very slow.		optimizer
Epoch	Neural Networks	One complete pass through the training dataset during neural-network training.	More epochs do not guarantee better generalization.		early stopping
Batch size	Neural Networks	The number of training observations used for one weight update.	It affects memory use, training noise, and computational speed.		epoch
Dropout	Neural Networks	A regularization method that randomly disables a fraction of neural units during training.	It discourages the network from relying too heavily on particular pathways.		regularization
Early stopping	Neural Networks	Stopping training when validation performance no longer improves and restoring the best weights.	It limits overfitting without choosing the maximum epoch blindly.		validation set, epoch
Long short-term memory network	Neural Networks	A recurrent neural network with gates designed to preserve or forget information across an ordered sequence.	It can learn time-dependent patterns from a lookback sequence.	LSTM	lookback window
Convolutional neural network	Neural Networks	A neural network that learns local visual filters while preserving the spatial arrangement of pixels.	It is specialized for images and learns edges, textures, shapes, and higher-level patterns.	CNN	convolution, pooling
Cross-validation	Validation and Uncertainty	Repeatedly dividing training data into folds so each fold is used for validation while the others train the model.	It estimates performance stability and supports model selection without touching the final test set.		fold, nested cross-validation
Fold	Validation and Uncertainty	One subset used as the validation portion during a cross-validation iteration.	Every observation should be placed according to the chosen validation design.		cross-validation
Stratification	Validation and Uncertainty	Splitting data while preserving approximately similar class proportions across subsets.	It reduces accidental class imbalance in classification folds.		classification
Nested cross-validation	Validation and Uncertainty	An outer validation loop estimates performance while an inner loop selects hyperparameters.	It separates tuning from evaluation and reduces optimistic model-selection bias.		cross-validation, tuning
Bootstrap	Validation and Uncertainty	Repeatedly sampling observations with replacement from the observed dataset and recalculating a statistic.	It approximates how an estimate might vary across repeated samples.		bootstrap standard error
Bootstrap bias	Validation and Uncertainty	The bootstrap mean estimate minus the original sample estimate.	It indicates whether the resampling distribution is systematically shifted.	bias=bootstrap mean-original estimate	bootstrap
Bootstrap standard error	Validation and Uncertainty	The standard deviation of the bootstrap estimates.	It estimates the sampling variability of the statistic.	SE_boot=SD(bootstrap estimates)	standard error
Standard error	Validation and Uncertainty	The estimated standard deviation of an estimator across repeated samples.	It describes uncertainty in an estimate, not the spread of individual observations.		standard deviation
Confidence interval	Validation and Uncertainty	A range produced by a procedure designed to contain the true population quantity at a stated long-run rate.	It expresses uncertainty about an estimate rather than the range of individual data values.		standard error
Percentile bootstrap interval	Validation and Uncertainty	A bootstrap interval formed from lower and upper percentiles of the resampled estimates.	It is simple and directly uses the empirical bootstrap distribution.		bootstrap
Time series	Forecasting	Observations ordered through time, often with dependence between neighboring rows.	Time order must be preserved during feature construction and evaluation.		forecasting
Forecasting	Forecasting	Predicting values that occur after the latest information available at a forecast origin.	A valid forecast must not use future information.		forecast horizon, backtesting
Forecast horizon	Forecasting	The number of future time steps between the forecast origin and the target being predicted.	A horizon of one predicts the next row; a horizon of seven predicts seven rows ahead.	h	lag
Lag	Forecasting	A past value indexed by how many rows earlier it occurred.	Lagged target values often provide useful information for forecasting.	x_(t-k)	forecast horizon
Rolling window	Forecasting	A summary calculated from a fixed number of the most recent observations.	Rolling means and standard deviations describe recent level and variability.		lag
Seasonality	Forecasting	A pattern that repeats at a regular time interval.	Seasonal structure motivates seasonal lags and seasonal naïve baselines.		seasonal cycle length
Seasonal cycle length	Forecasting	The number of rows in one complete repeating cycle.	Examples include 24 hourly rows for a daily cycle or 12 monthly rows for a yearly cycle.	s	seasonality
Chronological split	Forecasting	Training on earlier rows and evaluating on later rows without shuffling time.	It reproduces the direction in which real forecasts are made.		backtesting
Backtesting	Forecasting	Simulating historical forecasting by repeatedly training on the past and evaluating on later periods.	It estimates operational performance while respecting time order.		time-series cross-validation
Naïve latest-value forecast	Forecasting	A baseline that predicts the next target will equal the most recently observed target.	A complex model should improve on this realistic simple rule.	y-hat_(t+h)=y_t	baseline
Seasonal naïve forecast	Forecasting	A baseline that predicts using the value from the corresponding point one seasonal cycle earlier.	It is a strong benchmark when seasonal patterns repeat.	y-hat_t=y_(t-s)	seasonality
Exogenous predictor	Forecasting	An additional forecasting input other than past values of the target.	Its future value must be known or itself forecast at the forecast origin.		data leakage
Lookback window	Forecasting	The number of consecutive past time steps supplied to a recurrent model for each prediction.	It controls how much recent history an LSTM receives.		long short-term memory network
Pixel	Computer Vision	A numerical image element at a particular row and column.	Images become model inputs through pixel intensity values.		image channel
Image channel	Computer Vision	A numerical layer of an image, commonly red, green, and blue for a color image.	A color pixel is represented by values across its channels.		pixel
Image classification	Computer Vision	Assigning an image to one of several category labels.	It is a supervised classification problem with high-dimensional structured inputs.		convolutional neural network
Flattened image features	Computer Vision	Pixel values reshaped from a spatial image grid into one long predictor vector.	Conventional models can use them, but the explicit two-dimensional neighborhood structure is lost.		feedforward neural network
Convolution	Computer Vision	A small learnable filter that moves across an image and responds to local patterns.	Convolutional layers learn edges, textures, and shapes while sharing weights across positions.		convolutional neural network
Pooling	Computer Vision	A downsampling operation that summarizes nearby activations.	It reduces spatial size and can make learned features less sensitive to small position changes.		convolution
Transfer learning	Computer Vision	Starting from a model pretrained on a large dataset and adapting it to a new task.	It can improve performance when the new labeled dataset is small.		pretrained model
Data augmentation	Computer Vision	Creating plausible transformed training images, such as small rotations, crops, or flips.	It increases visual variation and can reduce overfitting without adding test information.		computer vision
Pretrained model	Computer Vision	A model whose weights were learned previously on another large dataset.	It can supply reusable visual or language features for a new task.		transfer learning
Responsible AI	Responsible AI	The practice of developing and using artificial-intelligence systems with attention to performance, fairness, transparency, privacy, safety, and human consequences.	A technically accurate model can still be inappropriate or harmful in context.		fairness, human oversight
Fairness	Responsible AI	The examination of whether model errors, benefits, or harms differ unjustifiably across people or groups.	Overall accuracy can conceal unequal performance.		responsible AI
Transparency	Responsible AI	Providing understandable information about data, methods, limitations, and intended use.	Users need enough information to judge whether a model is appropriate.		explainability
Explainability	Responsible AI	Methods and communication that help people understand how a model uses information or why it produces outputs.	Explanations describe the fitted model and do not prove causation.		permutation importance, SHAP
Permutation importance	Responsible AI	The decrease in held-out model performance after one predictor is shuffled.	It estimates how much the fitted model relies on that predictor for prediction.		model explanation
SHAP	Responsible AI	A family of methods that attributes a model prediction to contributions from its input features.	SHAP explains the model's behavior, not the true causal effect of a variable.		model explanation
Human oversight	Responsible AI	Meaningful human review, responsibility, and ability to question or override an automated recommendation.	High-stakes decisions should not be delegated blindly to a model.		responsible AI
Privacy	Responsible AI	Protection of personal or sensitive information during data collection, analysis, storage, and sharing.	Useful predictions do not justify unnecessary exposure of individual data.		responsible AI
""".strip()


def _parse_glossary():
    entries = []
    for raw_line in GLOSSARY_TSV.splitlines():
        fields = raw_line.split("\t")
        if len(fields) < 4:
            continue
        fields += [""] * (6 - len(fields))
        term, category, definition, why, formula, related = fields[:6]
        entries.append({
            "term": term.strip(),
            "category": category.strip(),
            "definition": definition.strip(),
            "why": why.strip(),
            "formula": formula.strip(),
            "related": [item.strip() for item in related.split(",") if item.strip()],
        })
    return entries


GLOSSARY_ENTRIES = _parse_glossary()
GLOSSARY_BY_TERM = {entry["term"].lower(): entry for entry in GLOSSARY_ENTRIES}


def _model_entry(family, task, how, input_text, output_text, use_when, strengths, limitations, parameters, evaluate):
    return {
        "family": family,
        "task": task,
        "how": how,
        "input": input_text,
        "output": output_text,
        "use_when": use_when,
        "strengths": strengths,
        "limitations": limitations,
        "parameters": parameters,
        "evaluate": evaluate,
    }


MODEL_CATALOG = {
    "Linear Regression": _model_entry(
        "Linear models", "Regression",
        "Fits an intercept and slopes so the predicted numerical target is a weighted sum of the predictors.",
        "Numerical and encoded categorical predictors in one fixed row.", "A numerical prediction.",
        "Use as an interpretable first model when a roughly additive linear relationship is plausible.",
        ["Clear coefficient interpretation", "Fast to train", "Strong baseline"],
        ["Misses nonlinear patterns unless they are engineered", "Sensitive to influential observations", "Coefficients can be unstable under multicollinearity"],
        [("No main complexity control", "Ordinary linear regression estimates coefficients directly. Predictor choice and preprocessing still matter.")],
        "Use held-out mean absolute error, root mean squared error, and R-squared; inspect residual patterns.",
    ),
    "Ridge Regression": _model_entry(
        "Regularized linear models", "Regression",
        "Fits linear coefficients while applying an L2 penalty that shrinks large coefficients toward zero.",
        "A fixed predictor row; numerical predictors should normally be standardized.", "A numerical prediction.",
        "Use when many predictors overlap and a stable linear prediction is preferred.",
        ["Reduces coefficient instability", "Retains all predictors", "Often improves generalization"],
        ["Coefficients are biased toward zero", "Does not automatically remove predictors", "Still primarily linear"],
        [("alpha", "Regularization strength. Larger alpha produces stronger shrinkage and a simpler model.")],
        "Compare held-out error with ordinary linear regression and examine whether stability improves.",
    ),
    "Lasso Regression": _model_entry(
        "Regularized linear models", "Regression",
        "Fits linear coefficients with an L1 penalty that may shrink some coefficients exactly to zero.",
        "A fixed, usually standardized predictor row.", "A numerical prediction and possibly a sparse coefficient set.",
        "Use when a simpler linear model with automatic variable removal is desirable.",
        ["Can create sparse models", "Performs regularization and selection together", "Interpretable"],
        ["Can select unpredictably among correlated predictors", "Too much penalty underfits", "Selection is not proof of scientific importance"],
        [("alpha", "Regularization strength. Larger alpha creates stronger shrinkage and usually more zero coefficients.")],
        "Use cross-validation to choose alpha and report held-out prediction error.",
    ),
    "Elastic Net": _model_entry(
        "Regularized linear models", "Regression",
        "Combines L1 and L2 penalties so coefficients can be both stabilized and set to zero.",
        "A fixed, standardized predictor row.", "A numerical prediction and regularized coefficients.",
        "Use when predictors are numerous and correlated but some sparsity is still useful.",
        ["Balances ridge and lasso behavior", "Handles correlated predictors better than pure lasso in many cases"],
        ["Requires tuning two controls", "Still assumes an additive linear structure"],
        [("alpha", "Overall penalty strength."), ("l1_ratio", "Mix between lasso and ridge: 1 is lasso-like; 0 is ridge-like.")],
        "Tune both controls with cross-validation and compare held-out error.",
    ),
    "Huber Regression": _model_entry(
        "Robust linear models", "Regression",
        "Fits a linear relationship while reducing the influence of observations with very large residuals.",
        "A fixed predictor row.", "A numerical prediction and robust linear coefficients.",
        "Use when unusual observations may dominate ordinary least squares but should not simply be deleted.",
        ["More resistant to outliers", "Retains linear interpretability"],
        ["Does not fix every data-quality problem", "The outlier threshold must be chosen", "Still linear"],
        [("epsilon", "Controls when an error begins receiving reduced influence. Smaller values are more robust."), ("alpha", "L2 regularization strength.")],
        "Compare held-out error and inspect which observations have large residuals.",
    ),
    "Logistic Regression": _model_entry(
        "Linear models", "Classification",
        "Builds a linear score from predictors and converts it through the logistic function into class probabilities.",
        "A fixed row of numerical and encoded categorical predictors.", "Class probabilities and a class label after applying a threshold.",
        "Use as an interpretable baseline for classification and probability estimation.",
        ["Coefficients and odds ratios are interpretable", "Fast", "Produces probabilities"],
        ["Linear decision boundary unless features are engineered", "Threshold choice affects errors", "May struggle with complex interactions"],
        [("C", "Inverse regularization strength. Smaller C means stronger regularization."), ("penalty", "L1 can create sparse coefficients; L2 shrinks them smoothly."), ("class_weight", "Balanced weighting gives more influence to rarer classes.")],
        "Use confusion-matrix metrics, ROC area under the curve, average precision, and calibration when probabilities matter.",
    ),
    "Linear Discriminant Analysis": _model_entry(
        "Discriminant models", "Classification",
        "Estimates class distributions with a shared covariance structure and creates linear class boundaries.",
        "A fixed numerical predictor row after appropriate preprocessing.", "Class probabilities and labels.",
        "Use when classes may be separated linearly and distributional assumptions are acceptable.",
        ["Efficient", "Can work well with modest samples", "Produces probabilities"],
        ["Shared-covariance assumption may be unrealistic", "Sensitive to severe non-normality or unstable covariance estimates"],
        [("solver", "Controls how the covariance and discriminant solution are estimated."), ("shrinkage", "Stabilizes covariance estimates when predictors are numerous.")],
        "Compare held-out class metrics and probability calibration.",
    ),
    "Quadratic Discriminant Analysis": _model_entry(
        "Discriminant models", "Classification",
        "Allows each class to have a separate covariance structure, creating curved decision boundaries.",
        "A fixed numerical predictor row.", "Class probabilities and labels.",
        "Use when class spreads differ and sufficient data are available to estimate them.",
        ["More flexible boundaries than linear discriminant analysis", "Produces probabilities"],
        ["Needs more data", "Covariance estimates can become unstable", "Can overfit"],
        [("reg_param", "Shrinks class covariance estimates toward a more stable structure. Larger values mean more regularization.")],
        "Use held-out metrics and compare against the simpler linear discriminant model.",
    ),
    "Decision Tree": _model_entry(
        "Tree models", "Regression and classification",
        "Repeatedly splits the data using predictor thresholds and makes a prediction within each final leaf.",
        "A fixed predictor row; scaling is usually unnecessary.", "A numerical prediction or class label and probability.",
        "Use when nonlinear relationships and interactions are expected and transparent rules are valuable.",
        ["Easy to visualize", "Captures interactions", "Handles nonlinearities"],
        ["Can overfit sharply", "Small data changes may produce a different tree", "Single-tree predictions can be unstable"],
        [("max_depth", "Maximum number of split levels. Larger values create more complex trees."), ("min_samples_leaf", "Minimum observations in a final leaf. Larger values smooth predictions.")],
        "Compare training and held-out performance to detect overfitting.",
    ),
    "Random Forest": _model_entry(
        "Tree ensembles", "Regression and classification",
        "Fits many trees to resampled data with randomized predictor subsets and averages or votes across them.",
        "A fixed predictor row; scaling is usually unnecessary.", "A numerical prediction or class probability and label.",
        "Use as a strong general-purpose nonlinear model when interactions are likely.",
        ["Stable compared with one tree", "Handles nonlinearities and interactions", "Little preprocessing"],
        ["Less interpretable than one tree", "Can be computationally heavier", "Importance can be shared among correlated predictors"],
        [("number of trees", "More trees usually stabilize results but increase computation."), ("max_depth", "Limits individual tree complexity."), ("max_features", "Controls how many predictors compete at each split."), ("min_samples_leaf", "Larger leaves reduce variance and smooth predictions.")],
        "Use held-out metrics, compare with a simple baseline, and inspect permutation importance cautiously.",
    ),
    "Gradient Boosting": _model_entry(
        "Boosted tree ensembles", "Regression and classification",
        "Builds shallow trees sequentially so each new stage focuses on errors left by the earlier ensemble.",
        "A fixed predictor row; scaling is usually unnecessary.", "A numerical prediction or class probability and label.",
        "Use when strong nonlinear predictive performance is needed and tuning time is available.",
        ["Often highly accurate", "Captures interactions", "Flexible"],
        ["Can overfit", "Learning rate and number of stages interact", "Sequential training is slower than independent trees"],
        [("boosting stages", "Number of trees added sequentially."), ("learning_rate", "Contribution of each new tree. Smaller values usually need more stages."), ("max_depth", "Complexity of each weak tree.")],
        "Tune on training folds and compare against random forest and linear baselines on the same holdout.",
    ),
    "K-Nearest Neighbors": _model_entry(
        "Distance-based models", "Regression and classification",
        "Finds the most similar training rows and averages their targets or votes among their classes.",
        "A fixed, usually standardized predictor row.", "A numerical prediction or class probability and label.",
        "Use for local patterns when distance between standardized observations is meaningful.",
        ["Simple idea", "Nonlinear without fitting a global equation", "Useful teaching model"],
        ["Prediction can be slow", "Sensitive to scale and irrelevant predictors", "Performance deteriorates in very high dimensions"],
        [("number of neighbors", "Small values create flexible, noisy predictions; large values create smoother predictions."), ("weights", "Distance weighting gives closer neighbors more influence.")],
        "Choose the number of neighbors with cross-validation and compare held-out error.",
    ),
    "Support Vector Machine": _model_entry(
        "Margin and kernel models", "Classification",
        "Finds a boundary with a wide margin; kernels can represent nonlinear separation without explicitly creating every transformed feature.",
        "A fixed, standardized predictor row.", "A class score, probability estimate, and label.",
        "Use for moderate-sized datasets with potentially complex boundaries or many predictors.",
        ["Effective in high dimensions", "Flexible kernels", "Strong margin principle"],
        ["Needs scaling", "Parameters can be sensitive", "Less transparent", "Training can become slow on large data"],
        [("C", "Penalty for classification errors. Larger C fits training data more strongly and may reduce the margin."), ("kernel", "Defines linear or nonlinear similarity."), ("gamma", "For nonlinear kernels, controls how local each training point's influence is."), ("class_weight", "Can increase attention to rare classes.")],
        "Tune using cross-validation and evaluate class metrics on untouched data.",
    ),
    "Support Vector Regression": _model_entry(
        "Margin and kernel models", "Regression",
        "Fits a function that ignores small errors inside an epsilon tube while penalizing larger deviations.",
        "A fixed, standardized predictor row.", "A numerical prediction.",
        "Use for nonlinear regression on moderate-sized datasets.",
        ["Flexible kernels", "Can be robust to small deviations", "Strong predictive model"],
        ["Needs scaling and tuning", "Can be slow", "Less interpretable"],
        [("C", "Strength of the penalty for errors outside the epsilon tube."), ("epsilon", "Width of the no-penalty error tube."), ("kernel", "Defines the functional shape."), ("gamma", "Controls locality for nonlinear kernels.")],
        "Tune on training folds and evaluate held-out mean absolute error and root mean squared error.",
    ),
    "Feedforward Neural Network (FFNN)": _model_entry(
        "Neural networks", "Regression and classification",
        "Passes a fixed predictor vector through one or more hidden layers that learn nonlinear combinations before the output layer.",
        "One fixed, standardized predictor row; images are flattened when this model is used for image classification.", "A numerical prediction or class probability and label.",
        "Use to demonstrate general nonlinear neural learning when a fixed-length input is available.",
        ["Learns nonlinear relationships", "Flexible architecture", "Works for regression and classification"],
        ["Needs scaling and tuning", "Can overfit", "Does not preserve image spatial structure or time memory by itself"],
        [("hidden layers", "Number and size of intermediate representations."), ("activation", "Nonlinear transformation used by hidden neurons."), ("alpha", "L2 regularization strength."), ("learning rate", "Initial weight-update step size."), ("maximum iterations", "Upper limit on training updates.")],
        "Use held-out metrics, learning behavior, and comparison with simpler models.",
    ),
    "Long Short-Term Memory Recurrent Neural Network (LSTM-RNN)": _model_entry(
        "Neural networks", "Time-series regression",
        "Processes an ordered lookback sequence with gated recurrent memory before predicting a future target.",
        "A sequence shaped as time steps by input channels.", "A numerical future prediction.",
        "Use when ordered dependencies across several past time steps may matter.",
        ["Learns sequential patterns", "Can combine target history with numerical external inputs"],
        ["Needs more data and computation", "Can overfit", "Longer memory does not guarantee improvement", "Must preserve time order"],
        [("lookback", "Number of past time steps supplied for each prediction."), ("memory units", "Size of the recurrent hidden representation."), ("dropout", "Regularization applied during training."), ("learning rate", "Weight-update step size."), ("epochs", "Maximum training passes."), ("batch size", "Sequences used in one update."), ("patience", "Epochs without validation improvement before stopping.")],
        "Use chronological backtesting and compare with latest-value, seasonal-naïve, and historical-mean baselines.",
    ),
    "Small Convolutional Neural Network (CNN)": _model_entry(
        "Neural networks", "Image classification",
        "Keeps the image as a height-by-width-by-channel grid and learns local filters, pooling summaries, and a final class probability layer.",
        "A resized spatial RGB image array.", "A probability for every image class and a predicted label.",
        "Use to teach why an image-specialized network differs from flattened-pixel classifiers.",
        ["Preserves spatial structure", "Learns edges, textures, and shapes", "End-to-end image features"],
        ["Needs TensorFlow and more computation", "Small image datasets can overfit", "Results depend on honest image splitting"],
        [("dense neurons", "Size of the final learned representation."), ("dropout", "Regularization before the output."), ("learning rate", "Optimizer step size."), ("epochs", "Maximum training passes."), ("batch size", "Images used per update."), ("patience", "Early-stopping wait after validation stops improving.")],
        "Use validation or final-test accuracy, weighted F1-score, and a confusion matrix with a leakage-safe split.",
    ),
    "Pretrained MobileNetV2": _model_entry(
        "Transfer learning", "General image recognition",
        "Uses convolutional features learned from the large ImageNet dataset to assign probabilities to known ImageNet categories.",
        "An RGB image resized and preprocessed to the network's expected shape.", "Probabilities for ImageNet labels.",
        "Use for instant demonstration of pretrained recognition, not as a specialist diagnosis.",
        ["Fast demonstration", "Rich visual features", "No classroom training required"],
        ["Limited to pretrained label vocabulary", "May not match the new context", "Confidence is not certainty"],
        [("top predictions", "How many highest-probability labels to display.")],
        "Inspect whether the labels are sensible and explain domain mismatch and uncertainty.",
    ),
}


COURSE_TOPIC_SOURCES = {
    "AI and Machine Learning Foundations": [],
    "Probability": ["Week 1"],
    "Data and Research Questions": ["Week 2"],
    "Visualization and Descriptive Statistics": ["Week 3"],
    "Association and Correlation": ["Week 4"],
    "Linear Regression": ["Week 5", "Week 6"],
    "Machine Learning Regression": ["Week 7", "Week 11"],
    "Classification and Logistic Regression": ["Week 9"],
    "Machine Learning Classification": ["Week 10", "Week 11"],
    "Neural Networks": ["Week 7", "Week 10", "Week 14", "Week 15"],
    "Model Evaluation": ["Week 11"],
    "Cross-Validation and Leakage": ["Week 12"],
    "Bootstrap and Uncertainty": ["Week 13"],
    "Time Series Forecasting": ["Week 14"],
    "Computer Vision": ["Week 15"],
    "Full Course Mixed Review": [],
}


COURSE_TOPIC_KEYWORDS = {
    "AI and Machine Learning Foundations": ["artificial intelligence", "machine learning", "supervised", "algorithm", "model", "training", "test", "generaliz", "overfit", "underfit", "hyperparameter", "regularization", "tuning", "neural network", "support vector"],
    "Probability": ["probability", "sample space", "event", "complement", "conditional", "independent", "expected", "variance", "coin", "Bayes"],
    "Data and Research Questions": ["research question", "target", "predictor", "feature", "row", "column", "leakage", "numerical", "categorical", "missing"],
    "Visualization and Descriptive Statistics": ["mean", "median", "standard deviation", "distribution", "histogram", "boxplot", "scatterplot", "bar chart", "outlier", "spread"],
    "Association and Correlation": ["correlation", "association", "Pearson", "Spearman", "Kendall", "partial", "eta", "Cram", "caus"],
    "Linear Regression": ["linear regression", "linear model", "linear models", "slope", "intercept", "residual", "coefficient", "R-squared", "multicollinearity", "variance inflation"],
    "Machine Learning Regression": ["regression", "random forest", "gradient boosting", "decision tree", "nearest", "support vector regression", "root mean squared", "mean absolute", "baseline"],
    "Classification and Logistic Regression": ["logistic", "linear classification", "linear model", "classification", "threshold", "odds", "positive class", "confusion", "precision", "recall", "F1", "ROC"],
    "Machine Learning Classification": ["classifier", "classification", "random forest", "gradient boosting", "support vector", "nearest", "decision tree", "class imbalance", "balanced accuracy"],
    "Neural Networks": ["neural", "FFNN", "LSTM", "CNN", "hidden layer", "activation", "epoch", "learning rate", "dropout", "batch", "lookback", "convolution"],
    "Model Evaluation": ["accuracy", "balanced accuracy", "precision", "recall", "F1", "ROC", "average precision", "calibration", "mean absolute", "root mean squared", "R-squared", "baseline", "held-out"],
    "Cross-Validation and Leakage": ["cross-validation", "fold", "strat", "nested", "leakage", "split", "preprocessing", "group", "time order"],
    "Bootstrap and Uncertainty": ["bootstrap", "confidence interval", "standard error", "bias", "resample", "uncertainty", "percentile"],
    "Time Series Forecasting": ["forecast", "time series", "lag", "season", "horizon", "lookback", "LSTM", "chronological", "naïve", "rolling"],
    "Computer Vision": ["image", "pixel", "channel", "convolution", "CNN", "transfer learning", "augmentation", "vision", "spatial", "sequence leakage"],
    "Full Course Mixed Review": [],
}


EXAM_CALCULATION_QUESTIONS = [
    {"topic":"Probability","concept":"Complement rule","difficulty":"Calculation","question":"If P(A)=0.37, what is P(A does not occur)?","options":["0.63","0.37","1.37","0.73"],"answer":"0.63","explanation":"Use the complement rule: 1-0.37=0.63."},
    {"topic":"Probability","concept":"Experimental probability","difficulty":"Calculation","question":"An event occurs 72 times in 120 trials. What is its experimental probability?","options":["0.60","0.72","0.48","1.67"],"answer":"0.60","explanation":"Experimental probability is successes divided by trials: 72/120=0.60."},
    {"topic":"Probability","concept":"Independent events","difficulty":"Calculation","question":"Independent events have P(A)=0.40 and P(B)=0.50. What is P(A and B)?","options":["0.20","0.90","0.45","0.10"],"answer":"0.20","explanation":"For independent events, multiply: 0.40 x 0.50=0.20."},
    {"topic":"Probability","concept":"Conditional multiplication rule","difficulty":"Calculation","question":"If P(A)=0.60 and P(B|A)=0.50, what is P(A and B)?","options":["0.30","1.10","0.55","0.10"],"answer":"0.30","explanation":"P(A and B)=P(A)P(B|A)=0.60 x 0.50=0.30."},
    {"topic":"Probability","concept":"Addition rule","difficulty":"Calculation","question":"Mutually exclusive events have probabilities 0.30 and 0.20. What is the probability that either occurs?","options":["0.50","0.06","0.10","0.70"],"answer":"0.50","explanation":"With no overlap, add the probabilities: 0.30+0.20=0.50."},
    {"topic":"Probability","concept":"Expected count","difficulty":"Calculation","question":"A defect probability is 0.15. Among 200 independent items, what is the expected number of defects?","options":["30","15","13.3","185"],"answer":"30","explanation":"Expected count is n times p: 200 x 0.15=30."},
    {"topic":"Probability","concept":"Expected value","difficulty":"Calculation","question":"A game pays $10 with probability 0.30 and $0 otherwise. What is the expected payout?","options":["$3","$7","$10","$0.30"],"answer":"$3","explanation":"E(X)=10(0.30)+0(0.70)=$3."},
    {"topic":"Probability","concept":"Binomial reasoning","difficulty":"Calculation","question":"A fair coin is tossed three times. What is the probability of exactly two heads?","options":["3/8","1/8","1/2","2/3"],"answer":"3/8","explanation":"There are three arrangements with two heads out of eight equally likely outcomes: HHT, HTH, and THH."},
    {"topic":"Visualization and Descriptive Statistics","concept":"Mean","difficulty":"Calculation","question":"What is the mean of 2, 4, 6, and 8?","options":["5","4","6","20"],"answer":"5","explanation":"Add the values and divide by four: 20/4=5."},
    {"topic":"Visualization and Descriptive Statistics","concept":"Median","difficulty":"Calculation","question":"What is the median of 1, 3, 9, 10, and 12?","options":["9","7","10","3"],"answer":"9","explanation":"After ordering, the middle of five values is 9."},
    {"topic":"Visualization and Descriptive Statistics","concept":"Range","difficulty":"Calculation","question":"What is the range of 4, 7, and 11?","options":["7","11","4","22"],"answer":"7","explanation":"Range equals maximum minus minimum: 11-4=7."},
    {"topic":"Linear Regression","concept":"Prediction equation","difficulty":"Calculation","question":"For the fitted line y-hat=3+2x, what is the prediction when x=4?","options":["11","8","7","5"],"answer":"11","explanation":"Substitute x=4: 3+2(4)=11."},
    {"topic":"Linear Regression","concept":"Residual","difficulty":"Calculation","question":"An observed value is 10 and the model predicts 8. Using residual=observed-predicted, what is the residual?","options":["2","-2","18","0.8"],"answer":"2","explanation":"Residual=10-8=2, so the model underpredicted by two units."},
    {"topic":"Linear Regression","concept":"Slope","difficulty":"Application","question":"A slope is 2.5 for hours studied predicting score. Which interpretation is correct?","options":["Each additional hour is associated with a 2.5-point increase in predicted score","Every student gains exactly 2.5 points","The model error is 2.5 points","The intercept is 2.5"],"answer":"Each additional hour is associated with a 2.5-point increase in predicted score","explanation":"A slope describes the fitted average change in prediction for a one-unit predictor increase; it does not guarantee an individual causal effect."},
    {"topic":"Model Evaluation","concept":"Mean absolute error","difficulty":"Calculation","question":"Prediction errors are 2, -4, and 0. What is the mean absolute error?","options":["2","0.67","6","4"],"answer":"2","explanation":"Take absolute values and average: (2+4+0)/3=2."},
    {"topic":"Model Evaluation","concept":"Root mean squared error","difficulty":"Calculation","question":"Squared prediction errors are 1, 4, and 9. What is the root mean squared error, rounded to two decimals?","options":["2.16","4.67","3.00","1.56"],"answer":"2.16","explanation":"RMSE=sqrt((1+4+9)/3)=sqrt(4.6667)=2.16."},
    {"topic":"Model Evaluation","concept":"Accuracy","difficulty":"Calculation","question":"A classifier makes 80 correct predictions out of 100. What is its accuracy?","options":["0.80","0.20","80","1.25"],"answer":"0.80","explanation":"Accuracy=80/100=0.80."},
    {"topic":"Model Evaluation","concept":"Precision","difficulty":"Calculation","question":"A model has 40 true positives and 10 false positives. What is precision?","options":["0.80","0.67","0.40","0.50"],"answer":"0.80","explanation":"Precision=TP/(TP+FP)=40/50=0.80."},
    {"topic":"Model Evaluation","concept":"Recall","difficulty":"Calculation","question":"A model has 40 true positives and 20 false negatives. What is recall?","options":["0.67","0.80","0.50","0.33"],"answer":"0.67","explanation":"Recall=TP/(TP+FN)=40/60=0.667."},
    {"topic":"Model Evaluation","concept":"F1-score","difficulty":"Calculation","question":"Precision is 0.75 and recall is 0.60. What is the F1-score, rounded to two decimals?","options":["0.67","0.68","0.45","1.35"],"answer":"0.67","explanation":"F1=2(0.75)(0.60)/(0.75+0.60)=0.667."},
    {"topic":"Model Evaluation","concept":"Balanced accuracy","difficulty":"Calculation","question":"Sensitivity is 0.80 and specificity is 0.60. What is balanced accuracy?","options":["0.70","0.48","1.40","0.20"],"answer":"0.70","explanation":"Balanced accuracy is their average: (0.80+0.60)/2=0.70."},
    {"topic":"Classification and Logistic Regression","concept":"Odds","difficulty":"Calculation","question":"A predicted probability is 0.75. What are the corresponding odds of the positive class?","options":["3 to 1","1 to 3","0.75 to 1","4 to 1"],"answer":"3 to 1","explanation":"Odds=p/(1-p)=0.75/0.25=3, or 3 to 1."},
    {"topic":"Classification and Logistic Regression","concept":"Decision threshold","difficulty":"Application","question":"What usually happens when a binary decision threshold is lowered, with all else fixed?","options":["Recall tends to increase while precision may decrease","Recall and precision must both increase","Fewer cases are predicted positive","The model is retrained automatically"],"answer":"Recall tends to increase while precision may decrease","explanation":"A lower threshold labels more cases positive, finding more true positives but often adding false positives."},
    {"topic":"Cross-Validation and Leakage","concept":"Cross-validation mean","difficulty":"Calculation","question":"Three validation scores are 0.70, 0.80, and 0.90. What is their mean?","options":["0.80","0.70","0.90","2.40"],"answer":"0.80","explanation":"(0.70+0.80+0.90)/3=0.80."},
    {"topic":"Cross-Validation and Leakage","concept":"Preprocessing leakage","difficulty":"Application","question":"Which workflow avoids scaling leakage during cross-validation?","options":["Fit the scaler separately inside each training fold","Scale the complete dataset before creating folds","Fit the scaler using the final test set","Remove every predictor that needs scaling"],"answer":"Fit the scaler separately inside each training fold","explanation":"Each validation fold must remain unseen while preprocessing is learned from the corresponding training folds."},
    {"topic":"Bootstrap and Uncertainty","concept":"Bootstrap bias","difficulty":"Calculation","question":"The original estimate is 50 and the mean bootstrap estimate is 51.2. What is estimated bootstrap bias?","options":["1.2","-1.2","101.2","0.024"],"answer":"1.2","explanation":"Bias=bootstrap mean-original estimate=51.2-50=1.2."},
    {"topic":"Bootstrap and Uncertainty","concept":"Confidence interval","difficulty":"Application","question":"A bootstrap slope interval is [0.20, 1.50]. What does it indicate in these resamples?","options":["The interval is entirely positive and supports a positive association","The slope is exactly 0.85","Twenty percent of observations are positive","The model proves causation"],"answer":"The interval is entirely positive and supports a positive association","explanation":"Zero is not inside the interval, so the resampled slope estimates are consistently positive; this does not prove causation."},
    {"topic":"Time Series Forecasting","concept":"Forecast horizon","difficulty":"Calculation","question":"With daily rows, what does a forecast horizon of 7 represent?","options":["Seven days ahead","Seven months ahead","The previous seven days only","Seven predictors"],"answer":"Seven days ahead","explanation":"The horizon counts future rows, so seven daily rows represent seven days."},
    {"topic":"Time Series Forecasting","concept":"Seasonal cycle length","difficulty":"Calculation","question":"For hourly data with a repeating daily pattern, what seasonal cycle length should be considered?","options":["24 rows","7 rows","12 rows","365 rows"],"answer":"24 rows","explanation":"There are 24 hourly observations in one day."},
    {"topic":"Time Series Forecasting","concept":"Forecast baseline","difficulty":"Application","question":"A neural network has test RMSE 3.0 and the seasonal-naïve baseline has RMSE 2.5. Which conclusion is correct?","options":["The seasonal-naïve baseline performs better","The neural network performs better because it is more complex","Both models have equal error","RMSE cannot compare forecasting methods"],"answer":"The seasonal-naïve baseline performs better","explanation":"Lower RMSE is better; the complex model did not add predictive value in this run."},
    {"topic":"Computer Vision","concept":"Image tensor","difficulty":"Application","question":"Why does a convolutional neural network usually outperform a flattened feedforward network on spatial image patterns?","options":["It preserves local pixel neighborhoods and learns shared filters","It sees the test labels during training","It removes every background automatically","It uses no parameters"],"answer":"It preserves local pixel neighborhoods and learns shared filters","explanation":"Convolution exploits two-dimensional spatial structure and weight sharing."},
    {"topic":"Computer Vision","concept":"Flattened image features","difficulty":"Calculation","question":"How many raw RGB pixel values are in a 32 by 32 image before color summaries are added?","options":["3,072","1,024","96","32,768"],"answer":"3,072","explanation":"An RGB image has 32 x 32 x 3 values, which equals 3,072."},
    {"topic":"Computer Vision","concept":"Pooling","difficulty":"Calculation","question":"A 64 by 64 feature map is reduced by 2 by 2 pooling with stride 2. What is the new spatial size?","options":["32 by 32","62 by 62","16 by 16","128 by 128"],"answer":"32 by 32","explanation":"Pooling with stride 2 halves both height and width."},
    {"topic":"Computer Vision","concept":"Image channels","difficulty":"Foundation","question":"What does the number 3 usually represent in an RGB image shape of 64 by 64 by 3?","options":["Red, green, and blue channels","Three image classes","Three convolution layers","Three test folds"],"answer":"Red, green, and blue channels","explanation":"The final dimension stores the three color channels."},
    {"topic":"Computer Vision","concept":"Group-aware evaluation","difficulty":"Application","question":"Images paper01-000 and paper01-005 come from the same capture sequence. What is the safest split?","options":["Keep the entire paper01 sequence in either training or evaluation","Randomly place each frame independently","Duplicate the frames in both sets","Use the class label as the group"],"answer":"Keep the entire paper01 sequence in either training or evaluation","explanation":"A group-aware split prevents nearly identical sequence frames from leaking across the boundary."},
    {"topic":"Computer Vision","concept":"Validation accuracy","difficulty":"Calculation","question":"A model correctly classifies 45 of 60 independent validation images. What is validation accuracy?","options":["0.75","0.60","0.45","1.33"],"answer":"0.75","explanation":"Accuracy is 45/60=0.75."},
    {"topic":"Computer Vision","concept":"Transfer learning","difficulty":"Application","question":"Why can transfer learning help when only a small labeled image dataset is available?","options":["The pretrained network already contains useful visual feature detectors","It guarantees perfect accuracy","It allows test images to be used for training","It removes the need for evaluation"],"answer":"The pretrained network already contains useful visual feature detectors","explanation":"Previously learned edges, textures, and shapes can be adapted instead of learned entirely from scratch."},
    {"topic":"Computer Vision","concept":"Data augmentation","difficulty":"Application","question":"Which augmentation is usually reasonable for ordinary object photographs when label meaning is unchanged?","options":["A small rotation or horizontal flip","Replacing the image with its class name","Copying evaluation images into training","Changing every image to random noise"],"answer":"A small rotation or horizontal flip","explanation":"Plausible transformations expose the model to variation without changing the label."},
    {"topic":"Computer Vision","concept":"CNN output","difficulty":"Foundation","question":"For a three-class convolutional neural network, what should the final softmax layer return?","options":["Three class probabilities that sum to one","One regression slope","The original image size only","A single unbounded number"],"answer":"Three class probabilities that sum to one","explanation":"Softmax converts the output scores into one probability for each class."},
]


TOPIC_ADVICE = {
    "AI and Machine Learning Foundations": "Review the distinctions among artificial intelligence, machine learning, algorithms, fitted models, parameters, hyperparameters, overfitting, and generalization.",
    "Probability": "Write the sample space first, identify the probability rule, substitute values carefully, and check that the result lies between zero and one.",
    "Data and Research Questions": "Practise identifying one outcome, predictors available at decision time, the unit represented by a row, and possible leakage.",
    "Visualization and Descriptive Statistics": "Match the plot to the variable types, read both axes and units, and distinguish center, spread, shape, and unusual observations.",
    "Association and Correlation": "Review which association measure matches each variable combination and remember that association does not establish causation.",
    "Linear Regression": "Practise the prediction equation, slope and intercept language, residual calculations, and the difference between simple and multiple regression.",
    "Machine Learning Regression": "Compare each nonlinear model with a mean and linear baseline using the same held-out rows; review complexity controls and overfitting.",
    "Classification and Logistic Regression": "Draw a confusion matrix, identify the positive class, calculate precision and recall, and explain how the threshold changes errors.",
    "Machine Learning Classification": "Review class imbalance, scaling requirements, model-specific boundaries, and fair comparison on the same evaluation data.",
    "Neural Networks": "Trace the flow from inputs through hidden units to outputs; then review activation, learning rate, epochs, batches, regularization, and architecture-specific inputs.",
    "Model Evaluation": "State whether higher or lower is better for each metric, calculate it from a small example, and connect the metric to the real cost of an error.",
    "Cross-Validation and Leakage": "Practise drawing the folds and placing preprocessing, tuning, groups, and time order inside the correct part of the validation design.",
    "Bootstrap and Uncertainty": "Separate the original estimate, bias, standard error, and interval; remember that an interval concerns an estimated quantity, not individual observations.",
    "Time Series Forecasting": "Translate rows into time units, identify the forecast origin, horizon, lags, seasonal cycle, and compare every model with realistic naïve baselines.",
    "Computer Vision": "Review pixels, channels, flattened features, convolution, pooling, honest train/validation/test separation, and sequence-level leakage.",
    "Full Course Mixed Review": "Use the question review to identify two weak topic areas, then complete a focused practice set for each one.",
}


def _question_text(item):
    return (str(item.get("question", "")) + " " + str(item.get("explanation", ""))).lower()


def _infer_question_difficulty(item):
    text = _question_text(item)
    if re.search(r"\bwhat is (the )?(mean|probability|precision|recall|f1|accuracy|residual|rmse|bias)|calculate|rounded|given .*\d", text):
        return "Calculation"
    if any(word in text for word in ["why", "which situation", "best evidence", "what happens", "which workflow", "interpret", "most appropriate", "should"]):
        return "Application"
    return "Foundation"


def _infer_question_concept(item):
    text = _question_text(item)
    ordered = [
        ("Convolutional neural network", ["convolution", "cnn"]),
        ("Long short-term memory", ["lstm", "lookback"]),
        ("Feedforward neural network", ["feedforward", "ffnn", "hidden layer", "activation"]),
        ("Bootstrap", ["bootstrap", "confidence interval", "standard error"]),
        ("Cross-validation", ["cross-validation", "fold", "nested"]),
        ("Data leakage", ["leakage", "future information", "group-aware"]),
        ("Precision and recall", ["precision", "recall", "f1"]),
        ("Confusion matrix", ["confusion matrix", "false positive", "false negative", "true positive"]),
        ("Regression error", ["root mean squared", "mean absolute", "residual"]),
        ("Linear regression", ["linear regression", "slope", "intercept", "coefficient"]),
        ("Probability", ["probability", "sample space", "conditional", "complement", "expected value"]),
        ("Forecasting", ["forecast", "lag", "season", "horizon", "naïve"]),
        ("Computer vision", ["image", "pixel", "channel", "augmentation", "transfer learning"]),
        ("Model complexity", ["overfit", "underfit", "regularization", "hyperparameter"]),
        ("Classification", ["classification", "logistic", "threshold", "class"]),
        ("Association", ["correlation", "association", "pearson", "spearman", "eta", "cram"]),
        ("Descriptive statistics", ["mean", "median", "histogram", "boxplot", "distribution", "outlier"]),
        ("Data and research", ["target", "predictor", "research question", "row", "column"]),
    ]
    for concept, keywords in ordered:
        if any(keyword in text for keyword in keywords):
            return concept
    return "General course concept"


def _base_course_questions():
    rows = []
    for week in [f"Week {number}" for number in range(1, 16) if number != 8]:
        for item in week_wrap_questions(week, {}, None):
            q = dict(item)
            q["source"] = week
            q["difficulty"] = q.get("difficulty") or _infer_question_difficulty(q)
            q["concept"] = q.get("concept") or _infer_question_concept(q)
            rows.append(q)
    for review_week, questions in REVIEW_WRAP_UP_QUESTIONS.items():
        for item in questions:
            q = dict(item)
            q["source"] = review_week
            q["difficulty"] = q.get("difficulty") or _infer_question_difficulty(q)
            q["concept"] = q.get("concept") or _infer_question_concept(q)
            rows.append(q)
    return rows


def exam_question_pool(topic, difficulties=None):
    difficulties = set(difficulties or ["Foundation", "Application", "Calculation"])
    all_questions = _base_course_questions()
    selected = []
    if topic == "Full Course Mixed Review":
        selected.extend(all_questions)
    else:
        source_weeks = set(COURSE_TOPIC_SOURCES.get(topic, []))
        keywords = [word.lower() for word in COURSE_TOPIC_KEYWORDS.get(topic, [])]
        for item in all_questions:
            text = _question_text(item)
            if item.get("source") in source_weeks or any(word in text for word in keywords):
                selected.append(item)
    for item in EXAM_CALCULATION_QUESTIONS:
        if topic == "Full Course Mixed Review" or item.get("topic") == topic:
            selected.append(dict(item, source="Exam practice calculation bank"))
    unique = []
    seen = set()
    for item in selected:
        question = str(item.get("question", "")).strip()
        if not question or question.lower() in seen:
            continue
        if item.get("difficulty", _infer_question_difficulty(item)) not in difficulties:
            continue
        answer = item.get("answer")
        options = list(dict.fromkeys(item.get("options", [])))
        if answer not in options or len(options) < 2:
            continue
        item = dict(item)
        item["difficulty"] = item.get("difficulty") or _infer_question_difficulty(item)
        item["concept"] = item.get("concept") or _infer_question_concept(item)
        item["options"] = options
        unique.append(item)
        seen.add(question.lower())
    return unique


def _safe_select_state(key, options):
    if options and st.session_state.get(key) not in options:
        st.session_state[key] = options[0]


def _glossary_matches(search, category):
    words = [word for word in re.split(r"\s+", search.lower().strip()) if word]
    matches = []
    for entry in GLOSSARY_ENTRIES:
        if category != "All topics" and entry["category"] != category:
            continue
        haystack = " ".join([
            entry["term"], entry["category"], entry["definition"], entry["why"],
            " ".join(entry["related"]),
        ]).lower()
        if all(word in haystack for word in words):
            matches.append(entry)
    return sorted(matches, key=lambda item: item["term"].lower())


def _open_library_for_term(term=""):
    st.session_state["v65_space"] = "Learning Library"
    st.session_state["library_search"] = term


def learning_library_page():
    st.title("📚 Learning Library")
    st.markdown(
        '<div class="simple-note"><strong>Find one idea at a time.</strong><br>'
        'Search the glossary, study how a selected model works, or follow the course map. '
        'The library is independent of the current dataset.</div>',
        unsafe_allow_html=True,
    )
    glossary_tab, model_tab, map_tab = st.tabs(["Glossary", "Model Explorer", "Course Map"])

    with glossary_tab:
        st.subheader("Search the course glossary")
        c1, c2 = st.columns([1.5, 1])
        with c1:
            search = st.text_input(
                "Search a term or idea",
                key="library_search",
                placeholder="Examples: probability, linear models, leakage, neural network, precision",
            )
        with c2:
            categories = ["All topics"] + sorted({entry["category"] for entry in GLOSSARY_ENTRIES})
            category = st.selectbox("Topic group", categories, key="library_category")
        matches = _glossary_matches(search, category)
        c1, c2, c3 = st.columns(3)
        c1.metric("Glossary terms", len(GLOSSARY_ENTRIES))
        c2.metric("Matching terms", len(matches))
        c3.metric("Topic groups", len({entry['category'] for entry in GLOSSARY_ENTRIES}))
        if not matches:
            st.warning("No glossary term matches this search. Try a shorter word or select All topics.")
        else:
            names = [entry["term"] for entry in matches]
            _safe_select_state("library_selected_term", names)
            selected_name = st.selectbox("Choose one matching term", names, key="library_selected_term")
            entry = next(item for item in matches if item["term"] == selected_name)
            st.markdown(
                f'<div class="library-card"><span class="tiny-label">{entry["category"]}</span>'
                f'<h3>{entry["term"]}</h3><p>{entry["definition"]}</p></div>',
                unsafe_allow_html=True,
            )
            st.markdown("**Why it matters**")
            st.write(entry["why"])
            if entry.get("formula"):
                st.markdown("**Formula or notation**")
                st.code(entry["formula"], language=None)
            if entry.get("related"):
                st.markdown("**Related terms**")
                st.write(" · ".join(entry["related"]))
            with st.expander("Show all matching terms", expanded=False):
                st.dataframe(
                    pd.DataFrame([{"Term": item["term"], "Topic": item["category"]} for item in matches]),
                    use_container_width=True,
                    hide_index=True,
                )

    with model_tab:
        st.subheader("Understand one model at a time")
        c1, c2 = st.columns([1.5, 1])
        with c1:
            model_search = st.text_input(
                "Search model name, family, or task",
                key="model_library_search",
                placeholder="Examples: linear, classification, neural network, image",
            )
        with c2:
            families = ["All model families"] + sorted({entry["family"] for entry in MODEL_CATALOG.values()})
            family = st.selectbox("Model family", families, key="model_library_family")
        candidates = []
        words = [word for word in model_search.lower().split() if word]
        for name, entry in MODEL_CATALOG.items():
            if family != "All model families" and entry["family"] != family:
                continue
            haystack = " ".join([name, entry["family"], entry["task"], entry["how"], entry["use_when"]]).lower()
            if all(word in haystack for word in words):
                candidates.append(name)
        candidates = sorted(candidates)
        if not candidates:
            st.warning("No model matches this search.")
        else:
            _safe_select_state("model_library_selected", candidates)
            model_name = st.selectbox("Choose a model", candidates, key="model_library_selected")
            model = MODEL_CATALOG[model_name]
            st.markdown(
                f'<div class="model-card"><span class="tiny-label">{model["family"]} · {model["task"]}</span>'
                f'<h3>{model_name}</h3><p>{model["how"]}</p></div>',
                unsafe_allow_html=True,
            )
            how_tab, parameter_tab, use_tab = st.tabs(["How it works", "Key parameters", "Use and evaluate"])
            with how_tab:
                st.markdown("**Input**")
                st.write(model["input"])
                st.markdown("**Output**")
                st.write(model["output"])
                st.markdown("**Conceptual flow**")
                st.code(f"Input data  →  {model_name}  →  {model['output']}", language=None)
            with parameter_tab:
                rows = [{"Parameter": name, "What it controls": meaning} for name, meaning in model["parameters"]]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption("Parameters shown here match the teaching controls used by the app whenever that model is available.")
            with use_tab:
                st.markdown("**Use it when**")
                st.write(model["use_when"])
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Strengths**")
                    for item in model["strengths"]:
                        st.markdown(f"- {item}")
                with c2:
                    st.markdown("**Limitations**")
                    for item in model["limitations"]:
                        st.markdown(f"- {item}")
                st.markdown("**How to evaluate it**")
                st.write(model["evaluate"])

    with map_tab:
        st.subheader("Course learning map")
        st.caption("Use the map to see how one week's idea prepares you for the next.")
        for week, lab in WEEKLY_LABS.items():
            st.markdown(
                f'<div class="course-map-card"><span class="lesson-number">{week.split()[-1]}</span>'
                f'<strong>{lab["title"]}</strong><br><span>{lab["learn"]}</span></div>',
                unsafe_allow_html=True,
            )


def _exam_signature(topic, count, difficulties, attempt_number):
    return hashlib.sha256(
        json.dumps([topic, count, sorted(difficulties), attempt_number], ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


def _new_exam_attempt(topic, count, difficulties, attempt_number):
    pool = exam_question_pool(topic, difficulties)
    if not pool:
        return None
    count = min(int(count), 20, len(pool))
    seed = int(hashlib.sha256(f"{topic}|{attempt_number}|{count}|{'|'.join(sorted(difficulties))}".encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    selected_indexes = rng.choice(len(pool), size=count, replace=False)
    selected = []
    for index in selected_indexes:
        item = dict(pool[int(index)])
        options = list(item["options"])
        options = [options[i] for i in rng.permutation(len(options))]
        item["options"] = options
        selected.append(item)
    return {
        "signature": _exam_signature(topic, count, difficulties, attempt_number),
        "topic": topic,
        "difficulties": list(difficulties),
        "attempt_number": int(attempt_number),
        "questions": selected,
        "index": 0,
        "score": 0,
        "responses": [],
        "feedback_pending": False,
        "complete": False,
    }


def _exam_score_advice(score, total):
    fraction = score / total if total else 0
    if fraction >= 0.90:
        return "Excellent command of this topic. Explain two answers aloud without looking at the options, then try a mixed review."
    if fraction >= 0.75:
        return "Strong progress. Review the concepts behind the missed questions and repeat a shorter set later."
    if fraction >= 0.60:
        return "Developing understanding. Read the suggested glossary terms, redo worked calculations by hand, and try the topic again."
    return "The foundations need more attention. Return to the related weekly lesson, study the glossary and model explanation, then begin with five questions."


def _practice_report_markdown(state):
    lines = [
        "# MATH 490 Exam Practice Report", "",
        f"**Topic:** {state.get('topic', '')}", "",
        f"**Score:** {state.get('score', 0)}/{len(state.get('questions', []))}", "",
        f"**Attempt:** {state.get('attempt_number', 1)}", "",
    ]
    for number, response in enumerate(state.get("responses", []), 1):
        lines.extend([
            f"## Question {number}", "",
            response.get("question", ""), "",
            f"**Selected:** {response.get('selected', '')}", "",
            f"**Correct answer:** {response.get('answer', '')}", "",
            f"**Result:** {'Correct' if response.get('correct') else 'Incorrect'}", "",
            f"**Explanation:** {response.get('explanation', '')}", "",
        ])
    return "\n".join(lines).strip() + "\n"


def exam_practice_page():
    st.title("📝 Exam Practice")
    st.markdown(
        '<div class="simple-note"><strong>Practise privately before the exam.</strong><br>'
        'Choose a curriculum topic and answer 5 to 20 questions one at a time. '
        'The app explains every answer and recommends what to review next.</div>',
        unsafe_allow_html=True,
    )
    project = ensure_project_state()
    records = project.setdefault("exam_practice", {})
    active = st.session_state.get("exam_practice_active")

    with st.expander("Choose my practice set", expanded=not bool(active and not active.get("complete"))):
        topic_search = st.text_input(
            "Search the curriculum topics",
            key="exam_topic_search",
            placeholder="Examples: probability, linear regression, neural networks, forecasting",
        )
        words = [word for word in topic_search.lower().split() if word]
        topics = []
        for topic in COURSE_TOPIC_SOURCES:
            haystack = (topic + " " + " ".join(COURSE_TOPIC_KEYWORDS.get(topic, []))).lower()
            if all(word in haystack for word in words):
                topics.append(topic)
        if not topics:
            st.warning("No topic matches the search. Remove a word or search more broadly.")
            return
        _safe_select_state("exam_selected_topic", topics)
        topic = st.selectbox("Practice topic", topics, key="exam_selected_topic")
        difficulties = st.multiselect(
            "Question types",
            ["Foundation", "Application", "Calculation"],
            default=["Foundation", "Application", "Calculation"],
            key="exam_difficulties",
        )
        if not difficulties:
            st.info("Select at least one question type.")
            return
        available = exam_question_pool(topic, difficulties)
        max_count = min(20, len(available))
        count_options = [value for value in [5, 10, 15, 20] if value <= max_count]
        if max_count and max_count not in count_options and max_count < 5:
            count_options.append(max_count)
        if not count_options:
            st.warning("No valid questions are available for this combination.")
            return
        _safe_select_state("exam_question_count", count_options)
        count = st.selectbox("Number of questions", count_options, key="exam_question_count")
        st.caption(f"{len(available)} questions are available in the selected pool; each attempt uses up to 20 without repetition.")
        if st.button("Start new practice", type="primary", key="exam_start"):
            previous_attempts = int(records.get(topic, {}).get("attempts", 0))
            current_state = st.session_state.get("exam_practice_active")
            if isinstance(current_state, dict) and current_state.get("topic") == topic:
                previous_attempts = max(previous_attempts, int(current_state.get("attempt_number", 0)))
            state = _new_exam_attempt(topic, count, difficulties, previous_attempts + 1)
            st.session_state.exam_practice_active = state
            st.rerun()

    state = st.session_state.get("exam_practice_active")
    if not state:
        st.info("Choose a topic and start a practice set.")
        return
    questions = state.get("questions", [])
    total = len(questions)
    if state.get("complete"):
        score = int(state.get("score", 0))
        percentage = 100 * score / total if total else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Score", f"{score}/{total}")
        c2.metric("Percentage", f"{percentage:.0f}%")
        c3.metric("Attempt", state.get("attempt_number", 1))
        st.markdown(f'<div class="success-note"><strong>{_exam_score_advice(score, total)}</strong></div>', unsafe_allow_html=True)
        incorrect = [response for response in state.get("responses", []) if not response.get("correct")]
        if incorrect:
            counts = pd.Series([response.get("concept", "General course concept") for response in incorrect]).value_counts()
            st.subheader("What to review next")
            for concept, missed in counts.head(4).items():
                st.markdown(f"**{concept}** — {missed} missed question{'s' if missed != 1 else ''}")
            st.info(TOPIC_ADVICE.get(state.get("topic"), "Review the explanations for the missed questions and try again."))
            weakest = str(counts.index[0])
            st.button(
                "Open the Learning Library",
                on_click=_open_library_for_term,
                args=(weakest,),
                key="exam_open_library",
            )
        else:
            st.success("Every question was correct. Try the Full Course Mixed Review or increase the number of questions.")
        with st.expander("Review every answer", expanded=bool(incorrect)):
            for number, response in enumerate(state.get("responses", []), 1):
                if response.get("correct"):
                    st.success(f"{number}. Correct — {response.get('selected', '')}")
                else:
                    st.error(f"{number}. Your answer: {response.get('selected', '')}")
                    st.markdown(f"**Correct answer:** {response.get('answer', '')}")
                st.caption(f"{response.get('concept', '')} · {response.get('difficulty', '')}")
                st.write(response.get("explanation", ""))
        st.download_button(
            "Download practice report",
            _practice_report_markdown(state),
            f"MATH490_exam_practice_{re.sub(r'[^A-Za-z0-9]+', '_', state.get('topic', 'topic')).strip('_')}.md",
            "text/markdown",
            key="exam_report_download",
        )
        if st.button("Practise this topic again", key="exam_retry"):
            previous_attempts = int(records.get(state.get("topic"), {}).get("attempts", state.get("attempt_number", 1)))
            new_state = _new_exam_attempt(state.get("topic"), total, state.get("difficulties", []), previous_attempts + 1)
            st.session_state.exam_practice_active = new_state
            st.rerun()
        return

    index = int(state.get("index", 0))
    if index >= total:
        state["complete"] = True
        st.rerun()
    question = questions[index]
    st.progress((index + 1) / total if total else 0)
    st.caption(f"Question {index + 1} of {total} · {question.get('difficulty', '')} · {question.get('concept', '')}")
    st.markdown(f'<div class="question-card"><strong>{question.get("question", "")}</strong></div>', unsafe_allow_html=True)
    answer_key = f"exam_answer_{state.get('signature')}_{index}"
    selected = st.radio(
        "Choose one answer",
        question.get("options", []),
        index=None,
        disabled=bool(state.get("feedback_pending")),
        key=answer_key,
    )
    if not state.get("feedback_pending"):
        if st.button("Check my answer", type="primary", key=f"exam_check_{state.get('signature')}_{index}"):
            if selected is None:
                st.warning("Select an answer before checking it.")
            else:
                correct = selected == question.get("answer")
                response = {
                    "question": question.get("question", ""),
                    "selected": selected,
                    "answer": question.get("answer", ""),
                    "correct": bool(correct),
                    "explanation": question.get("explanation", ""),
                    "concept": question.get("concept", "General course concept"),
                    "difficulty": question.get("difficulty", "Foundation"),
                }
                state["responses"].append(response)
                if correct:
                    state["score"] += 1
                state["feedback_pending"] = True
                st.rerun()
    else:
        response = state["responses"][-1]
        if response.get("correct"):
            st.success("Correct.")
        else:
            st.error("Not correct this time.")
            st.markdown(f"**Correct answer:** {response.get('answer', '')}")
        st.write(response.get("explanation", ""))
        button_label = "Finish practice" if index == total - 1 else "Next question"
        if st.button(button_label, type="primary", key=f"exam_next_{state.get('signature')}_{index}"):
            state["feedback_pending"] = False
            state["index"] = index + 1
            if state["index"] >= total:
                state["complete"] = True
                topic = state.get("topic", "")
                previous = records.get(topic, {})
                current_score = int(state.get("score", 0))
                previous_best_score = int(previous.get("best_score", 0))
                previous_best_total = int(previous.get("best_total", 0))
                current_fraction = current_score / total if total else 0
                previous_fraction = previous_best_score / previous_best_total if previous_best_total else -1
                if current_fraction >= previous_fraction:
                    best_score, best_total = current_score, total
                else:
                    best_score, best_total = previous_best_score, previous_best_total
                records[topic] = {
                    "attempts": max(int(previous.get("attempts", 0)), int(state.get("attempt_number", 1))),
                    "latest_score": current_score,
                    "latest_total": total,
                    "best_score": best_score,
                    "best_total": best_total,
                }
            st.rerun()


def footer():
    st.markdown('<hr><div style="text-align:center;color:#666;font-size:.9rem">MATH 490 Applied AI Lab Studio · Created by Chibuike Ibebuchi, PhD · © 2026</div>',unsafe_allow_html=True)

def main():
    ensure_project_state()
    route = sidebar()
    space = route["space"]
    if space == "Today's Lab":
        todays_lab_page(route["week"])
    elif space == "Practical Studio":
        practical_studio_page(route["week"])
    elif space == "Wrap-Up":
        wrap_up_page(route["week"])
    elif space == "My Notebook":
        my_notebook_page(route["week"])
    elif space == "Instructor Setup":
        instructor_setup_page_v65(route["week"])
    elif space == "Learning Library":
        learning_library_page()
    elif space == "Exam Practice":
        exam_practice_page()
    else:
        full_studio_page(route["page"])
    footer()


if __name__ == "__main__":
    main()
