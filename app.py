from __future__ import annotations

import io
import os
import math
import warnings
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from PIL import Image

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
)
from sklearn.inspection import permutation_importance

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor,
    LogisticRegression,
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPRegressor, MLPClassifier

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

warnings.filterwarnings("ignore")
st.set_page_config(page_title="MATH 490 Applied AI Lab Studio", page_icon="📊", layout="wide")

# -----------------------------------------------------------------------------
# Teaching metadata
# -----------------------------------------------------------------------------
KAGGLE_LINKS = {
    "Students Performance in Exams": "https://www.kaggle.com/datasets/spscientist/students-performance-in-exams",
    "Student Performance Data Set": "https://www.kaggle.com/datasets/larsen0966/student-performance-data-set",
    "Credit Card Fraud Detection": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
    "Loan Approval Classification": "https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data",
    "S&P 500 Stocks": "https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks",
    "World Stock Prices": "https://www.kaggle.com/datasets/nelgiriyewithana/world-stock-prices-daily-updating",
    "Air Quality Time Series UCI": "https://www.kaggle.com/datasets/aayushkandpal/air-quality-time-series-data-uci",
    "Heart Disease": "https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data",
    "PlantVillage": "https://www.kaggle.com/datasets/mohitsingh1804/plantvillage",
}

PAGE_LIST = [
    "Home",
    "1. Data Explorer",
    "2. Visualization Lab",
    "3. Correlation & Partial Correlation",
    "4. Regression Lab",
    "5. Classification Lab",
    "6. Model Evaluation & Comparison",
    "7. Cross-Validation Lab",
    "8. Bootstrap Lab",
    "9. Forecasting Lab",
    "10. Computer Vision Lab",
    "11. Assignment Builder",
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def demo_students(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gender = rng.choice(["female", "male"], size=n)
    prep = rng.choice(["none", "completed"], p=[0.65, 0.35], size=n)
    lunch = rng.choice(["standard", "free/reduced"], p=[0.65, 0.35], size=n)
    parent = rng.choice(["high school", "some college", "bachelor", "master"], size=n)
    study_hours = rng.normal(5, 2, n).clip(0, 12)
    base = 55 + 4 * study_hours + (prep == "completed") * 7 + (lunch == "standard") * 4
    math_score = (base + rng.normal(0, 10, n)).clip(0, 100)
    reading_score = (base + (gender == "female") * 4 + rng.normal(0, 9, n)).clip(0, 100)
    writing_score = (base + (gender == "female") * 5 + rng.normal(0, 9, n)).clip(0, 100)
    df = pd.DataFrame({
        "gender": gender,
        "lunch": lunch,
        "parent_education": parent,
        "test_preparation": prep,
        "study_hours": np.round(study_hours, 2),
        "math_score": np.round(math_score, 1),
        "reading_score": np.round(reading_score, 1),
        "writing_score": np.round(writing_score, 1),
    })
    df["passed_math"] = (df["math_score"] >= 60).astype(int)
    return df

@st.cache_data(show_spinner=False)
def demo_health(n: int = 450, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(29, 78, n)
    cholesterol = rng.normal(220, 45, n).clip(120, 390)
    max_hr = rng.normal(165 - 0.7 * age, 15, n).clip(70, 205)
    resting_bp = rng.normal(125 + 0.25 * age, 15, n).clip(85, 200)
    chest_pain = rng.choice([0, 1, 2, 3], n)
    exercise_angina = rng.binomial(1, 0.25, n)
    logit = -6 + 0.04 * age + 0.01 * cholesterol + 0.025 * resting_bp - 0.025 * max_hr + 0.6 * exercise_angina + 0.25 * chest_pain
    p = 1 / (1 + np.exp(-logit))
    disease = rng.binomial(1, p)
    return pd.DataFrame({
        "age": age, "cholesterol": np.round(cholesterol, 1), "max_heart_rate": np.round(max_hr, 1),
        "resting_bp": np.round(resting_bp, 1), "chest_pain_type": chest_pain,
        "exercise_angina": exercise_angina, "heart_disease": disease,
    })

@st.cache_data(show_spinner=False)
def demo_finance(n: int = 900, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.lognormal(10.8, 0.45, n) / 1000
    loan_amount = rng.normal(130, 55, n).clip(15, 400)
    credit_score = rng.normal(680, 75, n).clip(350, 850)
    debt_to_income = rng.beta(2.2, 6, n).clip(0.02, 0.9)
    employment_years = rng.integers(0, 25, n)
    logit = -1.5 + 0.012 * (credit_score - 650) + 0.015 * income - 0.01 * loan_amount - 3.0 * debt_to_income + 0.035 * employment_years
    p = 1 / (1 + np.exp(-logit))
    approved = rng.binomial(1, p)
    return pd.DataFrame({
        "income_k": np.round(income, 1), "loan_amount_k": np.round(loan_amount, 1),
        "credit_score": np.round(credit_score, 0), "debt_to_income": np.round(debt_to_income, 3),
        "employment_years": employment_years, "loan_approved": approved,
    })

@st.cache_data(show_spinner=False)
def demo_air_quality(n: int = 900, seed: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    date = pd.date_range("2024-01-01", periods=n, freq="H")
    hour = date.hour
    temp = 18 + 8 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 2, n)
    humidity = 60 - 0.8 * temp + rng.normal(0, 8, n)
    traffic = 50 + 22 * ((hour >= 7) & (hour <= 9)) + 18 * ((hour >= 16) & (hour <= 19)) + rng.normal(0, 8, n)
    pm25 = 9 + 0.12 * traffic + 0.18 * humidity - 0.07 * temp + rng.normal(0, 3, n)
    return pd.DataFrame({
        "datetime": date, "hour": hour, "temperature": np.round(temp, 2),
        "humidity": np.round(humidity, 2), "traffic_index": np.round(traffic, 2),
        "pm25": np.round(pm25.clip(1, None), 2),
    })

@st.cache_data(show_spinner=False)
def demo_stock(n: int = 520, seed: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    date = pd.bdate_range("2024-01-01", periods=n)
    returns = rng.normal(0.0005, 0.015, n)
    close = 100 * np.exp(np.cumsum(returns))
    volume = rng.normal(2_000_000, 250_000, n).clip(500_000, None)
    return pd.DataFrame({"date": date, "close": np.round(close, 2), "volume": np.round(volume, 0)})

DEMO_DATASETS = {
    "Demo: Student Performance": demo_students,
    "Demo: Health / Heart Disease": demo_health,
    "Demo: Finance / Loan Approval": demo_finance,
    "Demo: Environment / Air Quality": demo_air_quality,
    "Demo: Finance / Stock Prices": demo_stock,
}


def lab_box(goal: str, do: List[str], interpret: List[str], assignment: List[str]) -> None:
    with st.expander("🎯 Lab guide: goal, hands-on task, interpretation, assignment", expanded=True):
        st.markdown(f"**Goal:** {goal}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Students should change:**")
            for x in do: st.markdown(f"- {x}")
        with c2:
            st.markdown("**Students should interpret:**")
            for x in interpret: st.markdown(f"- {x}")
        with c3:
            st.markdown("**Assignment prompt:**")
            for x in assignment: st.markdown(f"- {x}")


def get_current_df() -> pd.DataFrame:
    if "df" not in st.session_state:
        st.session_state.df = demo_students()
        st.session_state.dataset_name = "Demo: Student Performance"
    return st.session_state.df.copy()


def numeric_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(exclude=np.number).columns.tolist()


def classification_target_cols(df: pd.DataFrame, max_unique: int = 20) -> List[str]:
    """Columns that are sensible classification targets.

    Categorical columns are always allowed. Numeric columns are included only
    when they are low-cardinality, such as 0/1 labels or ordinal class codes.
    """
    out: List[str] = []
    for c in df.columns:
        nunique = df[c].dropna().nunique()
        if not pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
        elif 2 <= nunique <= max_unique:
            out.append(c)
    return out


def stable_selectbox(label: str, options: List[Any], key: str, default: Optional[Any] = None, **kwargs) -> Optional[Any]:
    """A selectbox that preserves a valid previous choice.

    The value changes only when the user changes it or when the old value is no
    longer valid for the current dataset/chart/problem.
    """
    options = list(options)
    if not options:
        st.warning(f"No valid options available for: {label}")
        return None
    if default is None or default not in options:
        default = options[0]
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = default
    return st.selectbox(label, options, key=key, **kwargs)


def stable_multiselect(
    label: str,
    options: List[Any],
    key: str,
    default: Optional[List[Any]] = None,
    **kwargs,
) -> List[Any]:
    """A multiselect that preserves previous valid choices."""
    options = list(options)
    if not options:
        st.warning(f"No valid options available for: {label}")
        return []
    if default is None:
        default = []
    default = [x for x in default if x in options]
    current = st.session_state.get(key, default)
    if current is None:
        current = []
    if not isinstance(current, list):
        current = [current]
    current = [x for x in current if x in options]
    if key not in st.session_state or st.session_state[key] != current:
        st.session_state[key] = current if current else default
    return st.multiselect(label, options, key=key, **kwargs)


def safe_numeric_pair(nums: List[str], x_key: str, y_key: str) -> Tuple[Optional[str], Optional[str]]:
    """Return persistent numeric X/Y choices without letting X equal Y."""
    if len(nums) < 2:
        st.warning("This section needs at least two numeric variables.")
        return None, None
    x = stable_selectbox("X variable", nums, key=x_key, default=nums[0])
    y_options = [c for c in nums if c != x]
    y_default = y_options[0] if y_options else None
    y = stable_selectbox("Y variable", y_options, key=y_key, default=y_default)
    return x, y


def clamp_cv_folds(n_rows: int, requested_k: int, min_class_count: Optional[int] = None) -> int:
    """Prevent cross-validation from asking for more folds than data/classes allow."""
    max_k = max(2, int(n_rows))
    if min_class_count is not None:
        max_k = min(max_k, int(min_class_count))
    return max(2, min(int(requested_k), max_k))


def make_preprocessor(df: pd.DataFrame, features: List[str], scale_numeric: bool = True) -> ColumnTransformer:
    nums = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cats = [c for c in features if c not in nums]
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    transformers = []
    if nums:
        transformers.append(("num", Pipeline(num_steps), nums))
    if cats:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cats))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def split_xy(df: pd.DataFrame, target: str, features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    clean = df[features + [target]].dropna(subset=[target])
    return clean[features], clean[target]


def regression_model(name: str, params: Dict[str, Any]):
    if name == "Linear Regression": return LinearRegression()
    if name == "Ridge": return Ridge(alpha=params.get("alpha", 1.0))
    if name == "Lasso": return Lasso(alpha=params.get("alpha", 0.1), max_iter=10000)
    if name == "Elastic Net": return ElasticNet(alpha=params.get("alpha", 0.1), l1_ratio=params.get("l1_ratio", 0.5), max_iter=10000)
    if name == "Huber": return HuberRegressor(epsilon=params.get("epsilon", 1.35), alpha=params.get("alpha", 0.0001), max_iter=500)
    if name == "Decision Tree": return DecisionTreeRegressor(max_depth=params.get("max_depth", 4), random_state=42)
    if name == "Random Forest": return RandomForestRegressor(n_estimators=params.get("n_estimators", 100), max_depth=params.get("max_depth", None), random_state=42, n_jobs=-1)
    if name == "Gradient Boosting": return GradientBoostingRegressor(n_estimators=params.get("n_estimators", 100), learning_rate=params.get("learning_rate", 0.05), random_state=42)
    if name == "Support Vector Regressor": return SVR(C=params.get("C", 1.0), epsilon=params.get("svr_epsilon", 0.1), kernel=params.get("kernel", "rbf"))
    if name == "Neural Network / FFN": return MLPRegressor(hidden_layer_sizes=(params.get("hidden_units", 32),), max_iter=params.get("max_iter", 500), random_state=42)
    if name == "XGBoost" and XGBOOST_AVAILABLE: return XGBRegressor(n_estimators=params.get("n_estimators", 100), learning_rate=params.get("learning_rate", 0.05), random_state=42, objective="reg:squarederror")
    return LinearRegression()


def classification_model(name: str, params: Dict[str, Any]):
    if name == "Logistic Regression": return LogisticRegression(C=params.get("C", 1.0), max_iter=1000)
    if name == "LDA": return LinearDiscriminantAnalysis()
    if name == "QDA": return QuadraticDiscriminantAnalysis()
    if name == "KNN": return KNeighborsClassifier(n_neighbors=params.get("k", 5))
    if name == "Decision Tree": return DecisionTreeClassifier(max_depth=params.get("max_depth", 4), random_state=42)
    if name == "Random Forest": return RandomForestClassifier(n_estimators=params.get("n_estimators", 100), max_depth=params.get("max_depth", None), random_state=42, n_jobs=-1)
    if name == "Gradient Boosting": return GradientBoostingClassifier(n_estimators=params.get("n_estimators", 100), learning_rate=params.get("learning_rate", 0.05), random_state=42)
    if name == "Support Vector Machine": return SVC(C=params.get("C", 1.0), kernel=params.get("kernel", "rbf"), probability=True, random_state=42)
    if name == "Neural Network / FFN": return MLPClassifier(hidden_layer_sizes=(params.get("hidden_units", 32),), max_iter=params.get("max_iter", 500), random_state=42)
    if name == "XGBoost" and XGBOOST_AVAILABLE: return XGBClassifier(n_estimators=params.get("n_estimators", 100), learning_rate=params.get("learning_rate", 0.05), random_state=42, eval_metric="logloss")
    return LogisticRegression(max_iter=1000)


def model_params_ui(prefix: str = "") -> Dict[str, Any]:
    with st.expander("⚙️ Model parameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            alpha = st.number_input("Regularization alpha", min_value=0.0, value=0.1, step=0.1, key=prefix+"alpha")
            C = st.number_input("SVM/Logistic C", min_value=0.01, value=1.0, step=0.1, key=prefix+"C")
        with c2:
            l1_ratio = st.slider("Elastic Net l1_ratio", 0.0, 1.0, 0.5, key=prefix+"l1")
            max_depth_val = st.slider("Tree max depth", 1, 20, 5, key=prefix+"depth")
        with c3:
            n_estimators = st.slider("Number of trees", 10, 500, 100, step=10, key=prefix+"trees")
            learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.05, step=0.01, key=prefix+"lr")
        kernel = st.selectbox("SVM kernel", ["rbf", "linear", "poly"], key=prefix+"kernel")
        hidden_units = st.slider("FFN hidden units", 4, 128, 32, key=prefix+"hidden")
        return {
            "alpha": alpha, "C": C, "l1_ratio": l1_ratio, "max_depth": max_depth_val,
            "n_estimators": n_estimators, "learning_rate": learning_rate, "kernel": kernel,
            "hidden_units": hidden_units, "max_iter": 500, "epsilon": 1.35, "svr_epsilon": 0.1,
            "k": 5,
        }


def feature_selector(df: pd.DataFrame, target_label: str = "Target") -> Tuple[str, List[str]]:
    cols = df.columns.tolist()
    target = st.selectbox(target_label, cols, index=len(cols)-1 if cols else 0)
    defaults = [c for c in cols if c != target][: min(6, len(cols)-1)]
    features = st.multiselect("Predictor variables", [c for c in cols if c != target], default=defaults)
    return target, features


def try_get_feature_names(pipe: Pipeline) -> List[str]:
    try:
        return pipe.named_steps["prep"].get_feature_names_out().tolist()
    except Exception:
        return []


def display_regression_results(y_test, pred):
    rmse = math.sqrt(mean_squared_error(y_test, pred))
    metrics = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R²"],
        "Value": [mean_absolute_error(y_test, pred), rmse, r2_score(y_test, pred)]
    })
    st.dataframe(metrics, use_container_width=True)
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred, alpha=0.65)
    mn = min(np.min(y_test), np.min(pred)); mx = max(np.max(y_test), np.max(pred))
    ax.plot([mn, mx], [mn, mx], linestyle="--")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title("Observed vs Predicted")
    st.pyplot(fig)


def display_classification_results(y_test, pred, proba=None):
    labels = sorted(pd.Series(y_test).dropna().unique().tolist())
    avg = "binary" if len(labels) == 2 else "weighted"
    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1"],
        "Value": [
            accuracy_score(y_test, pred),
            precision_score(y_test, pred, average=avg, zero_division=0),
            recall_score(y_test, pred, average=avg, zero_division=0),
            f1_score(y_test, pred, average=avg, zero_division=0),
        ]
    })
    st.dataframe(metrics, use_container_width=True)
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    if proba is not None and len(labels) == 2:
        try:
            auc = roc_auc_score(y_test, proba[:, 1])
            fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
            st.write(f"**ROC-AUC:** {auc:.3f}")
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr)
            ax2.plot([0, 1], [0, 1], linestyle="--")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("ROC Curve")
            st.pyplot(fig2)
        except Exception:
            pass


def partial_corr(df: pd.DataFrame, x: str, y: str, controls: List[str]) -> Optional[float]:
    """Compute partial Pearson correlation after removing selected numeric controls.

    A partial correlation is only meaningful here when at least one control
    variable is selected. Without controls, the value collapses to the ordinary
    Pearson correlation, so the app deliberately returns None instead of showing
    a misleading duplicate number.
    """
    if not controls:
        return None
    sub = df[[x, y] + controls].dropna()
    if len(sub) < 5:
        return None
    Xc = sub[controls].to_numpy()
    Xc = np.column_stack([np.ones(len(Xc)), Xc])
    bx = np.linalg.lstsq(Xc, sub[x].to_numpy(), rcond=None)[0]
    by = np.linalg.lstsq(Xc, sub[y].to_numpy(), rcond=None)[0]
    rx = sub[x].to_numpy() - Xc @ bx
    ry = sub[y].to_numpy() - Xc @ by
    return float(np.corrcoef(rx, ry)[0, 1])


def render_dataset_sidebar():
    st.sidebar.header("Dataset")
    source = st.sidebar.radio("Choose data source", ["Demo dataset", "Upload CSV/Excel"], horizontal=False)
    if source == "Demo dataset":
        name = st.sidebar.selectbox("Demo dataset", list(DEMO_DATASETS.keys()))
        if st.sidebar.button("Load selected demo dataset") or "df" not in st.session_state:
            st.session_state.df = DEMO_DATASETS[name]()
            st.session_state.dataset_name = name
    else:
        file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        if file is not None:
            try:
                if file.name.lower().endswith(".csv"):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                st.session_state.df = df
                st.session_state.dataset_name = file.name
                st.sidebar.success("Dataset loaded")
            except Exception as e:
                st.sidebar.error(f"Could not read file: {e}")
    st.sidebar.caption("Kaggle files must be downloaded manually, then uploaded here.")

# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
def page_home():
    st.title("📊 MATH 490 Applied AI Lab Studio")
    st.subheader("Behavior • Finance • Environment • Health • Society • Computer Vision")
    st.markdown("""
This app is designed for undergraduate applied AI labs. Students can upload a Kaggle dataset, select variables, train models, test assumptions, and interpret results.

**Main learning flow:** data → visualization → correlation → regression/classification → assumptions → evaluation → cross-validation → assignment.
""")
    st.info("Teaching rule: start with simple linear models first, then compare against more complex models. Students should explain whether complexity improved interpretation or only improved prediction.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Recommended Kaggle datasets")
        for name, link in KAGGLE_LINKS.items():
            st.markdown(f"- [{name}]({link})")
    with c2:
        st.markdown("### Teaching Plan")
        st.markdown("""
1. **10 min**: introduce question and dataset  
2. **15 min**: explore variables and plots  
3. **20 min**: fit model and change parameters  
4. **10 min**: interpret coefficients/odds ratios/errors  
5. **5 min**: submit short assignment response
""")
    st.markdown("### Modules")
    st.write("Data Explorer, Visualization, Correlation, Regression, Classification, Model Evaluation, Cross-Validation, Bootstrap, Forecasting, Computer Vision, Assignment Builder.")


def page_data_explorer():
    df = get_current_df()
    st.title("1. Data Explorer")
    lab_box(
        "Understand rows, columns, variable types, missing values, and simple summaries.",
        ["Dataset", "Number of rows to preview", "Columns to inspect"],
        ["What each row represents", "Which variables are numerical/categorical", "Whether missing data may affect modeling"],
        ["Describe the dataset in 5 sentences.", "Identify one possible target variable and three predictors."],
    )
    st.write(f"**Current dataset:** {st.session_state.get('dataset_name', 'Unknown')}")
    st.dataframe(df.head(st.slider("Rows to preview", 5, 50, 10)), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0]); c2.metric("Columns", df.shape[1]); c3.metric("Missing cells", int(df.isna().sum().sum()))
    st.subheader("Column types")
    st.dataframe(pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns], "missing": df.isna().sum().values}), use_container_width=True)
    st.subheader("Numerical summary")
    st.dataframe(df.describe().T, use_container_width=True)



def page_visualization():
    df = get_current_df()
    st.title("2. Visualization Lab")
    lab_box(
        "Use plots to find distributions, outliers, group differences, and relationships.",
        ["Chart type", "Only the variables required for that chart", "Optional categorical group variable when supported"],
        ["Skewness and outliers", "Whether relationships look linear", "Whether groups differ"],
        ["Create two plots and explain what each plot teaches us."],
    )

    nums = numeric_cols(df)
    cats = categorical_cols(df)

    chart = stable_selectbox(
        "Chart type",
        ["Histogram", "Boxplot", "Scatter plot", "Bar chart", "Correlation heatmap"],
        key="viz_chart_type",
        default="Histogram",
    )

    st.caption(
        "The controls below change by chart type. Numeric dropdowns show only numeric variables; "
        "group/color dropdowns show only categorical variables. Choices remain stable while you stay in this page."
    )

    if chart == "Histogram":
        if not nums:
            st.warning("Histogram requires at least one numeric variable.")
            return
        x = stable_selectbox("Numeric variable", nums, key="viz_hist_numeric", default=nums[0])
        bins = st.slider("Number of bins", 5, 80, 25, key="viz_hist_bins")
        fig, ax = plt.subplots()
        sns.histplot(df[x].dropna(), kde=True, bins=bins, ax=ax)
        ax.set_title(f"Distribution of {x}")
        ax.set_xlabel(x)
        ax.set_ylabel("Count")
        st.pyplot(fig)
        st.info("Use a histogram when you want to understand the distribution of one numeric variable.")

    elif chart == "Boxplot":
        if not nums:
            st.warning("Boxplot requires a numeric outcome variable.")
            return
        y = stable_selectbox("Numeric outcome", nums, key="viz_box_numeric", default=nums[0])
        group_options = ["None"] + cats
        group = stable_selectbox("Optional categorical group", group_options, key="viz_box_group", default="None")
        fig, ax = plt.subplots()
        if group == "None":
            sns.boxplot(y=df[y], ax=ax)
            ax.set_title(f"Boxplot of {y}")
            ax.set_ylabel(y)
        else:
            sns.boxplot(data=df, x=group, y=y, ax=ax)
            ax.set_title(f"{y} by {group}")
            ax.set_xlabel(group)
            ax.set_ylabel(y)
            plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)
        st.info("Use a boxplot for one numeric variable, optionally grouped by a categorical variable.")

    elif chart == "Scatter plot":
        if len(nums) < 2:
            st.warning("Scatter plot requires at least two numeric variables.")
            return
        x, y = safe_numeric_pair(nums, "viz_scatter_x", "viz_scatter_y")
        if x is None or y is None:
            return
        group_options = ["None"] + cats
        hue = stable_selectbox("Optional categorical color/group", group_options, key="viz_scatter_group", default="None")
        show_trend = st.checkbox("Add regression line", value=False, key="viz_scatter_trend")
        fig, ax = plt.subplots()
        if show_trend and hue == "None":
            sns.regplot(data=df, x=x, y=y, ax=ax, scatter_kws={"alpha": 0.6})
        else:
            sns.scatterplot(data=df, x=x, y=y, hue=None if hue == "None" else hue, ax=ax, alpha=0.75)
        ax.set_title(f"{y} versus {x}")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        st.pyplot(fig)
        st.info("Use a scatter plot only for two numeric variables. The optional color/group variable is categorical.")

    elif chart == "Bar chart":
        if not cats:
            st.warning("Bar chart requires at least one categorical variable.")
            return
        x = stable_selectbox("Categorical variable", cats, key="viz_bar_category", default=cats[0])
        mode = stable_selectbox(
            "Bar chart value",
            ["Count rows", "Mean of numeric variable", "Median of numeric variable", "Sum of numeric variable"],
            key="viz_bar_mode",
            default="Count rows",
        )
        if mode == "Count rows":
            plot_df = df[x].astype(str).value_counts(dropna=False).reset_index()
            plot_df.columns = [x, "count"]
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=plot_df, x=x, y="count", ax=ax)
            ax.set_title(f"Count by {x}")
            ax.set_xlabel(x)
            ax.set_ylabel("Count")
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig)
        else:
            if not nums:
                st.warning("This bar-chart mode requires at least one numeric variable.")
                return
            y = stable_selectbox("Numeric variable to aggregate", nums, key="viz_bar_numeric", default=nums[0])
            agg_map = {
                "Mean of numeric variable": "mean",
                "Median of numeric variable": "median",
                "Sum of numeric variable": "sum",
            }
            agg_name = agg_map[mode]
            plot_df = df.groupby(x, dropna=False)[y].agg(agg_name).reset_index()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=plot_df, x=x, y=y, ax=ax)
            ax.set_title(f"{mode} of {y} by {x}")
            ax.set_xlabel(x)
            ax.set_ylabel(f"{agg_name}({y})")
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig)
        st.info("Use a bar chart for a categorical X variable. It can show row counts or an aggregate of one numeric variable.")

    elif chart == "Correlation heatmap":
        if len(nums) < 2:
            st.warning("Correlation heatmap requires at least two numeric variables.")
            return
        selected = stable_multiselect(
            "Numeric variables to include",
            nums,
            key="viz_heatmap_nums",
            default=nums[: min(6, len(nums))],
        )
        if len(selected) < 2:
            st.warning("Select at least two numeric variables.")
            return
        method = stable_selectbox("Correlation method", ["Pearson", "Spearman", "Kendall"], key="viz_heatmap_method", default="Pearson")
        corr = df[selected].corr(method=method.lower())
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title(f"{method} correlation heatmap")
        st.pyplot(fig)
        st.info("Use a correlation heatmap only for numeric variables.")



def page_correlation():
    df = get_current_df()
    st.title("3. Correlation & Partial Correlation")
    lab_box(
        "Measure simple correlation and partial correlation while controlling for other numeric variables.",
        ["Numeric X variable", "Numeric Y variable", "Numeric control variables"],
        ["Direction and strength", "Difference between correlation and partial correlation", "Correlation does not prove causation"],
        ["Report one simple correlation and one partial correlation. Explain why they differ."],
    )

    nums = numeric_cols(df)
    if len(nums) < 2:
        st.warning("Need at least two numeric variables for correlation.")
        return

    x, y = safe_numeric_pair(nums, "corr_x", "corr_y")
    if x is None or y is None:
        return

    control_options = [c for c in nums if c not in [x, y]]
    controls = stable_multiselect(
        "Numeric control variables for partial correlation",
        control_options,
        key="corr_controls",
        default=[],
    ) if control_options else []

    method = stable_selectbox("Simple correlation method", ["Pearson", "Spearman", "Kendall"], key="corr_method", default="Pearson")

    sub = df[[x, y]].dropna()
    if len(sub) < 3:
        st.warning("Not enough complete rows for the selected X and Y variables.")
        return

    if method == "Pearson":
        simple, p_value = stats.pearsonr(sub[x], sub[y])
    elif method == "Spearman":
        simple, p_value = stats.spearmanr(sub[x], sub[y])
    else:
        simple, p_value = stats.kendalltau(sub[x], sub[y])

    pc = partial_corr(df, x, y, controls)
    n_partial = len(df[[x, y] + controls].dropna()) if controls else 0

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{method} correlation", f"{simple:.3f}")
    c2.metric("p-value", f"{p_value:.4f}")
    c3.metric("Complete rows", f"{len(sub):,}")

    c4, c5 = st.columns(2)
    if controls:
        c4.metric("Partial Pearson correlation", "NA" if pc is None else f"{pc:.3f}")
        c5.metric("Rows for partial correlation", f"{n_partial:,}")
    else:
        c4.metric("Partial Pearson correlation", "Select controls")
        c5.metric("Rows for partial correlation", "NA")
        st.info("Partial correlation is calculated only after at least one numeric control variable is selected. Without controls, it collapses to the ordinary Pearson correlation, so the app does not show it as a separate result.")

    st.markdown(
        "**Interpretation guide:** simple correlation describes the relationship between two numeric variables. "
        "Partial Pearson correlation estimates the relationship between X and Y after removing the linear effect of the selected numeric controls."
    )

    fig, ax = plt.subplots()
    sns.regplot(data=df, x=x, y=y, ax=ax, scatter_kws={"alpha": 0.6})
    ax.set_title(f"{y} versus {x}")
    st.pyplot(fig)



def page_regression():
    df = get_current_df()
    st.title("4. Regression Lab")
    lab_box(
        "Predict a continuous outcome and interpret coefficients, errors, residuals, and assumptions.",
        ["Numeric target", "Predictors", "Model", "Train/test split", "Regularization strength"],
        ["Coefficient sign and size", "MAE/RMSE/R²", "Residual pattern", "Whether assumptions are reasonable"],
        ["Fit one linear model and one complex model. Which is more interpretable? Which predicts better?"],
    )

    nums = numeric_cols(df)
    if not nums:
        st.warning("Need at least one numeric outcome for regression.")
        return

    target = stable_selectbox("Continuous numeric target", nums, key="reg_target", default=nums[-1])
    feature_options = [c for c in df.columns if c != target]
    default_features = feature_options[: min(5, len(feature_options))]
    features = stable_multiselect("Predictor variables", feature_options, key="reg_features", default=default_features)
    if not features:
        st.warning("Select at least one predictor.")
        return

    model_names = ["Linear Regression", "Ridge", "Lasso", "Elastic Net", "Huber", "Decision Tree", "Random Forest", "Gradient Boosting", "Support Vector Regressor", "Neural Network / FFN"]
    if XGBOOST_AVAILABLE:
        model_names.append("XGBoost")
    model_name = stable_selectbox("Regression model", model_names, key="reg_model", default="Linear Regression")
    params = model_params_ui("reg_")
    test_size = st.slider("Test size", 0.1, 0.5, 0.25, key="reg_test_size")

    X, y = split_xy(df, target, features)
    if len(X) < 10:
        st.warning("The selected target/predictors leave fewer than 10 complete target rows. Choose different variables or clean the data.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    pipe = Pipeline([("prep", make_preprocessor(X, features, scale_numeric=True)), ("model", regression_model(model_name, params))])

    if st.button("Train regression model", type="primary", key="reg_train_button"):
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        display_regression_results(y_test, pred)

        st.subheader("Coefficient interpretation / feature importance")
        model = pipe.named_steps["model"]
        feature_names = try_get_feature_names(pipe)
        if hasattr(model, "coef_") and feature_names:
            coefs = np.ravel(model.coef_)
            st.dataframe(
                pd.DataFrame({"feature": feature_names[:len(coefs)], "coefficient": coefs})
                .sort_values("coefficient", key=np.abs, ascending=False),
                use_container_width=True,
            )
            st.info("For linear models: holding other predictors constant, a positive coefficient means the predicted outcome increases as that feature increases.")
        else:
            try:
                imp = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=42)
                st.dataframe(pd.DataFrame({"feature": features, "importance": imp.importances_mean}).sort_values("importance", ascending=False), use_container_width=True)
            except Exception:
                st.write("Feature importance not available for this model.")

        st.subheader("Assumption checks for linear-style regression")
        residuals = y_test - pred
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            ax.scatter(pred, residuals, alpha=0.6)
            ax.axhline(0, linestyle="--")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs Predicted")
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots()
            stats.probplot(residuals, dist="norm", plot=ax)
            ax.set_title("Q-Q Plot of Residuals")
            st.pyplot(fig)

        if STATSMODELS_AVAILABLE and all(pd.api.types.is_numeric_dtype(X[c]) for c in features):
            Xnum = X[features].dropna()
            if Xnum.shape[1] >= 2 and Xnum.shape[0] > Xnum.shape[1] + 1:
                Xv = sm.add_constant(Xnum)
                vif = pd.DataFrame({"variable": Xv.columns, "VIF": [variance_inflation_factor(Xv.values, i) for i in range(Xv.shape[1])]})
                st.write("**Multicollinearity check: VIF**")
                st.dataframe(vif, use_container_width=True)



def page_classification():
    df = get_current_df()
    st.title("5. Classification Lab")
    lab_box(
        "Predict a class label and interpret probabilities, odds ratios, decision boundaries, and confusion matrices.",
        ["Classification target", "Predictors", "Classifier", "Threshold/parameters", "Train/test split"],
        ["Accuracy vs precision/recall", "False positives/false negatives", "Odds ratios for logistic regression", "Class imbalance"],
        ["Compare logistic regression with one complex classifier. Explain which mistakes matter most."],
    )

    target_options = classification_target_cols(df)
    if not target_options:
        st.warning("No suitable classification target found. Use a categorical column or a low-cardinality numeric label such as 0/1.")
        return

    target = stable_selectbox("Classification target", target_options, key="clf_target", default=target_options[-1])
    feature_options = [c for c in df.columns if c != target]
    features = stable_multiselect("Predictor variables", feature_options, key="clf_features", default=feature_options[: min(5, len(feature_options))])
    if not features:
        st.warning("Select at least one predictor.")
        return

    model_names = ["Logistic Regression", "LDA", "QDA", "KNN", "Decision Tree", "Random Forest", "Gradient Boosting", "Support Vector Machine", "Neural Network / FFN"]
    if XGBOOST_AVAILABLE:
        model_names.append("XGBoost")
    model_name = stable_selectbox("Classifier", model_names, key="clf_model", default="Logistic Regression")
    params = model_params_ui("clf_")
    test_size = st.slider("Test size", 0.1, 0.5, 0.25, key="clf_test")

    X, y_raw = split_xy(df, target, features)
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)

    if len(np.unique(y)) < 2:
        st.warning("Target needs at least two classes.")
        return

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
    pipe = Pipeline([("prep", make_preprocessor(X, features, scale_numeric=True)), ("model", classification_model(model_name, params))])

    if st.button("Train classification model", type="primary", key="clf_train_button"):
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
        display_classification_results(y_test, pred, proba)
        st.write("**Class labels:**", dict(enumerate(le.classes_)))

        if model_name == "Logistic Regression":
            model = pipe.named_steps["model"]
            feature_names = try_get_feature_names(pipe)
            if hasattr(model, "coef_") and len(le.classes_) == 2:
                coefs = np.ravel(model.coef_)
                odds = np.exp(coefs)
                st.subheader("Logistic regression odds ratios")
                st.dataframe(
                    pd.DataFrame({"feature": feature_names[:len(coefs)], "coefficient_log_odds": coefs, "odds_ratio": odds})
                    .sort_values("odds_ratio", ascending=False),
                    use_container_width=True,
                )
                st.info("Odds ratio > 1 means the odds of the positive class increase as that feature increases. Odds ratio < 1 means the odds decrease.")



def page_model_eval():
    df = get_current_df()
    st.title("6. Model Evaluation & Comparison")
    lab_box(
        "Compare multiple models fairly using the same target, predictors, split, and metric.",
        ["Problem type", "Models to compare", "Evaluation metric", "Train/test split", "Optional cross-validation"],
        ["Simple vs complex model tradeoff", "Overfitting risk", "Best model by metric vs best model for explanation"],
        ["Create a model comparison table and recommend one model for prediction and one for interpretation."],
    )

    st.markdown("""
### Controlled model comparison
Use this page like a small experiment. Keep the same target, predictors, split, and metric, then compare several models fairly.  
The best predictive model is not always the best model for explanation.
""")

    problem = st.radio("Problem type", ["Regression", "Classification"], horizontal=True, key="eval_problem")

    if problem == "Regression":
        target_options = numeric_cols(df)
        if not target_options:
            st.warning("Regression needs a numeric target.")
            return
        target = stable_selectbox("Numeric target", target_options, key="eval_reg_target", default=target_options[-1])
    else:
        target_options = classification_target_cols(df)
        if not target_options:
            st.warning("Classification needs a categorical or low-cardinality numeric target.")
            return
        target = stable_selectbox("Classification target", target_options, key="eval_clf_target", default=target_options[-1])

    feature_options = [c for c in df.columns if c != target]
    features = stable_multiselect(
        "Predictor variables",
        feature_options,
        key=f"eval_{problem.lower()}_features",
        default=feature_options[: min(5, len(feature_options))],
    )
    if not features:
        st.warning("Select at least one predictor.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        train_percent = st.slider("Training data size (%)", 50, 90, 75, step=5, key="eval_train_pct")
    with c2:
        random_seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1, key="eval_seed")
    with c3:
        use_cv = st.checkbox("Also run cross-validation", value=False, key="eval_use_cv")

    test_size = 1 - (train_percent / 100)
    cv_folds = st.slider("Cross-validation folds", 3, 10, 5, key="eval_cv_folds") if use_cv else None

    params = model_params_ui("eval_")
    X, y_raw = split_xy(df, target, features)
    if len(X) < 10:
        st.warning("The selected target/predictors leave too few usable rows. Choose different variables or clean missing values.")
        return

    results = []
    predictions = {}

    if problem == "Regression":
        available_models = [
            "Linear Regression", "Ridge", "Lasso", "Elastic Net", "Huber",
            "Decision Tree", "Random Forest", "Gradient Boosting",
            "Support Vector Regressor", "Neural Network / FFN",
        ]
        if XGBOOST_AVAILABLE:
            available_models.append("XGBoost")

        selected_models = stable_multiselect(
            "Models to compare",
            available_models,
            key="eval_reg_models",
            default=["Linear Regression", "Ridge", "Random Forest", "Gradient Boosting"],
        )
        metric_name = stable_selectbox(
            "Evaluation metric",
            ["Root Mean Squared Error", "Mean Absolute Error", "R-squared"],
            key="eval_reg_metric",
            default="Root Mean Squared Error",
        )

        def reg_score(y_true, y_pred):
            if metric_name == "Root Mean Squared Error":
                return math.sqrt(mean_squared_error(y_true, y_pred))
            if metric_name == "Mean Absolute Error":
                return mean_absolute_error(y_true, y_pred)
            return r2_score(y_true, y_pred)

        lower_is_better = metric_name in ["Root Mean Squared Error", "Mean Absolute Error"]
        scoring_lookup = {
            "Root Mean Squared Error": "neg_root_mean_squared_error",
            "Mean Absolute Error": "neg_mean_absolute_error",
            "R-squared": "r2",
        }

        if st.button("Run model comparison", type="primary", key="eval_reg_run"):
            if not selected_models:
                st.warning("Select at least one model.")
                return
            X_train, X_test, y_train, y_test = train_test_split(X, y_raw, test_size=test_size, random_state=int(random_seed))
            for model_name in selected_models:
                try:
                    pipe = Pipeline([
                        ("prep", make_preprocessor(X, features, scale_numeric=True)),
                        ("model", regression_model(model_name, params)),
                    ])
                    pipe.fit(X_train, y_train)
                    train_pred = pipe.predict(X_train)
                    test_pred = pipe.predict(X_test)
                    train_score = reg_score(y_train, train_pred)
                    test_score = reg_score(y_test, test_pred)
                    gap = test_score - train_score if lower_is_better else train_score - test_score
                    row = {
                        "Model": model_name,
                        f"Train {metric_name}": train_score,
                        f"Test {metric_name}": test_score,
                        "Overfitting gap": gap,
                    }
                    if use_cv and cv_folds is not None:
                        k_safe = min(int(cv_folds), len(X))
                        if k_safe >= 2:
                            cv = KFold(n_splits=k_safe, shuffle=True, random_state=int(random_seed))
                            scores = cross_val_score(pipe, X, y_raw, cv=cv, scoring=scoring_lookup[metric_name])
                            scores = -scores if scoring_lookup[metric_name].startswith("neg_") else scores
                            row[f"Cross-validation {metric_name}"] = float(np.mean(scores))
                            row["CV folds used"] = k_safe
                    results.append(row)
                    predictions[model_name] = (y_test, test_pred)
                except Exception as e:
                    results.append({"Model": model_name, "Error": str(e)})

            res_df = pd.DataFrame(results)
            score_col = f"Test {metric_name}"
            if score_col in res_df.columns:
                res_df = res_df.sort_values(score_col, ascending=lower_is_better)
            st.subheader("Model comparison table")
            st.dataframe(res_df, use_container_width=True)

            if score_col in res_df.columns and res_df[score_col].notna().any():
                best_model = res_df.dropna(subset=[score_col]).iloc[0]["Model"]
                st.success(f"Best predictive model by {metric_name}: {best_model}")
                st.info("Interpretation tip: linear, Ridge, Lasso, Elastic Net, and Huber models are usually easier to explain. Tree ensembles and neural networks may predict well but are harder to interpret.")

                plot_df = res_df[["Model", score_col]].dropna().set_index("Model")
                st.subheader(f"Test performance by model: {metric_name}")
                st.bar_chart(plot_df)

                if best_model in predictions:
                    y_test, test_pred = predictions[best_model]
                    st.subheader("Best model: observed vs predicted")
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, test_pred, alpha=0.65)
                    mn = min(np.min(y_test), np.min(test_pred))
                    mx = max(np.max(y_test), np.max(test_pred))
                    ax.plot([mn, mx], [mn, mx], linestyle="--")
                    ax.set_xlabel("Observed")
                    ax.set_ylabel("Predicted")
                    ax.set_title(f"Observed vs Predicted: {best_model}")
                    st.pyplot(fig)

                st.subheader("Overfitting check")
                gap_df = res_df[["Model", "Overfitting gap"]].dropna().set_index("Model")
                st.bar_chart(gap_df)
                st.caption("For error metrics, a larger positive gap means the test error is much worse than the training error. For R-squared, a larger positive gap means training performance is much higher than test performance.")

    else:
        available_models = [
            "Logistic Regression", "LDA", "QDA", "KNN", "Decision Tree",
            "Random Forest", "Gradient Boosting", "Support Vector Machine", "Neural Network / FFN",
        ]
        if XGBOOST_AVAILABLE:
            available_models.append("XGBoost")

        selected_models = stable_multiselect(
            "Models to compare",
            available_models,
            key="eval_clf_models",
            default=["Logistic Regression", "LDA", "Random Forest", "Support Vector Machine"],
        )
        metric_name = stable_selectbox(
            "Evaluation metric",
            ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
            key="eval_clf_metric",
            default="Accuracy",
        )

        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)
        if len(np.unique(y)) < 2:
            st.warning("The target needs at least two classes for classification.")
            return

        def class_score(y_true, y_pred, y_proba=None):
            labels = sorted(pd.Series(y_true).dropna().unique().tolist())
            avg = "binary" if len(labels) == 2 else "weighted"
            if metric_name == "Accuracy":
                return accuracy_score(y_true, y_pred)
            if metric_name == "Precision":
                return precision_score(y_true, y_pred, average=avg, zero_division=0)
            if metric_name == "Recall":
                return recall_score(y_true, y_pred, average=avg, zero_division=0)
            if metric_name == "F1-score":
                return f1_score(y_true, y_pred, average=avg, zero_division=0)
            if metric_name == "ROC-AUC" and y_proba is not None and len(labels) == 2:
                return roc_auc_score(y_true, y_proba[:, 1])
            return np.nan

        scoring_lookup = {
            "Accuracy": "accuracy",
            "Precision": "precision_weighted",
            "Recall": "recall_weighted",
            "F1-score": "f1_weighted",
            "ROC-AUC": "roc_auc",
        }

        if st.button("Run model comparison", type="primary", key="eval_clf_run"):
            if not selected_models:
                st.warning("Select at least one model.")
                return
            if metric_name == "ROC-AUC" and len(np.unique(y)) != 2:
                st.warning("ROC-AUC is only available here for binary classification. Choose Accuracy, Precision, Recall, or F1-score for multiclass targets.")
                return
            stratify = y if y.value_counts().min() >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_seed), stratify=stratify
            )
            for model_name in selected_models:
                try:
                    pipe = Pipeline([
                        ("prep", make_preprocessor(X, features, scale_numeric=True)),
                        ("model", classification_model(model_name, params)),
                    ])
                    pipe.fit(X_train, y_train)
                    train_pred = pipe.predict(X_train)
                    test_pred = pipe.predict(X_test)
                    train_proba = pipe.predict_proba(X_train) if hasattr(pipe, "predict_proba") else None
                    test_proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
                    train_score = class_score(y_train, train_pred, train_proba)
                    test_score = class_score(y_test, test_pred, test_proba)
                    row = {
                        "Model": model_name,
                        f"Train {metric_name}": train_score,
                        f"Test {metric_name}": test_score,
                        "Overfitting gap": train_score - test_score if not pd.isna(train_score) and not pd.isna(test_score) else np.nan,
                    }
                    if use_cv and cv_folds is not None:
                        min_class = int(y.value_counts().min())
                        if min_class >= 2:
                            k_safe = min(int(cv_folds), min_class)
                            cv = StratifiedKFold(n_splits=k_safe, shuffle=True, random_state=int(random_seed))
                            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring_lookup[metric_name])
                            row[f"Cross-validation {metric_name}"] = float(np.mean(scores))
                            row["CV folds used"] = k_safe
                        else:
                            row[f"Cross-validation {metric_name}"] = np.nan
                            row["CV note"] = "At least one class has fewer than 2 rows."
                    results.append(row)
                    predictions[model_name] = (y_test, test_pred)
                except Exception as e:
                    results.append({"Model": model_name, "Error": str(e)})

            res_df = pd.DataFrame(results)
            score_col = f"Test {metric_name}"
            if score_col in res_df.columns:
                res_df = res_df.sort_values(score_col, ascending=False)
            st.subheader("Model comparison table")
            st.dataframe(res_df, use_container_width=True)

            if score_col in res_df.columns and res_df[score_col].notna().any():
                best_model = res_df.dropna(subset=[score_col]).iloc[0]["Model"]
                st.success(f"Best predictive model by {metric_name}: {best_model}")
                st.info("Interpretation tip: Logistic Regression, LDA, QDA, and Decision Tree are often easier to explain. Random Forest, Gradient Boosting, SVM, and neural networks may be stronger for prediction but less transparent.")

                plot_df = res_df[["Model", score_col]].dropna().set_index("Model")
                st.subheader(f"Test performance by model: {metric_name}")
                st.bar_chart(plot_df)

                if best_model in predictions:
                    y_test, test_pred = predictions[best_model]
                    st.subheader("Best model: confusion matrix")
                    labels = sorted(pd.Series(y_test).dropna().unique().tolist())
                    cm = confusion_matrix(y_test, test_pred, labels=labels)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title(f"Confusion Matrix: {best_model}")
                    st.pyplot(fig)
                    st.caption(f"Encoded class labels: {dict(enumerate(le.classes_))}")

                st.subheader("Overfitting check")
                gap_df = res_df[["Model", "Overfitting gap"]].dropna().set_index("Model")
                st.bar_chart(gap_df)
                st.caption("A larger positive gap means the training score is much higher than the test score. That can indicate overfitting.")



def page_cv():
    df = get_current_df()
    st.title("7. Cross-Validation Lab")
    lab_box(
        "Estimate generalization performance using repeated train/test splits through k-fold cross-validation.",
        ["k folds", "Model", "Metric", "Predictors"],
        ["Mean CV score", "Score variability", "Why one split can be misleading"],
        ["Run CV with k=5 and k=10. Did the conclusion change?"],
    )

    problem = st.radio("Problem", ["Regression", "Classification"], horizontal=True, key="cvprob")

    if problem == "Regression":
        target_options = numeric_cols(df)
        if not target_options:
            st.warning("Regression cross-validation needs a numeric target.")
            return
        target = stable_selectbox("Numeric target", target_options, key="cv_reg_target", default=target_options[-1])
    else:
        target_options = classification_target_cols(df)
        if not target_options:
            st.warning("Classification cross-validation needs a categorical or low-cardinality numeric target.")
            return
        target = stable_selectbox("Classification target", target_options, key="cv_clf_target", default=target_options[-1])

    feature_options = [c for c in df.columns if c != target]
    features = stable_multiselect("Predictor variables", feature_options, key=f"cv_{problem.lower()}_features", default=feature_options[: min(5, len(feature_options))])
    if not features:
        st.warning("Select at least one predictor.")
        return

    k_requested = st.slider("Number of folds", 3, 10, 5, key="cv_k")
    X, y_raw = split_xy(df, target, features)
    if len(X) < 3:
        st.warning("Need at least 3 usable rows for cross-validation.")
        return

    params = model_params_ui("cv_")

    if problem == "Regression":
        model_name = stable_selectbox(
            "Model",
            ["Linear Regression", "Ridge", "Elastic Net", "Random Forest", "Gradient Boosting", "Support Vector Regressor"],
            key="cv_reg_model",
            default="Linear Regression",
        )
        scoring_name = stable_selectbox(
            "Scoring metric",
            ["Mean Absolute Error", "Root Mean Squared Error", "R-squared"],
            key="cv_reg_metric",
            default="Mean Absolute Error",
        )
        scoring_map = {
            "Mean Absolute Error": "neg_mean_absolute_error",
            "Root Mean Squared Error": "neg_root_mean_squared_error",
            "R-squared": "r2",
        }
        scoring = scoring_map[scoring_name]
        pipe = Pipeline([("prep", make_preprocessor(X, features)), ("model", regression_model(model_name, params))])
        k_safe = min(int(k_requested), len(X))
        cv = KFold(n_splits=k_safe, shuffle=True, random_state=42)
        if k_safe < int(k_requested):
            st.info(f"Using {k_safe} folds because only {len(X)} usable rows are available.")
        if st.button("Run cross-validation", type="primary", key="cv_reg_run"):
            scores = cross_val_score(pipe, X, y_raw, cv=cv, scoring=scoring)
            shown = -scores if scoring.startswith("neg_") else scores
            st.write(pd.Series(shown).describe())
            st.line_chart(pd.DataFrame({f"{scoring_name} by fold": shown}))
    else:
        model_name = stable_selectbox(
            "Model",
            ["Logistic Regression", "LDA", "QDA", "KNN", "Random Forest", "Gradient Boosting", "Support Vector Machine"],
            key="cv_clf_model",
            default="Logistic Regression",
        )
        scoring_name = stable_selectbox("Scoring metric", ["Accuracy", "Weighted F1-score"], key="cv_clf_metric", default="Accuracy")
        scoring_map = {
            "Accuracy": "accuracy",
            "Weighted F1-score": "f1_weighted",
        }
        scoring = scoring_map[scoring_name]
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)
        if len(np.unique(y)) < 2:
            st.warning("The target needs at least two classes for classification.")
            return
        min_class = int(y.value_counts().min())
        if min_class < 2:
            st.warning("At least one class has fewer than 2 rows, so stratified cross-validation is not possible.")
            return
        k_safe = min(int(k_requested), min_class)
        if k_safe < int(k_requested):
            st.info(f"Using {k_safe} folds because the smallest class has only {min_class} rows.")
        pipe = Pipeline([("prep", make_preprocessor(X, features)), ("model", classification_model(model_name, params))])
        cv = StratifiedKFold(n_splits=k_safe, shuffle=True, random_state=42)
        if st.button("Run cross-validation", type="primary", key="cv_clf_run"):
            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
            st.write(pd.Series(scores).describe())
            st.line_chart(pd.DataFrame({f"{scoring_name} by fold": scores}))



def page_bootstrap():
    df = get_current_df()
    st.title("8. Bootstrap Lab")
    lab_box(
        "Use resampling with replacement to estimate uncertainty around a statistic.",
        ["Numeric variable", "Statistic", "Number of bootstrap samples"],
        ["Bootstrap distribution", "Confidence interval", "Sampling variability"],
        ["Compute a 95% bootstrap confidence interval and interpret it in words."],
    )

    nums = numeric_cols(df)
    if not nums:
        st.warning("Bootstrap needs a numeric variable.")
        return

    var = stable_selectbox("Numeric variable", nums, key="boot_var", default=nums[0])
    stat = stable_selectbox("Statistic", ["mean", "median", "standard deviation"], key="boot_stat", default="mean")
    B = st.slider("Bootstrap samples", 100, 5000, 1000, step=100, key="boot_B")
    data = df[var].dropna().to_numpy()
    if len(data) < 2:
        st.warning("Need at least two non-missing values for bootstrap.")
        return

    if st.button("Run bootstrap", type="primary", key="boot_run"):
        rng = np.random.default_rng(42)
        vals = []
        for _ in range(B):
            sample = rng.choice(data, size=len(data), replace=True)
            vals.append(np.mean(sample) if stat == "mean" else np.median(sample) if stat == "median" else np.std(sample, ddof=1))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        st.metric("95% bootstrap CI", f"[{lo:.3f}, {hi:.3f}]")
        fig, ax = plt.subplots()
        sns.histplot(vals, kde=True, ax=ax)
        ax.axvline(lo, linestyle="--")
        ax.axvline(hi, linestyle="--")
        ax.set_title("Bootstrap Distribution")
        st.pyplot(fig)


def make_lag_features(df: pd.DataFrame, date_col: str, target: str, lags: List[int]) -> pd.DataFrame:
    out = df[[date_col, target]].dropna().copy().sort_values(date_col)
    for lag in lags:
        out[f"lag_{lag}"] = out[target].shift(lag)
    out["rolling_mean_3"] = out[target].shift(1).rolling(3).mean()
    out["rolling_mean_7"] = out[target].shift(1).rolling(7).mean()
    return out.dropna()



def page_forecasting():
    df = get_current_df()
    st.title("9. Forecasting Lab")
    lab_box(
        "Convert a time series into supervised learning using lagged features, then forecast future values.",
        ["Date column", "Numeric target", "Lags", "Model"],
        ["Train/test split by time", "Why random splitting is dangerous for forecasting", "MAE/RMSE and observed-vs-forecast plot"],
        ["Build a one-step-ahead forecast and explain whether lagged values were useful."],
    )

    cols = df.columns.tolist()
    nums = numeric_cols(df)
    if not nums:
        st.warning("Forecasting needs a numeric target variable.")
        return

    date_col = stable_selectbox("Date/time column", cols, key="fc_date_col", default=cols[0])
    target = stable_selectbox("Numeric forecast target", nums, key="fc_target", default=nums[-1])

    try:
        df2 = df.copy()
        df2[date_col] = pd.to_datetime(df2[date_col])
    except Exception:
        st.warning("Selected date column could not be parsed as datetime. Choose a date/time column.")
        return

    lag_text = st.text_input("Lags, comma-separated", "1,2,3,7", key="fc_lags")
    lags = [int(x.strip()) for x in lag_text.split(",") if x.strip().isdigit()]
    if not lags:
        st.warning("Enter at least one lag, such as 1 or 1,2,3.")
        return

    model_name = stable_selectbox(
        "Forecast model",
        ["Linear Regression", "Ridge", "Elastic Net", "Random Forest", "Gradient Boosting", "Support Vector Regressor", "Neural Network / FFN"],
        key="fc_model",
        default="Linear Regression",
    )
    params = model_params_ui("fc_")

    if st.button("Run forecast model", type="primary", key="fc_run"):
        sup = make_lag_features(df2, date_col, target, lags)
        if len(sup) < 10:
            st.warning("Not enough rows remain after creating lag features. Use fewer/lower lags or a longer time series.")
            return
        features = [c for c in sup.columns if c not in [date_col, target]]
        split = int(len(sup) * 0.75)
        train, test = sup.iloc[:split], sup.iloc[split:]
        X_train, y_train = train[features], train[target]
        X_test, y_test = test[features], test[target]
        pipe = Pipeline([("prep", make_preprocessor(X_train, features)), ("model", regression_model(model_name, params))])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        st.dataframe(
            pd.DataFrame({"Metric": ["MAE", "RMSE"], "Value": [mean_absolute_error(y_test, pred), math.sqrt(mean_squared_error(y_test, pred))]}),
            use_container_width=True,
        )
        plotdf = pd.DataFrame({"date": test[date_col].values, "observed": y_test.values, "forecast": pred})
        st.line_chart(plotdf.set_index("date"))
        st.info("Forecasting note: this page uses a time-ordered split, not a random split, to avoid look-ahead bias.")


def image_feature_vector(img: Image.Image, size: int = 64) -> np.ndarray:
    """Create a small, fast feature vector from an image.

    This is intentionally simple for undergraduate labs: resized RGB pixels plus
    basic color summaries. It is not a replacement for a CNN, but it lets
    students train and test a real computer-vision classifier inside Streamlit.
    """
    img = img.convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    flat = arr.reshape(-1)
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    return np.concatenate([flat, means, stds])


def read_image_zip(zip_file, max_images_per_class: int = 60, size: int = 64) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Read a ZIP where images are stored in class folders.

    Expected structure:
        plant_subset.zip
            Healthy/image1.jpg
            Healthy/image2.jpg
            Diseased/image1.jpg
            Diseased/image2.jpg

    Returns X feature matrix, y labels, and a small manifest table.
    """
    X, y, rows = [], [], []
    counts: Dict[str, int] = {}
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    with zipfile.ZipFile(zip_file) as z:
        names = [n for n in z.namelist() if n.lower().endswith(valid_ext) and not n.startswith("__MACOSX/")]
        for name in names:
            parts = [p for p in name.split("/") if p]
            if len(parts) < 2:
                continue
            label = parts[-2]
            counts.setdefault(label, 0)
            if counts[label] >= max_images_per_class:
                continue
            try:
                with z.open(name) as f:
                    img = Image.open(f).convert("RGB")
                    X.append(image_feature_vector(img, size=size))
                    y.append(label)
                    counts[label] += 1
                    rows.append({"file": name, "class": label, "width": img.width, "height": img.height})
            except Exception:
                continue
    if len(X) == 0:
        return np.empty((0, 0)), np.array([]), pd.DataFrame()
    return np.vstack(X), np.array(y), pd.DataFrame(rows)



def cv_model(name: str, params: Dict[str, Any]):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, C=params.get("C", 1.0))
    if name == "Linear SVM":
        return SVC(kernel="linear", C=params.get("C", 1.0), probability=True)
    if name == "RBF SVM":
        return SVC(kernel="rbf", C=params.get("C", 1.0), gamma="scale", probability=True)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=params.get("n_estimators", 100), max_depth=params.get("max_depth", None), random_state=42)
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=params.get("k", 5))
    return MLPClassifier(hidden_layer_sizes=(params.get("hidden", 64),), max_iter=300, random_state=42)


@st.cache_resource(show_spinner=False)
def load_imagenet_model():
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    return MobileNetV2(weights="imagenet")


def imagenet_predict(img: Image.Image, top_k: int = 5) -> pd.DataFrame:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
    from tensorflow.keras.preprocessing.image import img_to_array
    model = load_imagenet_model()
    img2 = img.convert("RGB").resize((224, 224))
    x = img_to_array(img2)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x, verbose=0)
    decoded = decode_predictions(preds, top=top_k)[0]
    return pd.DataFrame([
        {"prediction": label.replace("_", " "), "confidence": float(prob)}
        for _, label, prob in decoded
    ])


def instant_cv_available() -> Tuple[bool, str]:
    try:
        import tensorflow  # noqa: F401
        return True, ""
    except Exception as e:
        return False, str(e)


def page_cv_images():
    st.title("10. Computer Vision Lab")
    lab_box(
        "Use computer vision in two ways: instant recognition with a pretrained model, or train a small classroom image classifier.",
        ["Upload one image for instant recognition", "Optionally upload a ZIP of labeled image folders", "Choose classifier", "Compare predictions and model accuracy"],
        ["Images are arrays of pixels", "Pretrained models already learned from millions of images", "Classroom models must be trained on labeled examples", "Why validation data matters"],
        ["Upload one image, record the top prediction and confidence, then explain why a pretrained model may still make mistakes."],
    )

    st.markdown("""
### Two computer-vision modes

**Mode A — Instant image recognition**  
Upload one image and the app predicts immediately using a pretrained ImageNet model. This is the mode to use when students upload a dog, cat, laptop, car, banana, chair, etc.

**Mode B — Train your own image classifier**  
Upload a ZIP file where each folder is a class name. This is better for PlantVillage or any class-specific dataset.

**Important teaching point:** instant recognition and class-trained recognition are different. The instant model already knows common ImageNet objects. A PlantVillage disease classifier must be trained or fine-tuned on plant-disease images.
""")

    mode = st.radio(
        "Choose computer-vision mode",
        ["Instant image recognition", "Train my own image classifier"],
        horizontal=True,
    )

    if mode == "Instant image recognition":
        st.subheader("Instant image recognition")
        st.write("Upload one image. The app will predict immediately. No training is required.")
        ok, err = instant_cv_available()
        if not ok:
            st.error("Instant recognition requires TensorFlow. Install it with: pip install tensorflow")
            st.code("pip install tensorflow", language="bash")
            st.caption(f"Technical message: {err}")
            return

        img_file = st.file_uploader("Upload one image", type=["png", "jpg", "jpeg", "webp"], key="instant_cv_single")
        top_k = st.slider("Number of top predictions", 1, 10, 5)
        if img_file:
            img = Image.open(img_file).convert("RGB")
            st.image(img, caption="Uploaded image", width=380)
            arr = np.array(img)
            c1, c2, c3 = st.columns(3)
            c1.metric("Height", arr.shape[0])
            c2.metric("Width", arr.shape[1])
            c3.metric("Channels", arr.shape[2])

            if st.button("Predict image", type="primary"):
                with st.spinner("Loading pretrained model and predicting..."):
                    pred_df = imagenet_predict(img, top_k=top_k)
                best = pred_df.iloc[0]
                st.success(f"Top prediction: {best['prediction']} ({best['confidence']:.1%} confidence)")
                st.dataframe(pred_df, use_container_width=True)
                st.bar_chart(pred_df.set_index("prediction"))
                st.info("Interpretation note: the model returns the most likely ImageNet labels. It may be wrong if the object is unusual, cropped, blurry, or outside the categories it learned.")
        else:
            st.info("Upload a JPG or PNG image to see instant predictions.")
        return

    st.subheader("Train my own image classifier")
    st.markdown("""
### Recommended dataset
Use a small subset of the **PlantVillage Kaggle dataset**. For class use, create a small ZIP file with 2–4 folders/classes and about 20–60 images per class.

**Required ZIP structure:**
```text
plant_subset.zip
├── Healthy/
│   ├── image_1.jpg
│   └── image_2.jpg
├── Early_blight/
│   ├── image_1.jpg
│   └── image_2.jpg
└── Late_blight/
    ├── image_1.jpg
    └── image_2.jpg
```

This mode uses simple image features and classical ML models so it runs quickly in a teaching setting. It is a real computer-vision workflow, but not yet a full CNN pipeline.
""")
    st.markdown("**PlantVillage Kaggle link:** https://www.kaggle.com/datasets/mohitsingh1804/plantvillage")

    c1, c2, c3 = st.columns(3)
    with c1:
        image_size = st.selectbox("Image resize size", [32, 48, 64, 96], index=2)
    with c2:
        max_per_class = st.slider("Max images per class", 10, 200, 60, step=10)
    with c3:
        test_size = st.slider("Test size", 0.15, 0.50, 0.25, step=0.05)

    zip_file = st.file_uploader("Upload ZIP of image folders", type=["zip"], key="cv_zip")
    if zip_file is not None:
        with st.spinner("Reading images and creating features..."):
            X, y, manifest = read_image_zip(zip_file, max_images_per_class=max_per_class, size=image_size)
        if len(y) == 0:
            st.error("No images found. Make sure the ZIP has images inside class folders.")
            return
        st.success(f"Loaded {len(y)} images from {len(np.unique(y))} classes.")
        st.dataframe(manifest.head(20), use_container_width=True)
        st.bar_chart(pd.Series(y).value_counts())

        model_name = st.selectbox("Image classifier", ["Logistic Regression", "Linear SVM", "RBF SVM", "Random Forest", "KNN", "Neural Network / FFN"])
        params = {
            "C": st.slider("C regularization parameter", 0.01, 10.0, 1.0),
            "n_estimators": st.slider("Trees for Random Forest", 50, 300, 100, step=50),
            "max_depth": st.slider("Tree max depth; 0 means unlimited", 0, 30, 10),
            "k": st.slider("K for KNN", 1, 15, 5),
            "hidden": st.slider("Hidden neurons for FFN", 16, 256, 64, step=16),
        }
        if params["max_depth"] == 0:
            params["max_depth"] = None

        if st.button("Train image classifier", type="primary"):
            if len(np.unique(y)) < 2:
                st.error("Need at least two image classes.")
                return
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            except Exception:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            clf = Pipeline([
                ("scale", StandardScaler(with_mean=True)),
                ("model", cv_model(model_name, params)),
            ])
            with st.spinner("Training model..."):
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
            st.session_state.cv_image_model = clf
            st.session_state.cv_image_size = image_size
            st.session_state.cv_image_classes = sorted(np.unique(y).tolist())

            c1, c2, c3 = st.columns(3)
            c1.metric("Test accuracy", f"{accuracy_score(y_test, pred):.3f}")
            c2.metric("Weighted F1", f"{f1_score(y_test, pred, average='weighted', zero_division=0):.3f}")
            c3.metric("Classes", len(np.unique(y)))

            st.subheader("Confusion matrix")
            labels = sorted(np.unique(y).tolist())
            cm = confusion_matrix(y_test, pred, labels=labels)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            st.subheader("Classification report")
            report = classification_report(y_test, pred, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).T, use_container_width=True)
            st.info("Teaching note: if accuracy is high but one class performs poorly, students should discuss imbalance and collect more images for that class.")

    st.divider()
    st.subheader("Classify a new image using your trained classifier")
    img_file = st.file_uploader("Upload one image for prediction", type=["png", "jpg", "jpeg"], key="cv_single")
    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, caption="Uploaded image", width=350)
        arr = np.array(img)
        c1, c2, c3 = st.columns(3)
        c1.metric("Height", arr.shape[0])
        c2.metric("Width", arr.shape[1])
        c3.metric("Channels", arr.shape[2])

        if "cv_image_model" in st.session_state:
            size = st.session_state.get("cv_image_size", 64)
            feat = image_feature_vector(img, size=size).reshape(1, -1)
            model = st.session_state.cv_image_model
            pred = model.predict(feat)[0]
            st.success(f"Predicted class: {pred}")
            if hasattr(model.named_steps["model"], "predict_proba"):
                try:
                    probs = model.predict_proba(feat)[0]
                    classes = model.classes_
                    prob_df = pd.DataFrame({"class": classes, "probability": probs}).sort_values("probability", ascending=False)
                    st.dataframe(prob_df, use_container_width=True)
                    st.bar_chart(prob_df.set_index("class"))
                except Exception:
                    pass
        else:
            st.warning("Train an image classifier above before predicting with this custom model. For no-training prediction, switch to Instant image recognition.")


def page_assignment():

    st.title("11. Assignment Builder")
    st.markdown("Generate a simple lab assignment prompt students can complete in class or submit after class.")
    topic = st.selectbox("Lab topic", ["Exploration", "Correlation", "Regression", "Classification", "Cross-validation", "Bootstrap", "Forecasting", "Computer Vision"])
    domain = st.selectbox("Domain", ["Behavior/Society", "Finance", "Environment", "Health", "Computer Vision"])
    st.subheader("Student assignment")
    prompt = f"""
### MATH 490 Lab Assignment: {topic} in {domain}

1. State the research question in one sentence.  
2. Identify the dataset, target variable, and predictor variables.  
3. Show one table or plot that helps explain the data.  
4. Fit the selected model or method in the app.  
5. Report the main result using correct statistical language.  
6. Interpret the result for a non-technical audience.  
7. State one limitation of the analysis.  
8. Submit one screenshot of your model output and one paragraph of interpretation.
"""
    st.markdown(prompt)
    st.download_button("Download assignment prompt", prompt, file_name=f"MATH490_{topic}_{domain}_assignment.md")



def render_footer():
    st.markdown(
        """
        <hr style="margin-top: 2rem; margin-bottom: 0.75rem;">
        <div style="text-align:center; color:#666; font-size:0.9rem;">
            This app was created by <strong>Chibuike Ibebuchi, PhD</strong>. 
            All rights reserved. &copy; 2026.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    render_dataset_sidebar()
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", PAGE_LIST)
    if page == "Home": page_home()
    elif page == "1. Data Explorer": page_data_explorer()
    elif page == "2. Visualization Lab": page_visualization()
    elif page == "3. Correlation & Partial Correlation": page_correlation()
    elif page == "4. Regression Lab": page_regression()
    elif page == "5. Classification Lab": page_classification()
    elif page == "6. Model Evaluation & Comparison": page_model_eval()
    elif page == "7. Cross-Validation Lab": page_cv()
    elif page == "8. Bootstrap Lab": page_bootstrap()
    elif page == "9. Forecasting Lab": page_forecasting()
    elif page == "10. Computer Vision Lab": page_cv_images()
    elif page == "11. Assignment Builder": page_assignment()
    render_footer()

if __name__ == "__main__":
    main()
