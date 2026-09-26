
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_absolute_error,
    mean_squared_error, r2_score
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Cholera Outbreak Surveillance System – Bauchi State",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# DATA
# -----------------------------
DATA_FILE = "bauchi_cholera_2019_52weeks.csv"

@st.cache_data
def load_data():
    path = Path(DATA_FILE)
    if not path.exists():
        st.error(f"Dataset not found: {DATA_FILE}")
        st.stop()

    d = pd.read_csv(path)

    text_cols = [
        "LGA", "Primary_Water_Source", "Secondary_Water_Source",
        "Water_Safety_Level", "Sanitation_Type", "Waste_Management_Level",
        "Access_Level", "Flood_Risk_Level", "Proximity_to_Water_Body",
        "Environmental_Risk"
    ]
    for c in text_cols:
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip()

    d = d.sort_values(["LGA", "epi_week"]).reset_index(drop=True)
    return d

df_raw = load_data()

# -----------------------------
# FORECAST DATASET
# Predict NEXT WEEK'S cholera activity/cases from information available
# in the current week.
# -----------------------------
NUMERIC_BASE = [
    "Rainfall_mm", "Temperature_C", "Humidity_%",
    "Toilet_Access_%", "Open_Defecation_%",
    "Health_Facilities_Count", "Hospital_Count",
    "Health_Workers", "Elevation_m", "Population (2016 census)"
]

CAT_BASE = [
    "LGA", "Primary_Water_Source", "Secondary_Water_Source",
    "Water_Safety_Level", "Sanitation_Type", "Waste_Management_Level",
    "Access_Level", "Flood_Risk_Level", "Proximity_to_Water_Body"
]

LAG_FEATURES = [
    "cases_lag1", "cases_lag2", "cases_lag3", "cases_lag4",
    "deaths_lag1", "deaths_lag2",
    "cases_roll4_mean", "cases_roll4_max"
]

FEATURES = NUMERIC_BASE + CAT_BASE + LAG_FEATURES

@st.cache_data
def make_forecast_data(d):
    x = d.copy()

    for lag in range(1, 5):
        x[f"cases_lag{lag}"] = x.groupby("LGA")["Number of cases"].shift(lag)
    for lag in range(1, 3):
        x[f"deaths_lag{lag}"] = x.groupby("LGA")["Deaths"].shift(lag)

    x["cases_roll4_mean"] = (
        x.groupby("LGA")["Number of cases"]
        .transform(lambda s: s.shift(1).rolling(4, min_periods=4).mean())
    )
    x["cases_roll4_max"] = (
        x.groupby("LGA")["Number of cases"]
        .transform(lambda s: s.shift(1).rolling(4, min_periods=4).max())
    )

    # Forecast target: the following epidemiological week's observed cases.
    x["next_week_cases"] = x.groupby("LGA")["Number of cases"].shift(-1)
    x["next_week_has_cases"] = (x["next_week_cases"] > 0).astype(int)

    # The final week has no future target and is excluded from training/testing.
    x = x.dropna(subset=["next_week_cases"] + LAG_FEATURES).copy()
    return x

forecast_df = make_forecast_data(df_raw)

# -----------------------------
# MODEL TRAINING
# -----------------------------
def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_BASE + LAG_FEATURES),
        ("cat", categorical_pipe, CAT_BASE),
    ])

@st.cache_resource
def train_models(data):
    # Strict temporal evaluation:
    # weeks 5–41 -> training
    # weeks 42–51 -> testing
    train = data[data["epi_week"] <= 41].copy()
    test = data[data["epi_week"] >= 42].copy()

    X_train = train[FEATURES]
    X_test = test[FEATURES]

    y_train_cls = train["next_week_has_cases"]
    y_test_cls = test["next_week_has_cases"]

    y_train_reg = train["next_week_cases"]
    y_test_reg = test["next_week_cases"]

    cls_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            max_depth=2,
            learning_rate=0.04,
            random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    }

    cls_results = {}
    cls_pipelines = {}

    for name, model in cls_models.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ])
        pipe.fit(X_train, y_train_cls)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]

        cls_results[name] = {
            "accuracy": accuracy_score(y_test_cls, pred),
            "precision": precision_score(y_test_cls, pred, zero_division=0),
            "recall": recall_score(y_test_cls, pred, zero_division=0),
            "f1": f1_score(y_test_cls, pred, zero_division=0),
            "pred": pred,
            "proba": proba,
        }
        cls_pipelines[name] = pipe

    # Select the early-warning classifier by F1, not raw accuracy.
    best_cls = max(cls_results, key=lambda n: cls_results[n]["f1"])

    reg_models = {
        "Poisson Regression": PoissonRegressor(alpha=1.0, max_iter=2000),
        "Gradient Boosting": GradientBoostingClassifier,  # placeholder replaced below
    }

    # Count prediction models.
    poisson_pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", PoissonRegressor(alpha=1.0, max_iter=2000)),
    ])
    poisson_pipe.fit(X_train, y_train_reg)
    poisson_pred = np.maximum(0, poisson_pipe.predict(X_test))

    # A tree regression model is useful for comparison.
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    rf_reg_pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1
        )),
    ])
    rf_reg_pipe.fit(X_train, y_train_reg)
    rf_pred = np.maximum(0, rf_reg_pipe.predict(X_test))

    gb_reg_pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", GradientBoostingRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.04,
            loss="huber", random_state=42
        )),
    ])
    gb_reg_pipe.fit(X_train, y_train_reg)
    gb_pred = np.maximum(0, gb_reg_pipe.predict(X_test))

    reg_results = {
        "Poisson Regression": {
            "mae": mean_absolute_error(y_test_reg, poisson_pred),
            "rmse": mean_squared_error(y_test_reg, poisson_pred) ** 0.5,
            "r2": r2_score(y_test_reg, poisson_pred),
            "pred": poisson_pred,
        },
        "Random Forest": {
            "mae": mean_absolute_error(y_test_reg, rf_pred),
            "rmse": mean_squared_error(y_test_reg, rf_pred) ** 0.5,
            "r2": r2_score(y_test_reg, rf_pred),
            "pred": rf_pred,
        },
        "Gradient Boosting": {
            "mae": mean_absolute_error(y_test_reg, gb_pred),
            "rmse": mean_squared_error(y_test_reg, gb_pred) ** 0.5,
            "r2": r2_score(y_test_reg, gb_pred),
            "pred": gb_pred,
        },
    }

    best_reg = min(reg_results, key=lambda n: reg_results[n]["mae"])

    reg_pipelines = {
        "Poisson Regression": poisson_pipe,
        "Random Forest": rf_reg_pipe,
        "Gradient Boosting": gb_reg_pipe,
    }

    return {
        "train": train,
        "test": test,
        "cls_results": cls_results,
        "cls_pipelines": cls_pipelines,
        "best_cls": best_cls,
        "reg_results": reg_results,
        "reg_pipelines": reg_pipelines,
        "best_reg": best_reg,
    }

models = train_models(forecast_df)

# -----------------------------
# SIDEBAR
# -----------------------------
LGAS = sorted(df_raw["LGA"].unique())
RISK_LEVELS = ["Low", "Moderate", "High"]

with st.sidebar:
    st.markdown("## 🦠 Cholera Outbreak Surveillance")
    st.markdown("**Bauchi State Early-Warning System**")
    st.markdown("---")

    nav = st.radio(
        "Navigation",
        [
            "📊 Dashboard Overview",
            "🗺️ Geographic Analysis",
            "📈 Trend Analysis",
            "🤖 Cholera Prediction",
            "🔍 Risk Assessment",
            "📋 Data Explorer",
            "ℹ️ About",
        ],
    )

    st.markdown("---")
    sel_lgas = st.multiselect("Filter by LGA", LGAS, default=[])

    epi_range = st.slider(
        "Epidemiological Week",
        int(df_raw["epi_week"].min()),
        int(df_raw["epi_week"].max()),
        (int(df_raw["epi_week"].min()), int(df_raw["epi_week"].max())),
    )

# Apply display filters only to surveillance pages.
df = df_raw.copy()
if sel_lgas:
    df = df[df["LGA"].isin(sel_lgas)]
df = df[df["epi_week"].between(epi_range[0], epi_range[1])]

# -----------------------------
# DASHBOARD OVERVIEW
# -----------------------------
if nav == "📊 Dashboard Overview":
    st.title("📊 Cholera Outbreak Surveillance Dashboard")
    st.caption("Bauchi State • 2019 • 52 epidemiological weeks")

    total_cases = int(df["Number of cases"].sum())
    total_deaths = int(df["Deaths"].sum())
    active_lgas = int((df.groupby("LGA")["Number of cases"].sum() > 0).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recorded Cholera Cases", f"{total_cases:,}")
    c2.metric("Recorded Deaths", f"{total_deaths:,}")
    c3.metric("LGAs With Cases", active_lgas)
    c4.metric("Weeks Covered", int(df["epi_week"].nunique()))

    weekly = (
        df.groupby("epi_week", as_index=False)["Number of cases"]
        .sum()
        .sort_values("epi_week")
    )

    fig = px.bar(
        weekly,
        x="epi_week",
        y="Number of cases",
        title="Weekly Reported Cholera Cases",
        labels={"epi_week": "Epidemiological Week", "Number of cases": "Cases"},
    )
    st.plotly_chart(fig, use_container_width=True)

    lga = (
        df.groupby("LGA", as_index=False)["Number of cases"]
        .sum()
        .sort_values("Number of cases", ascending=False)
    )
    fig2 = px.bar(
        lga,
        x="Number of cases",
        y="LGA",
        orientation="h",
        title="Reported Cholera Cases by LGA",
    )
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# GEOGRAPHIC
# -----------------------------
elif nav == "🗺️ Geographic Analysis":
    st.title("🗺️ Geographic Analysis")

    lga = (
        df.groupby("LGA")
        .agg(
            Cases=("Number of cases", "sum"),
            Deaths=("Deaths", "sum"),
            Mean_Rainfall=("Rainfall_mm", "mean"),
            Mean_Open_Defecation=("Open_Defecation_%", "mean"),
        )
        .reset_index()
        .sort_values("Cases", ascending=False)
    )

    fig = px.bar(
        lga,
        x="Cases",
        y="LGA",
        orientation="h",
        title="Cumulative Reported Cholera Cases by LGA",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(lga, use_container_width=True)

# -----------------------------
# TREND
# -----------------------------
elif nav == "📈 Trend Analysis":
    st.title("📈 Epidemiological Trend Analysis")

    weekly = (
        df.groupby("epi_week")
        .agg(
            Cases=("Number of cases", "sum"),
            Rainfall=("Rainfall_mm", "mean"),
            Temperature=("Temperature_C", "mean"),
        )
        .reset_index()
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=weekly["epi_week"], y=weekly["Cases"], name="Reported Cases"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=weekly["epi_week"],
            y=weekly["Rainfall"],
            mode="lines+markers",
            name="Mean Rainfall",
        ),
        secondary_y=True,
    )
    fig.update_xaxes(title_text="Epidemiological Week")
    fig.update_yaxes(title_text="Cholera Cases", secondary_y=False)
    fig.update_yaxes(title_text="Rainfall (mm)", secondary_y=True)
    fig.update_layout(title="Cholera Cases and Rainfall by Week", height=450)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# PREDICTIVE MODEL
# -----------------------------
elif nav == "🤖 Cholera Prediction":
    st.title("🤖 ML Cholera Outbreak Prediction")
    st.write(
        "The model predicts whether cholera cases will be reported in the "
        "following epidemiological week and separately estimates the expected "
        "number of cases."
    )

    st.info(
        "Evaluation is chronological: earlier weeks are used for training and "
        "later weeks are held out for testing. This avoids randomly mixing "
        "future observations into the training data."
    )

    cls_rows = []
    for name, r in models["cls_results"].items():
        cls_rows.append({
            "Model": name,
            "Accuracy": round(r["accuracy"], 3),
            "Precision": round(r["precision"], 3),
            "Recall": round(r["recall"], 3),
            "F1": round(r["f1"], 3),
        })
    cls_table = pd.DataFrame(cls_rows).sort_values("F1", ascending=False)

    st.subheader("1. Cholera Activity Prediction")
    st.dataframe(cls_table, use_container_width=True)

    best_cls = models["best_cls"]
    r = models["cls_results"][best_cls]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Classifier", best_cls)
    c2.metric("Accuracy", f"{r['accuracy']*100:.1f}%")
    c3.metric("Recall", f"{r['recall']*100:.1f}%")
    c4.metric("F1 Score", f"{r['f1']*100:.1f}%")

    cm = confusion_matrix(
        models["test"]["next_week_has_cases"], r["pred"], labels=[0, 1]
    )
    fig_cm = px.imshow(
        cm,
        x=["No cases", "Cases"],
        y=["No cases", "Cases"],
        text_auto=True,
        title="Confusion Matrix – Next-Week Cholera Activity",
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("2. Next-Week Cholera Case Prediction")
    reg_rows = []
    for name, rr in models["reg_results"].items():
        reg_rows.append({
            "Model": name,
            "MAE (cases)": round(rr["mae"], 2),
            "RMSE (cases)": round(rr["rmse"], 2),
            "R²": round(rr["r2"], 3),
        })
    st.dataframe(
        pd.DataFrame(reg_rows).sort_values("MAE (cases)"),
        use_container_width=True,
    )

    best_reg = models["best_reg"]
    rr = models["reg_results"][best_reg]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Case Model", best_reg)
    c2.metric("MAE", f"{rr['mae']:.2f} cases")
    c3.metric("RMSE", f"{rr['rmse']:.2f} cases")

    test_plot = models["test"][["LGA", "epi_week", "next_week_cases"]].copy()
    test_plot["Predicted"] = rr["pred"]
    test_plot["Observed"] = test_plot["next_week_cases"]

    # Aggregate observed/predicted cases by target week for a clear surveillance view.
    test_plot["target_week"] = test_plot["epi_week"] + 1
    weekly_pred = (
        test_plot.groupby("target_week")
        .agg(Observed=("Observed", "sum"), Predicted=("Predicted", "sum"))
        .reset_index()
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=weekly_pred["target_week"],
        y=weekly_pred["Observed"],
        mode="lines+markers",
        name="Observed Cases",
    ))
    fig2.add_trace(go.Scatter(
        x=weekly_pred["target_week"],
        y=weekly_pred["Predicted"],
        mode="lines+markers",
        name="Predicted Cases",
    ))
    fig2.update_layout(
        title=f"Observed vs Predicted Cholera Cases ({best_reg})",
        xaxis_title="Target Epidemiological Week",
        yaxis_title="Cases",
        height=430,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Important: the case model is trained on only one year of data (2019) "
        "and should be treated as a research prototype, not a clinically or "
        "public-health validated forecasting service."
    )

# -----------------------------
# RISK ASSESSMENT
# -----------------------------
elif nav == "🔍 Risk Assessment":
    st.title("🔍 Next-Week Cholera Risk Assessment")

    st.write(
        "Enter information available for the current week. The system estimates "
        "the probability that cholera cases will be reported in the next week "
        "and predicts the expected case count."
    )

    clf = models["cls_pipelines"][models["best_cls"]]
    reg = models["reg_pipelines"][models["best_reg"]]

    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            lga = st.selectbox("LGA", LGAS)
            rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 50.0)
            temperature = st.number_input("Temperature (°C)", 10.0, 50.0, 30.0)
            humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)

        with col2:
            toilet = st.number_input("Toilet Access (%)", 0.0, 100.0, 40.0)
            open_def = st.number_input("Open Defecation (%)", 0.0, 100.0, 25.0)
            water = st.selectbox(
                "Primary Water Source",
                sorted(df_raw["Primary_Water_Source"].unique())
            )
            secondary_water = st.selectbox(
                "Secondary Water Source",
                sorted(df_raw["Secondary_Water_Source"].unique())
            )

        with col3:
            water_safety = st.selectbox(
                "Water Safety Level",
                sorted(df_raw["Water_Safety_Level"].unique())
            )
            sanitation = st.selectbox(
                "Sanitation Type",
                sorted(df_raw["Sanitation_Type"].unique())
            )
            waste = st.selectbox(
                "Waste Management Level",
                sorted(df_raw["Waste_Management_Level"].unique())
            )
            access = st.selectbox(
                "Access Level",
                sorted(df_raw["Access_Level"].unique())
            )

        col4, col5, col6 = st.columns(3)
        with col4:
            flood = st.selectbox(
                "Flood Risk Level",
                sorted(df_raw["Flood_Risk_Level"].unique())
            )
            proximity = st.selectbox(
                "Proximity to Water Body",
                sorted(df_raw["Proximity_to_Water_Body"].unique())
            )
            elevation = st.number_input("Elevation (m)", 0.0, 3000.0, 400.0)

        with col5:
            facilities = st.number_input("Health Facilities", 0, 100, 15)
            hospitals = st.number_input("Hospitals", 0, 50, 2)
            health_workers = st.number_input("Health Workers", 0, 1000, 30)

        with col6:
            population_default = int(
                df_raw.loc[df_raw["LGA"] == lga, "Population (2016 census)"].iloc[0]
            )
            population = st.number_input(
                "Population", min_value=1, value=population_default
            )

        st.markdown("### Previous cholera surveillance")
        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            lag1 = st.number_input("Cases – previous week", 0.0, 10000.0, 0.0)
            lag2 = st.number_input("Cases – 2 weeks ago", 0.0, 10000.0, 0.0)
        with lc2:
            lag3 = st.number_input("Cases – 3 weeks ago", 0.0, 10000.0, 0.0)
            lag4 = st.number_input("Cases – 4 weeks ago", 0.0, 10000.0, 0.0)
        with lc3:
            dlag1 = st.number_input("Deaths – previous week", 0.0, 1000.0, 0.0)
            dlag2 = st.number_input("Deaths – 2 weeks ago", 0.0, 1000.0, 0.0)

        run = st.form_submit_button("🚨 Predict Next-Week Cholera Risk")

    if run:
        row = pd.DataFrame([{
            "Rainfall_mm": rainfall,
            "Temperature_C": temperature,
            "Humidity_%": humidity,
            "Toilet_Access_%": toilet,
            "Open_Defecation_%": open_def,
            "Health_Facilities_Count": facilities,
            "Hospital_Count": hospitals,
            "Health_Workers": health_workers,
            "Elevation_m": elevation,
            "Population (2016 census)": population,
            "LGA": lga,
            "Primary_Water_Source": water,
            "Secondary_Water_Source": secondary_water,
            "Water_Safety_Level": water_safety,
            "Sanitation_Type": sanitation,
            "Waste_Management_Level": waste,
            "Access_Level": access,
            "Flood_Risk_Level": flood,
            "Proximity_to_Water_Body": proximity,
            "cases_lag1": lag1,
            "cases_lag2": lag2,
            "cases_lag3": lag3,
            "cases_lag4": lag4,
            "deaths_lag1": dlag1,
            "deaths_lag2": dlag2,
            "cases_roll4_mean": np.mean([lag1, lag2, lag3, lag4]),
            "cases_roll4_max": np.max([lag1, lag2, lag3, lag4]),
        }])

        probability = float(clf.predict_proba(row)[0, 1])
        predicted_cases = max(0.0, float(reg.predict(row)[0]))

        if probability < 0.33:
            risk = "LOW"
        elif probability < 0.67:
            risk = "MODERATE"
        else:
            risk = "HIGH"

        if risk == "HIGH":
            st.error(f"🔴 HIGH CHOLERA ACTIVITY RISK — {probability*100:.1f}%")
        elif risk == "MODERATE":
            st.warning(f"🟡 MODERATE CHOLERA ACTIVITY RISK — {probability*100:.1f}%")
        else:
            st.success(f"🟢 LOW CHOLERA ACTIVITY RISK — {probability*100:.1f}%")

        c1, c2 = st.columns(2)
        c1.metric("Probability of Cases Next Week", f"{probability*100:.1f}%")
        c2.metric("Predicted Cases Next Week", f"{predicted_cases:.1f}")

        st.info(
            "Risk bands are a research-defined interpretation of the model's "
            "probability of next-week cholera activity. They are not an official "
            "declaration of a cholera outbreak."
        )

# -----------------------------
# DATA EXPLORER
# -----------------------------
elif nav == "📋 Data Explorer":
    st.title("📋 Data Explorer")
    st.dataframe(df, use_container_width=True)

    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    st.download_button(
        "📥 Download filtered data",
        data=buf.getvalue(),
        file_name="Surveillance_Matrix_Subset.csv",
        mime="text/csv",
    )

# -----------------------------
# ABOUT
# -----------------------------
elif nav == "ℹ️ About":
    st.title("ℹ️ About the System")
    st.write(
        "This research prototype uses historical Bauchi State weekly surveillance "
        "records to estimate next-week cholera activity and case burden."
    )

    st.subheader("What the ML model now predicts")
    st.markdown("""
    - **Next-week cholera activity:** probability that at least one cholera case
      will be reported in an LGA in the following epidemiological week.
    - **Next-week case count:** estimated number of cholera cases.
    - **Risk category:** Low, Moderate, or High based on the predicted probability.
    """)

    st.warning(
        "The current dataset contains one year of observations (2019), so the "
        "model should be presented as a prototype/academic early-warning model. "
        "Additional years and independently validated surveillance data would be "
        "needed before operational public-health deployment."
    )
