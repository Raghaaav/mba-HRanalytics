# app.py - Full HR Analytics Dashboard (Dashboard + ML + Upload + preloaded.json)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError

import plotly.express as px

# -------- Configuration --------
st.set_page_config(page_title="HR Analytics Dashboard", layout="wide", page_icon="📊")
st.title("📊 LIBA HR Analytics Dashboard")

# -------- Helpers & Sample Data (Indian salary bands, SalaryINR) --------
@st.cache_data
def generate_sample_data(n=500, seed=42):
    np.random.seed(seed)
    departments = ["Sales", "HR", "IT", "Finance", "Operations"]
    jobroles = [
        "Sales Executive", "Sales Manager", "HR Executive", "HR Manager",
        "Software Engineer", "Data Analyst", "System Administrator",
        "Finance Analyst", "Operations Executive", "Operations Manager"
    ]
    genders = ["Male", "Female"]

    # realistic INR salary ranges (annual, in rupees)
    def sample_salary(dept, role):
        # values in INR per year
        if dept == "IT":
            if "Engineer" in role or "Developer" in role or "Analyst" in role:
                return int(np.random.normal(900000, 200000))  # avg 9 LPA
            else:
                return int(np.random.normal(700000, 150000))
        if dept == "HR":
            if "Manager" in role:
                return int(np.random.normal(1000000, 150000))
            return int(np.random.normal(500000, 100000))
        if dept == "Sales":
            if "Manager" in role:
                return int(np.random.normal(900000, 200000))
            return int(np.random.normal(450000, 120000))
        if dept == "Finance":
            if "Manager" in role:
                return int(np.random.normal(1200000, 250000))
            return int(np.random.normal(700000, 150000))
        if dept == "Operations":
            if "Manager" in role:
                return int(np.random.normal(900000, 200000))
            return int(np.random.normal(550000, 120000))
        return int(np.random.normal(600000, 150000))

    rows = []
    for i in range(1, n + 1):
        dept = np.random.choice(departments)
        role = np.random.choice(jobroles)
        # control gender distribution roughly 56% male, 44% female
        gender = np.random.choice(genders, p=[0.56, 0.44])
        salary = max(150000, sample_salary(dept, role))
        row = {
            "EmployeeID": i,
            "Name": f"Employee {i}",
            "username": f"user{i}",
            "email": f"user{i}@example.com",
            "phone": f"9{np.random.randint(600000000,999999999)}",
            "website": f"employee{i}.in",
            "Age": int(np.random.randint(22, 58)),
            "Gender": gender,
            "Department": dept,
            "JobRole": role,
            "SalaryINR": int(salary),
            "ExperienceYears": int(np.random.randint(0, min(35, max(1, salary//100000)))) ,
            "PerformanceScore": int(np.random.randint(1, 6)),
            "Attrition": int(np.random.choice([0,1], p=[0.7,0.3]))
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

SAMPLE_DF = generate_sample_data(500)

# -------- Sidebar: data source and sample download --------
st.sidebar.header("Data options")
data_source = st.sidebar.radio("Choose data source:", ["Sample (500 rows)", "Preloaded (local JSON)", "Upload CSV"])

# provide downloads for sample data
st.sidebar.download_button("Download sample CSV (500 rows)", SAMPLE_DF.to_csv(index=False), file_name="hr_sample_500.csv", mime="text/csv")
st.sidebar.download_button("Download sample JSON (100 rows)", SAMPLE_DF.head(100).to_json(orient='records'), file_name="preloaded.json", mime="application/json")

# -------- Load preloaded.json from local file --------
@st.cache_data
def load_preloaded_data(file_path="preloaded.json"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        df = pd.DataFrame(raw)
        # drop nested dict columns if any
        nested_cols = [c for c in df.columns if df[c].apply(lambda x: isinstance(x, dict)).any()]
        if nested_cols:
            df = df.drop(columns=nested_cols, errors='ignore')
        return df
    except FileNotFoundError:
        st.warning("preloaded.json not found; using generated sample subset.")
        return SAMPLE_DF.head(100)
    except Exception as e:
        st.error(f"Failed to load preloaded.json: {e}")
        return SAMPLE_DF.head(100)

# -------- Load data based on selection --------
uploaded_file = None
if data_source == "Sample (500 rows)":
    df = SAMPLE_DF.copy()
elif data_source == "Preloaded (local JSON)":
    df = load_preloaded_data()
else:
    uploaded_file = st.sidebar.file_uploader("Upload your HR CSV file", type=["csv", "json"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith('.json'):
                raw = json.load(uploaded_file)
                df = pd.DataFrame(raw)
            else:
                df = pd.read_csv(uploaded_file)
            st.success("Uploaded dataset loaded successfully!")
        except Exception as e:
            st.error("Unable to read uploaded file — please upload a valid CSV or JSON.")
            st.stop()
    else:
        st.info("No file uploaded. Using sample dataset for preview.")
        df = SAMPLE_DF.copy()

# -------- Column detection & canonicalization (keep SalaryINR as main) --------
original_cols = df.columns.tolist()
cols_lower = {c.lower(): c for c in original_cols}

def pick_col(possible_names):
    for name in possible_names:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return None

col_emp = pick_col(["employeeid", "employee_id", "id"])
col_name = pick_col(["name"])
col_username = pick_col(["username", "user"])
col_email = pick_col(["email"])
col_phone = pick_col(["phone", "contact"])
col_website = pick_col(["website", "site"])
col_age = pick_col(["age"])
col_gender = pick_col(["gender", "sex"])
col_dept = pick_col(["department", "dept"])
col_job = pick_col(["jobrole", "job_role", "role"])
col_salary_inr = pick_col(["salaryinr", "salary_inr", "salary", "ctc", "annualsalary"])
col_exp = pick_col(["experienceyears", "experience_years", "experience", "length_of_service"])
col_perf = pick_col(["performancescore", "performance_score", "rating", "previous_year_rating"])
col_attr = pick_col(["attrition", "promoted", "left", "is_left"])

rename_map = {}
if col_emp: rename_map[col_emp] = "EmployeeID"
if col_name: rename_map[col_name] = "Name"
if col_username: rename_map[col_username] = "username"
if col_email: rename_map[col_email] = "email"
if col_phone: rename_map[col_phone] = "phone"
if col_website: rename_map[col_website] = "website"
if col_age: rename_map[col_age] = "Age"
if col_gender: rename_map[col_gender] = "Gender"
if col_dept: rename_map[col_dept] = "Department"
if col_job: rename_map[col_job] = "JobRole"
if col_salary_inr: rename_map[col_salary_inr] = "SalaryINR"
if col_exp: rename_map[col_exp] = "ExperienceYears"
if col_perf: rename_map[col_perf] = "PerformanceScore"
if col_attr: rename_map[col_attr] = "Attrition"

df = df.rename(columns=rename_map)

# -------- Ensure required columns exist and sanitize numeric data --------
# Clean salary-like strings (remove currency symbols, commas, text)
def sanitize_numeric_col(series):
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

# SalaryINR priority: use SalaryINR if present; else create synthetic INR salary
if "SalaryINR" in df.columns:
    df["SalaryINR"] = sanitize_numeric_col(df["SalaryINR"]).fillna(np.nan)
else:
    # if SalaryUSD present, convert (edge-case) — but primary assumption: INR provided
    col_usd = pick_col(["salaryusd", "salary_usd", "salary_usd_inr"])
    if col_usd:
        df["SalaryINR"] = sanitize_numeric_col(df[col_usd]) * 83
    else:
        # generate synthetic INR salary with broad realistic range
        df["SalaryINR"] = np.random.randint(300000, 1500000, size=len(df))

# Numeric coercion for other numeric fields
for col in ["Age", "ExperienceYears", "PerformanceScore"]:
    if col in df.columns:
        df[col] = sanitize_numeric_col(df[col])

# Clip unrealistic experience and fill missing sensible defaults
if "ExperienceYears" in df.columns:
    df["ExperienceYears"] = df["ExperienceYears"].clip(lower=0, upper=50).fillna(0)
if "Age" in df.columns:
    df["Age"] = df["Age"].clip(lower=16, upper=100).fillna(int(df["Age"].median() if pd.notna(df["Age"].median()) else 30))

# Categorical defaults
if "Gender" not in df.columns:
    df["Gender"] = np.random.choice(["Male", "Female"], len(df))
if "Department" not in df.columns:
    df["Department"] = np.random.choice(["Sales", "HR", "IT", "Finance", "Operations"], len(df))
if "JobRole" not in df.columns:
    df["JobRole"] = np.random.choice(["Executive", "Manager", "Analyst", "Engineer"], len(df))
if "PerformanceScore" not in df.columns:
    df["PerformanceScore"] = np.random.randint(1, 6, len(df))
if "Attrition" not in df.columns:
    df["Attrition"] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
if "EmployeeID" not in df.columns:
    df["EmployeeID"] = np.arange(1, len(df) + 1)

# Convert Attrition to int if possible
try:
    df["Attrition"] = pd.to_numeric(df["Attrition"], errors="coerce").fillna(0).astype(int)
except Exception:
    df["Attrition"] = df["Attrition"].astype(str).map(lambda x: 1 if str(x).strip().lower() in ["1","true","yes","y"] else 0)

# -------- Sidebar Filters (persist across reruns) --------
st.sidebar.header("Filters")
dept_options = sorted(df["Department"].dropna().unique().tolist())
sel_departments = st.sidebar.multiselect("Departments", options=dept_options, default=dept_options)

gender_options = ["All"] + sorted(df["Gender"].dropna().unique().tolist())
sel_gender = st.sidebar.selectbox("Gender", options=gender_options, index=0)

min_age = int(np.nanmin(df["Age"]))
max_age = int(np.nanmax(df["Age"]))
sel_age_range = st.sidebar.slider("Age range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

# Apply filters
filtered_df = df.copy()
if sel_departments:
    filtered_df = filtered_df[filtered_df["Department"].isin(sel_departments)]
if sel_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == sel_gender]
filtered_df = filtered_df[(filtered_df["Age"] >= sel_age_range[0]) & (filtered_df["Age"] <= sel_age_range[1])]

# -------- Tabs --------
tab_overview, tab_viz, tab_model, tab_heatmap, tab_download = st.tabs([
    "Overview", "Visualizations", "AI Predictions", "Correlation Heatmap", "Download Data"
])

# -------- Overview --------
with tab_overview:
    st.subheader("Key HR KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Employees", f"{len(filtered_df):,}")
    avg_salary = filtered_df["SalaryINR"].dropna().mean()
    c2.metric("Avg Salary (INR)", f"₹{avg_salary:,.0f}" if not np.isnan(avg_salary) else "N/A")
    if "Attrition" in filtered_df.columns:
        c3.metric("Attrition Rate", f"{filtered_df['Attrition'].mean() * 100:.1f}%")
    else:
        c3.metric("Attrition Rate", "N/A")
    c4.metric("Avg Experience (yrs)", f"{filtered_df['ExperienceYears'].mean():.1f}")

    st.markdown("#### Data sample")
    st.dataframe(filtered_df.head(10), width="stretch")

# -------- Visualizations --------
with tab_viz:
    st.subheader("Visual Insights")

    # ---------- Color palettes ----------
    gender_colors = {"Male": "#1f77b4", "Female": "#ff7f0e"}  # Blue & Orange
    dept_colors = {
        "Sales": "#636EFA",
        "HR": "#EF553B",
        "IT": "#00CC96",
        "Finance": "#AB63FA",
        "Operations": "#FFA15A"
    }

    # ---------- 1. Gender Distribution Pie Chart (3D-like donut) ----------
    if "Gender" in filtered_df.columns:
        gender_counts = filtered_df["Gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]

        fig_gender = px.pie(
            gender_counts,
            names="Gender",
            values="Count",
            title="Gender Distribution",
            color="Gender",
            color_discrete_map=gender_colors,
            hole=0.4  # donut style
        )
        fig_gender.update_traces(marker=dict(line=dict(width=0)))  # remove black outline
        fig_gender.update_layout(width=700, height=500)
        st.plotly_chart(fig_gender, width='stretch')

    # ---------- 2. Department Distribution Line Chart ----------
    if "Department" in filtered_df.columns:
        dept_counts = filtered_df["Department"].value_counts().reset_index()
        dept_counts.columns = ["Department", "Count"]

        fig_dept = px.line(
            dept_counts.sort_values("Department"),
            x="Department",
            y="Count",
            title="Employees by Department (Trend)",
            markers=True,
            color_discrete_sequence=["#636EFA"]
        )
        fig_dept.update_layout(width=800, height=450, yaxis_title="Number of Employees")
        st.plotly_chart(fig_dept, width='stretch')

    # ---------- 3. Salary Distribution by Department (Violin) ----------
    if "SalaryINR" in filtered_df.columns:
        fig_salary = px.violin(
            filtered_df,
            y="SalaryINR",
            x="Department",
            color="Department",
            box=True,
            points="all",
            title="Salary Distribution by Department (INR)",
            color_discrete_map=dept_colors
        )
        fig_salary.update_layout(width=900, height=500, yaxis_title="Salary (INR)", xaxis_title="Department")
        st.plotly_chart(fig_salary, width='stretch')

    # ---------- 4. Attrition Rate by Department (Line) ----------
    if "Attrition" in filtered_df.columns:
        attr_dept = filtered_df.groupby("Department")["Attrition"].mean().reset_index()
        attr_dept["AttritionRate"] = attr_dept["Attrition"] * 100

        fig_attr = px.line(
            attr_dept.sort_values("Department"),
            x="Department",
            y="AttritionRate",
            markers=True,
            title="Attrition Rate by Department (%)",
            color_discrete_sequence=["#EF553B"]
        )
        fig_attr.update_layout(width=800, height=450, yaxis_title="Attrition Rate (%)")
        st.plotly_chart(fig_attr, width='stretch')

    # ---------- 5. Average Experience by Department (Line) ----------
    if "ExperienceYears" in filtered_df.columns:
        exp_dept = filtered_df.groupby("Department")["ExperienceYears"].mean().reset_index()
        fig_exp = px.line(
            exp_dept.sort_values("Department"),
            x="Department",
            y="ExperienceYears",
            markers=True,
            title="Average Experience by Department (years)",
            color_discrete_sequence=["#00CC96"]
        )
        fig_exp.update_layout(width=800, height=450, yaxis_title="Average Experience (yrs)")
        st.plotly_chart(fig_exp, width='stretch')

    # ---------- 6. Gender Distribution within Departments (Stacked) ----------
    if "Gender" in filtered_df.columns and "Department" in filtered_df.columns:
        import itertools
        departments = filtered_df["Department"].unique()
        genders = ["Male", "Female"]
        all_combos = pd.DataFrame(list(itertools.product(departments, genders)), columns=["Department", "Gender"])
        dept_gender = filtered_df.groupby(["Department", "Gender"]).size().reset_index(name="Count")
        dept_gender = all_combos.merge(dept_gender, on=["Department", "Gender"], how="left").fillna(0)

        fig_stacked = px.bar(
            dept_gender,
            x="Department",
            y="Count",
            color="Gender",
            title="Gender Distribution within Departments",
            text="Count",
            color_discrete_map=gender_colors
        )
        fig_stacked.update_layout(width=900, height=500, yaxis_title="Number of Employees", xaxis_title="", barmode='stack')
        fig_stacked.update_traces(textposition='inside')
        st.plotly_chart(fig_stacked, width='stretch')


    # ---------- 8. Salary vs Performance Scatter ----------
    if all(col in filtered_df.columns for col in ["SalaryINR", "PerformanceScore"]):
        fig_scatter = px.scatter(
            filtered_df,
            x="PerformanceScore",
            y="SalaryINR",
            color="Department",
            hover_data=["EmployeeID", "JobRole", "Gender"],
            title="Salary vs Performance Score",
            color_discrete_map=dept_colors
        )
        fig_scatter.update_layout(width=900, height=500, xaxis_title="Performance Score", yaxis_title="Salary (INR)")
        st.plotly_chart(fig_scatter, width='stretch')

    # ---------- 9. Attrition vs Age (Scatter & Box Plot) ----------
    if "Age" in filtered_df.columns and "Attrition" in filtered_df.columns:
        fig_age = px.box(
            filtered_df,
            x="Attrition",
            y="Age",
            color="Attrition",
            title="Attrition vs Age",
            labels={"Attrition": "Attrition (0=Stay, 1=Leave)", "Age": "Age"},
            color_discrete_map={0: "#1f77b4", 1: "#ff7f0e"}
        )
        fig_age.update_layout(width=800, height=500)
        st.plotly_chart(fig_age, width='stretch')

    # ---------- 10. Correlation Heatmap (numeric columns) ----------
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        corr = filtered_df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap (numeric columns)"
        )
        fig_corr.update_layout(width=900, height=500)
        st.plotly_chart(fig_corr, width='stretch')

    # ---------- 11. Top 5 JobRoles by Attrition Rate ----------
    if "JobRole" in filtered_df.columns and "Attrition" in filtered_df.columns:
        job_attr = filtered_df.groupby("JobRole")["Attrition"].mean().sort_values(ascending=False).reset_index()
        top_jobs = job_attr.head(5)
        fig_top_jobs = px.bar(
            top_jobs,
            x="JobRole",
            y="Attrition",
            text=top_jobs["Attrition"].round(2),
            title="Top 5 Job Roles by Attrition Rate",
            color="JobRole",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_top_jobs.update_traces(textposition='outside')
        fig_top_jobs.update_layout(width=800, height=500, yaxis_title="Attrition Rate")
        st.plotly_chart(fig_top_jobs, width='stretch')

    # ---------- 12. Salary Distribution by JobRole ----------
    if "SalaryINR" in filtered_df.columns and "JobRole" in filtered_df.columns:
        fig_salary_role = px.violin(
            filtered_df,
            y="SalaryINR",
            x="JobRole",
            color="JobRole",
            box=True,
            points="all",
            title="Salary Distribution by Job Role (INR)"
        )
        fig_salary_role.update_layout(width=900, height=500, yaxis_title="Salary (INR)", xaxis_title="Job Role")
        st.plotly_chart(fig_salary_role, width='stretch')

# -------- AI Predictions Suite --------

with tab_model:
    st.header("🤖 AI Predictions (Attrition / Salary / Performance / Promotion / Absenteeism)")

    # ----- Define per-model features (only use columns we expect in this app) -----
    FEATURES = {
        "Attrition": ["Age", "ExperienceYears", "PerformanceScore", "SalaryINR", "Department", "JobRole", "Gender"],
        "Salary Prediction (INR)": ["Age", "ExperienceYears", "PerformanceScore", "Department", "JobRole", "Gender"],
        "Performance Score Prediction": ["Age", "ExperienceYears", "SalaryINR", "Department", "JobRole", "Gender"],
        "Promotion Eligibility": ["ExperienceYears", "PerformanceScore", "SalaryINR", "Department", "JobRole"],
        "Absenteeism Risk": ["Age", "ExperienceYears", "Department", "JobRole", "PerformanceScore"]
    }

    # ----- Models chosen per task -----
    MODELS = {
        "Attrition": RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=42),
        "Salary Prediction (INR)": RandomForestRegressor(n_estimators=400, max_depth=15, random_state=42),
        "Performance Score Prediction": GradientBoostingRegressor(n_estimators=300, random_state=42),
        "Promotion Eligibility": RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=42),
        "Absenteeism Risk": RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=42)
    }

    st.markdown("### Choose prediction")
    prediction_choice = st.selectbox("Prediction type:", list(FEATURES.keys()))

    selected_features = [c for c in FEATURES[prediction_choice] if c in df.columns]
    missing = [c for c in FEATURES[prediction_choice] if c not in df.columns]
    if missing:
        st.warning(f"Note: these expected columns are missing from the dataset and will not be used: {missing}")

    st.markdown("### Enter input values")
    # dynamic inputs: show appropriate input controls based on feature type/name
    user_input = {}
    cols_left, cols_right = st.columns(2)
    for i, feat in enumerate(selected_features):
        container = cols_left if (i % 2 == 0) else cols_right
        with container:
            if feat in ["Department", "JobRole", "Gender"]:
                opts = sorted(df[feat].dropna().unique().tolist())
                if not opts:
                    opts = ["Unknown"]
                user_input[feat] = st.selectbox(feat, opts, index=0)
            else:
                # numeric -- provide a sensible range based on dataset
                col_series = pd.to_numeric(df[feat], errors="coerce")
                col_min = int(np.nanmin(col_series)) if pd.notna(col_series.min()) else 0
                col_max = int(np.nanmax(col_series)) if pd.notna(col_series.max()) else col_min + 10
                median = int(col_series.median()) if pd.notna(col_series.median()) else (col_min + col_max) // 2
                # make bounds slightly wider so users can explore
                min_safe = max(0, col_min - int(abs(col_min)*0.2 + 1))
                max_safe = int(col_max + max(10, abs(col_max)*0.2))
                # pick slider for reasonable range, number_input if range huge
                if feat == "SalaryINR":
                    user_input[feat] = st.number_input(feat, min_value=0, value=median, step=1000)
                else:
                    # use slider when range small
                    if max_safe - min_safe <= 1000:
                        user_input[feat] = st.slider(feat, min_value=min_safe, max_value=max_safe, value=median)
                    else:
                        user_input[feat] = st.number_input(feat, value=median, min_value=min_safe, max_value=max_safe)

    # Push missing features with defaults (if any required numeric missing from df, fill reasonable default)
    for feat in selected_features:
        if feat not in user_input:
            # fallback defaults
            if feat in ["Department", "JobRole", "Gender"]:
                user_input[feat] = df[feat].dropna().unique()[0] if feat in df.columns and len(df[feat].dropna())>0 else "Unknown"
            else:
                user_input[feat] = int(df[feat].dropna().median()) if feat in df.columns and df[feat].dropna().shape[0]>0 else 0

    st.markdown("---")

    # Prepare input DataFrame
    input_df = pd.DataFrame([user_input])[selected_features]  # ensure column order

    # Build preprocessor (determine numeric vs categorical from selected_features)
    numeric_cols = [c for c in selected_features if np.issubdtype(df[c].dtype, np.number)]
    categorical_cols = [c for c in selected_features if c not in numeric_cols]

    # use StandardScaler for numeric columns (works for both classification/regression here)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols) if numeric_cols else ("num", "passthrough", []),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols) if categorical_cols else ("cat", "passthrough", [])
        ],
        remainder="drop"
    )

    # model pipeline
    model_obj = MODELS[prediction_choice]
    pipeline = Pipeline([("pre", preprocessor), ("model", model_obj)])

    # target column mapping for training
    # prefer real columns if present, otherwise cannot train
    target_map = {
        "Attrition": "Attrition",
        "Salary Prediction (INR)": "SalaryINR",
        "Performance Score Prediction": "PerformanceScore",
        "Promotion Eligibility": "Attrition",       # proxy: treat recent attrition/promotion as label if no explicit promoted column
        "Absenteeism Risk": "Attrition"            # proxy (use attrition as risk-ish target) — replace when real target available
    }
    target_col = target_map[prediction_choice]
    model_filename = "prediction_models/" + f"model_{prediction_choice.replace(' ', '_').lower()}.pkl"

    can_train = (target_col in df.columns) and (df[target_col].dropna().shape[0] > 10)

    # Train/load model
    trained_model = None
    if can_train:
        # Prepare training data X_train, y_train (dropna on required cols + target)
        train_df = df[selected_features + [target_col]].dropna()
        X_all = train_df[selected_features]
        y_all = train_df[target_col].astype(float) if np.issubdtype(train_df[target_col].dtype, np.number) else train_df[target_col]

        # For classification tasks, ensure at least 2 classes
        is_classification = prediction_choice in ["Attrition", "Promotion Eligibility", "Absenteeism Risk"]
        if is_classification:
            if y_all.nunique() < 2:
                st.warning(f"Target '{target_col}' does not have at least 2 classes; training skipped.")
                can_train = False

    # Try to load saved model first
    if can_train:
        try:
            trained_model = pickle.load(open(model_filename, "rb"))
            st.info(f"Loaded saved model: {model_filename}")
        except Exception:
            st.info("No saved model found — training a new model (this may take a few seconds).")
            try:
                # For classification, stratify split where possible
                if is_classification:
                    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25, random_state=42, stratify=y_all)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25, random_state=42)

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                # Save
                pickle.dump(pipeline, open(model_filename, "wb"))
                trained_model = pipeline
                # Show simple metric for classification
                if is_classification:
                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"Model trained. Holdout accuracy: {acc*100:.2f}%")
                else:
                    st.success("Regression model trained and saved.")
            except Exception as e:
                st.error(f"Training failed: {e}")
                can_train = False

    else:
        st.warning(f"Cannot train model — target column '{target_col}' not available or insufficient data. Prediction will attempt using saved model if present.")

        # try load model even if can't train
        try:
            trained_model = pickle.load(open(model_filename, "rb"))
            st.info(f"Loaded saved model: {model_filename} (no retrain)")
        except Exception:
            st.info("No pre-trained model available.")

    st.markdown("---")

    # Prediction action
    if st.button("Run Prediction"):
        if trained_model is None:
            st.error("No trained model available to make a prediction. Provide target column in dataset or place a pre-trained model file.")
        else:
            try:
                pred = trained_model.predict(input_df)
                result = pred[0]
                # classification: show probabilities if available
                if prediction_choice in ["Attrition", "Promotion Eligibility", "Absenteeism Risk"]:
                    # probabilities
                    proba = None
                    try:
                        proba = trained_model.predict_proba(input_df)[0]
                    except Exception:
                        proba = None
                    st.write("### Prediction result")
                    st.write(f"**{prediction_choice}** → **{result}**")
                    if proba is not None:
                        # for binary: show prob of class 1 if two classes
                        if len(proba) == 2:
                            st.write(f"Probability of positive (class 1): {proba[1]*100:.2f}%")
                        else:
                            st.write("Class probabilities:", {str(i): f"{p*100:.2f}%" for i, p in enumerate(proba)})
                else:
                    # regression result
                    if prediction_choice == "Salary Prediction (INR)":
                        st.write("### Estimated Salary (INR)")
                        st.metric("Estimated Salary", f"₹{int(result):,}")
                    else:
                        st.write("### Prediction result")
                        st.write(result)
            except NotFittedError:
                st.error("Model is not fitted. Try training a model first.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")


# -------- Heatmap --------
with tab_heatmap:
    st.subheader("HR Metrics Correlation Heatmap")

    # Select relevant numeric columns including Attrition
    num_df = filtered_df.select_dtypes(include=[np.number]).copy()
    if num_df.shape[1] <= 1:
        st.warning("Not enough numeric columns for a correlation heatmap.")
    else:
        corr = num_df.corr()
        if corr.isnull().all().all():
            st.warning("Correlation matrix contains only NaN values — not enough numeric data.")
        else:
            # Mask upper triangle for cleaner display
            mask = np.triu(np.ones_like(corr, dtype=bool))

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                corr,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                linewidths=1,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 10}
            )
            ax.set_title("Correlation Heatmap: HR Metrics", fontsize=14)
            st.pyplot(fig)
            plt.close(fig)


# -------- Download --------
with tab_download:
    st.subheader("Download filtered data")
    buffered = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV of filtered data", buffered, file_name="filtered_hr_data.csv", mime="text/csv")

# Debugging info (optional)
with st.expander("Show detected/renamed columns (debug)"):
    st.write("Renamed columns mapping:", rename_map)
    st.write("Current dataframe columns:", filtered_df.columns.tolist())
