# app.py - Full HR Analytics Dashboard (Dashboard + ML + Upload + preloaded.json)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import plotly.express as px

# -------- Configuration --------
st.set_page_config(page_title="HR Analytics Dashboard", layout="wide", page_icon="📊")
st.title("📊 HR Analytics Dashboard — Full (INR salaries)")

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

    # Gender counts
    if "Gender" in filtered_df.columns:
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
        sns.countplot(data=filtered_df, x="Gender", order=filtered_df["Gender"].value_counts().index, ax=ax)
        ax.set_title("Gender Counts")
        st.pyplot(fig)
        plt.close(fig)

    # Department counts
    if "Department" in filtered_df.columns:
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        sns.countplot(data=filtered_df, x="Department", order=filtered_df["Department"].value_counts().index, ax=ax)
        ax.set_title("Employees by Department")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)

    # Salary distribution (INR)
    if "SalaryINR" in filtered_df.columns and filtered_df["SalaryINR"].notna().any():
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        sns.histplot(filtered_df["SalaryINR"].dropna(), kde=True, ax=ax)
        ax.set_title("Salary Distribution (INR)")
        st.pyplot(fig)
        plt.close(fig)

    # Attrition by department
    if "Attrition" in filtered_df.columns:
        promo_by_dept = filtered_df.groupby("Department")["Attrition"].mean().reset_index()
        try:
            fig_px = px.bar(promo_by_dept, x="Department", y="Attrition", title="Attrition Rate by Department")
            st.plotly_chart(fig_px, width="stretch")
        except Exception:
            fig = plt.figure(figsize=(6, 3))
            ax = fig.add_subplot(111)
            sns.barplot(data=promo_by_dept, x="Department", y="Attrition", ax=ax)
            ax.set_title("Attrition Rate by Department")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)

# -------- ML Model: Attrition --------
with tab_model:
    st.subheader("AI: Attrition Prediction (Logistic Regression)")

    recommended_features = ["Age", "ExperienceYears", "PerformanceScore", "SalaryINR", "Department", "JobRole", "Gender"]
    available_features = [f for f in recommended_features if f in filtered_df.columns]
    st.write("Features available for model:", available_features)
    sel_features = st.multiselect("Select features to use for model", options=available_features, default=available_features)

    if "Attrition" not in filtered_df.columns:
        st.error("No 'Attrition' column found — cannot train model.")
    else:
        unique_classes = filtered_df["Attrition"].dropna().unique()
        if len(unique_classes) < 2:
            st.error(f"Target column 'Attrition' has only one class: {unique_classes}. Provide a dataset with both 0 and 1.")
        elif not sel_features:
            st.warning("Select at least one feature to train the model.")
        else:
            model_df = filtered_df[sel_features + ["Attrition"]].dropna()
            if len(model_df) < 50:
                st.warning("Dataset is small; model results may be unreliable. Prefer >50 rows.")
            X = model_df[sel_features].copy()
            y = model_df["Attrition"].astype(int)

            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

            # coerce numeric columns if encoded as strings
            for col in numeric_cols:
                X[col] = pd.to_numeric(X[col], errors="coerce")

            if X.shape[1] == 0:
                st.error("No usable features found after preprocessing.")
            else:
                # Train-test split
                strat = y if y.nunique() > 1 else None
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=strat)
                except Exception:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

                # ColumnTransformer and Pipeline (OneHotEncoder with sparse_output=False)
                preprocessor = ColumnTransformer(
                    transformers=[
                        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
                    ],
                    remainder="passthrough"
                )

                pipeline = Pipeline(steps=[
                    ("preproc", preprocessor),
                    ("clf", LogisticRegression(max_iter=5000))
                ])

                try:
                    pipeline.fit(X_train, y_train)
                except Exception as e:
                    st.error(f"Model training failed: {e}")
                else:
                    y_pred = pipeline.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"Model trained — accuracy on holdout: {acc*100:.2f}%")

                    # Interactive prediction UI
                    st.markdown("### Try a prediction")
                    input_vals = {}
                    for feat in sel_features:
                        if feat in numeric_cols:
                            col_min = int(float(X[feat].min())) if not pd.isna(X[feat].min()) else 0
                            col_max = int(float(X[feat].max())) if not pd.isna(X[feat].max()) else col_min + 1
                            col_median = int(float(X[feat].median())) if not pd.isna(X[feat].median()) else col_min
                            if col_min == col_max:
                                col_min = max(0, col_min - 1)
                                col_max = col_max + 1
                            input_vals[feat] = st.number_input(feat, value=col_median, min_value=col_min, max_value=col_max)
                        else:
                            opts = sorted(X[feat].dropna().unique().tolist())
                            if not opts:
                                opts = ["Unknown"]
                            input_vals[feat] = st.selectbox(feat, options=opts, index=0)

                    if st.button("Predict Attrition"):
                        single_df = pd.DataFrame([input_vals])
                        for nc in numeric_cols:
                            if nc in single_df.columns:
                                single_df[nc] = pd.to_numeric(single_df[nc], errors="coerce")
                        try:
                            pred = pipeline.predict(single_df)[0]
                            proba = pipeline.predict_proba(single_df)[0] if hasattr(pipeline, "predict_proba") else None
                            st.write(f"Prediction: **{pred}**  (0 = stay, 1 = leave)")
                            if proba is not None:
                                st.write(f"Probabilities: stay={proba[0]:.2f}, leave={proba[1]:.2f}")
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")

# -------- Heatmap --------
with tab_heatmap:
    st.subheader("Correlation Heatmap (numeric columns only)")
    num_df = filtered_df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    num_df = num_df.drop(columns=["Attrition"], errors="ignore")

    if num_df.shape[1] <= 1:
        st.warning("Not enough numeric columns for a correlation heatmap.")
    else:
        corr = num_df.corr()
        if corr.isnull().all().all():
            st.warning("Correlation matrix contains only NaN values — not enough numeric data.")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
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
