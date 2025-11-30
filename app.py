# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="HR Analytics Dashboard", layout="wide", page_icon="📊")
st.title("📊 HR Analytics Dashboard — Fixed & Improved")

# -------------------------
# Helper: generate sample dataset (500 rows)
# -------------------------
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
    df = pd.DataFrame({
        "EmployeeID": np.arange(1, n + 1),
        "Age": np.random.randint(21, 60, n),
        "Gender": np.random.choice(genders, n),
        "Department": np.random.choice(departments, n),
        "JobRole": np.random.choice(jobroles, n),
        "Education": np.random.randint(1, 6, n),
        "SalaryUSD": np.random.randint(30000, 120000, n),  # USD base
        "ExperienceYears": np.random.randint(0, 36, n),
        "PerformanceScore": np.random.randint(1, 6, n),
        # Balanced Attrition ~ roughly 30% positive
        "Attrition": np.random.choice([0, 1], n, p=[0.7, 0.3])
    })
    # Ensure variety: some high earners, some low
    return df

# -------------------------
# Sidebar: Data source
# -------------------------
st.sidebar.header("Data options")
data_source = st.sidebar.radio("Choose data source:", ["Sample (500 rows)", "Preloaded API (demo)", "Upload CSV"])

# Download sample dataset button
sample_df = generate_sample_data(500)
csv_sample = sample_df.to_csv(index=False)
st.sidebar.download_button("Download sample CSV (500 rows)", csv_sample, file_name="hr_sample_500.csv", mime="text/csv")

# -------------------------
# Load data based on selection
# -------------------------
def load_api_demo():
    # use a small demo JSON but avoid verify=False to prevent warnings
    url = "https://jsonplaceholder.typicode.com/users"
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            df = pd.DataFrame(r.json())
            # create some demo HR columns so visuals work
            n = len(df)
            df["EmployeeID"] = np.arange(1, n + 1)
            df["Age"] = np.random.randint(25, 55, n)
            df["Gender"] = np.random.choice(["Male", "Female"], n)
            df["Department"] = np.random.choice(["Sales","HR","IT","Finance","Operations"], n)
            df["JobRole"] = np.random.choice(["Executive","Manager","Analyst","Engineer"], n)
            df["SalaryUSD"] = np.random.randint(30000, 120000, n)
            df["ExperienceYears"] = np.random.randint(0, 30, n)
            df["PerformanceScore"] = np.random.randint(1,6, n)
            df["Attrition"] = np.random.choice([0,1], n, p=[0.7,0.3])
            return df
        else:
            st.warning("API demo failed — using sample data instead.")
            return generate_sample_data(50)
    except Exception as e:
        st.warning("Could not fetch demo API data — using sample data instead.")
        return generate_sample_data(50)

uploaded_file = None
if data_source == "Sample (500 rows)":
    df = sample_df.copy()
elif data_source == "Preloaded API (demo)":
    df = load_api_demo()
else:
    uploaded_file = st.sidebar.file_uploader("Upload your HR CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Uploaded dataset loaded successfully!")
        except Exception as e:
            st.error("Unable to read CSV — please upload a valid CSV file.")
            st.stop()
    else:
        st.info("No file uploaded. Using sample dataset for preview.")
        df = sample_df.copy()

# -------------------------
# Ensure essential columns exist and standardize names
# -------------------------
# Accept many possible names by lowercasing column names for detection
original_cols = df.columns.tolist()
cols_lower = {c.lower(): c for c in original_cols}

def pick_col(possible_names):
    for name in possible_names:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return None

# column detection
col_emp = pick_col(["employeeid", "employee_id", "id"])
col_age = pick_col(["age"])
col_gender = pick_col(["gender", "sex"])
col_dept = pick_col(["department", "dept"])
col_job = pick_col(["jobrole", "job_role", "role"])
col_salary_usd = pick_col(["salaryusd", "salary_usd", "salary"])
col_salary_inr = pick_col(["salaryinr", "salary_inr"])
col_exp = pick_col(["experienceyears", "experience_years", "experience", "length_of_service"])
col_perf = pick_col(["performancescore", "performance_score", "rating", "previous_year_rating"])
col_attr = pick_col(["attrition", "promoted", "left"])

# Rename detected columns to canonical names for app convenience
rename_map = {}
if col_emp: rename_map[col_emp] = "EmployeeID"
if col_age: rename_map[col_age] = "Age"
if col_gender: rename_map[col_gender] = "Gender"
if col_dept: rename_map[col_dept] = "Department"
if col_job: rename_map[col_job] = "JobRole"
if col_salary_usd: rename_map[col_salary_usd] = "SalaryUSD"
if col_salary_inr: rename_map[col_salary_inr] = "SalaryINR"
if col_exp: rename_map[col_exp] = "ExperienceYears"
if col_perf: rename_map[col_perf] = "PerformanceScore"
if col_attr: rename_map[col_attr] = "Attrition"

df = df.rename(columns=rename_map)

# -------------------------
# Create missing essential columns with reasonable defaults (to avoid crashes)
# -------------------------
# Salary handling:
# - If SalaryUSD exists and SalaryINR doesn't -> create SalaryINR using exchange rate
# - If SalaryINR exists and SalaryUSD doesn't -> keep as is
# - If neither exist -> create SalaryUSD synthetic and SalaryINR computed
USD_TO_INR = 83  # fixed conversion rate; change if you want live rates

if "SalaryINR" not in df.columns and "SalaryUSD" in df.columns:
    # convert
    try:
        df["SalaryINR"] = pd.to_numeric(df["SalaryUSD"], errors="coerce") * USD_TO_INR
    except Exception:
        df["SalaryINR"] = np.nan

if "SalaryUSD" not in df.columns and "SalaryINR" in df.columns:
    # compute USD back for completeness (approx)
    try:
        df["SalaryUSD"] = pd.to_numeric(df["SalaryINR"], errors="coerce") / USD_TO_INR
    except Exception:
        df["SalaryUSD"] = np.nan

if "SalaryINR" not in df.columns and "SalaryUSD" not in df.columns:
    st.warning("Salary column not found. Creating a synthetic SalaryUSD column for demo and converting to INR.")
    df["SalaryUSD"] = np.random.randint(30000, 120000, size=len(df))
    df["SalaryINR"] = df["SalaryUSD"] * USD_TO_INR

# ExperienceYears
if "ExperienceYears" not in df.columns:
    df["ExperienceYears"] = np.random.randint(0, 30, len(df))

# Age
if "Age" not in df.columns:
    df["Age"] = np.random.randint(21, 60, len(df))

# Gender
if "Gender" not in df.columns:
    df["Gender"] = np.random.choice(["Male", "Female"], len(df))

# Department
if "Department" not in df.columns:
    df["Department"] = np.random.choice(["Sales", "HR", "IT", "Finance", "Operations"], len(df))

# JobRole
if "JobRole" not in df.columns:
    df["JobRole"] = np.random.choice(["Executive", "Manager", "Analyst", "Engineer"], len(df))

# PerformanceScore
if "PerformanceScore" not in df.columns:
    df["PerformanceScore"] = np.random.randint(1, 6, len(df))

# Attrition target
if "Attrition" not in df.columns:
    # create a realistic attrition column (~30% attrition) so ML demo works
    df["Attrition"] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])

# EmployeeID
if "EmployeeID" not in df.columns:
    df["EmployeeID"] = np.arange(1, len(df) + 1)

# -------------------------
# Sidebar: Filters (placed outside nesting so they persist)
# -------------------------
st.sidebar.header("Filters")
dept_options = sorted(df["Department"].dropna().unique().tolist())
sel_departments = st.sidebar.multiselect("Departments", options=dept_options, default=dept_options)

gender_options = ["All"] + sorted(df["Gender"].dropna().unique().tolist())
sel_gender = st.sidebar.selectbox("Gender", options=gender_options, index=0)

min_age = int(df["Age"].min())
max_age = int(df["Age"].max())
sel_age_range = st.sidebar.slider("Age range", min_age, max_age, (min_age, max_age))

# Apply filters
filtered_df = df.copy()
if sel_departments:
    filtered_df = filtered_df[filtered_df["Department"].isin(sel_departments)]
if sel_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == sel_gender]
filtered_df = filtered_df[(filtered_df["Age"] >= sel_age_range[0]) & (filtered_df["Age"] <= sel_age_range[1])]

# -------------------------
# Main layout: Tabs
# -------------------------
tab_overview, tab_viz, tab_model, tab_heatmap, tab_download = st.tabs(
    ["Overview", "Visualizations", "AI Predictions", "Correlation Heatmap", "Download Data"]
)

# Overview
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
    st.dataframe(filtered_df.head(10))

# Visualizations
with tab_viz:
    st.subheader("Visual Insights")

    # Gender distribution
    if "Gender" in filtered_df.columns:
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
        sns.countplot(data=filtered_df, x="Gender", order=filtered_df["Gender"].value_counts().index, ax=ax)
        ax.set_title("Gender Counts")
        st.pyplot(fig)

    # Dept counts
    if "Department" in filtered_df.columns:
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        sns.countplot(data=filtered_df, x="Department", order=filtered_df["Department"].value_counts().index, ax=ax)
        ax.set_title("Employees by Department")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Salary distribution (INR)
    if "SalaryINR" in filtered_df.columns:
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        sns.histplot(filtered_df["SalaryINR"].dropna(), kde=True, ax=ax)
        ax.set_title("Salary Distribution (INR)")
        st.pyplot(fig)

    # Promotion/Attrition by department (mean)
    if "Attrition" in filtered_df.columns:
        promo_by_dept = filtered_df.groupby("Department")["Attrition"].mean().reset_index()
        fig = px := None
        try:
            import plotly.express as px
            fig = px.bar(promo_by_dept, x="Department", y="Attrition", title="Attrition Rate by Department")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            # fallback to matplotlib if plotly not available
            fig = plt.figure(figsize=(6, 3))
            ax = fig.add_subplot(111)
            sns.barplot(data=promo_by_dept, x="Department", y="Attrition", ax=ax)
            ax.set_title("Attrition Rate by Department")
            plt.xticks(rotation=45)
            st.pyplot(fig)

# Model tab: Attrition prediction
with tab_model:
    st.subheader("AI: Attrition Prediction (Logistic Regression)")

    # Select features to use (only numeric + encoded categorical)
    # Basic recommended features
    recommended_features = ["Age", "ExperienceYears", "PerformanceScore", "SalaryINR", "Department", "JobRole", "Gender"]

    # Check what features exist in the filtered_df
    available_features = [f for f in recommended_features if f in filtered_df.columns]
    st.write("Features available for model:", available_features)
    sel_features = st.multiselect("Select features to use for model", options=available_features, default=available_features)

    # Check target
    if "Attrition" not in filtered_df.columns:
        st.error("No 'Attrition' column found — cannot train model.")
    else:
        # Check number of classes in target (must be >=2)
        unique_classes = filtered_df["Attrition"].dropna().unique()
        if len(unique_classes) < 2:
            st.error(f"Target column 'Attrition' has only one class: {unique_classes}. Provide a dataset with both 0 and 1.")
        elif not sel_features:
            st.warning("Select at least one feature to train the model.")
        else:
            # Prepare X and y
            model_df = filtered_df[sel_features + ["Attrition"]].dropna()
            X = model_df[sel_features]
            y = model_df["Attrition"].astype(int)

            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

            # Build pipeline: one-hot categorical features, passthrough numeric
            categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

            preprocessor = ColumnTransformer(transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_cols)
            ], remainder="passthrough")  # passthrough numeric columns

            pipeline = Pipeline(steps=[
                ("preproc", preprocessor),
                ("clf", LogisticRegression(max_iter=500))
            ])

            # Fit model - wrap in try/except in case of issues
            try:
                pipeline.fit(X_train, y_train)
            except ValueError as e:
                st.error(f"Model training failed: {e}")
            else:
                # Evaluate
                y_pred = pipeline.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"Model trained — accuracy on holdout: {acc*100:.2f}%")

                # Interactive prediction: take values for selected features
                st.markdown("### Try a prediction")
                input_vals = {}
                for feat in sel_features:
                    if feat in numeric_cols:
                        # numeric input
                        vmin = int(float(X[feat].min()))
                        vmax = int(float(X[feat].max()))
                        default = int(float(X[feat].median()))
                        input_vals[feat] = st.number_input(feat, value=default, min_value=vmin, max_value=vmax)
                    else:
                        # categorical input
                        opts = sorted(X[feat].dropna().unique().tolist())
                        if opts:
                            input_vals[feat] = st.selectbox(feat, options=opts, index=0)
                        else:
                            # no options available
                            input_vals[feat] = st.text_input(feat, value="")

                if st.button("Predict Attrition"):
                    # create DataFrame with one row and same columns
                    single_df = pd.DataFrame([input_vals])
                    # Ensure numeric columns cast
                    for nc in numeric_cols:
                        single_df[nc] = pd.to_numeric(single_df[nc], errors="coerce")
                    try:
                        pred = pipeline.predict(single_df)[0]
                        proba = pipeline.predict_proba(single_df)[0]
                        st.write(f"Prediction: **{pred}**  (0 = stay, 1 = leave)")
                        st.write(f"Probabilities: stay={proba[0]:.2f}, leave={proba[1]:.2f}")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

# Heatmap tab
with tab_heatmap:
    st.subheader("Correlation Heatmap (numeric columns only)")
    num_df = filtered_df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")

    if num_df.shape[1] <= 1:
        st.warning("Not enough numeric columns for a correlation heatmap.")
    else:
        corr = num_df.corr()
        # If corr is all NaN, warn
        if corr.isnull().all().all():
            st.warning("Correlation matrix contains only NaN values — not enough numeric data.")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)

# Download tab
with tab_download:
    st.subheader("Download filtered data")
    buffered = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV of filtered data", buffered, file_name="filtered_hr_data.csv", mime="text/csv")

# Optional: show raw columns mapping for debugging (hidden by default)
with st.expander("Show detected/renamed columns (for debugging)"):
    st.write("Renamed columns mapping:", rename_map)
    st.write("Current dataframe columns:", filtered_df.columns.tolist())
