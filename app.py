import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Advanced HR Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Advanced HR Analytics Dashboard")
st.markdown("Analyze employee performance, HR metrics & run AI predictions.")

# -----------------------------------------------------------
# FETCH SAMPLE DATA
# -----------------------------------------------------------
@st.cache_data
def fetch_data():
    try:
        response = requests.get(
            "https://jsonplaceholder.typicode.com/users",
            verify=False
        )
        df = pd.DataFrame(response.json())
        df["address"] = df["address"].apply(lambda x: f"{x['street']} {x['suite']}, {x['city']} {x['zipcode']}")
        df["company"] = df["company"].apply(lambda x: f"{x['bs']} {x['catchPhrase']} {x['name']}")
        return df
    except:
        return pd.DataFrame()

preloaded_df = fetch_data()

# -----------------------------------------------------------
# SIDEBAR DATA SOURCE
# -----------------------------------------------------------
st.sidebar.header("📤 Data Options")
data_source = st.sidebar.radio(
    "Select Data Source",
    ["Preloaded Data", "Upload Your Own Data"]
)

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
if data_source == "Upload Your Own Data":
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.success("Uploaded data loaded successfully!")
    else:
        st.warning("Upload a CSV file to continue.")
        st.stop()

else:
    df = preloaded_df.copy()
    # Add synthetic HR fields only for demo dataset
    df['department'] = np.random.choice(['HR', 'Sales', 'IT', 'Finance', 'Operations'], len(df))
    df['gender'] = np.random.choice(['Male', 'Female'], len(df))
    df['age'] = np.random.randint(25, 60, len(df))
    df['length_of_service'] = np.random.randint(1, 20, len(df))
    df['no_of_trainings'] = np.random.randint(1, 10, len(df))
    df['previous_year_rating'] = np.random.randint(1, 6, len(df))
    df['salary'] = np.random.randint(30000, 150000, len(df))
    df['promoted'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])

# -----------------------------------------------------------
# SIDEBAR FILTERS (This part caused the bug earlier)
# NOW placed OUTSIDE the condition → fixes reset issue
# -----------------------------------------------------------
st.sidebar.header("🔍 Filters")

if "department" in df.columns:
    selected_depts = st.sidebar.multiselect(
        "Select Departments",
        options=df["department"].unique(),
        default=df["department"].unique()
    )
else:
    selected_depts = None

gender_filter = st.sidebar.selectbox(
    "Gender",
    ["All", "Male", "Female"]
)

if "age" in df.columns:
    min_age, max_age = st.sidebar.slider(
        "Age Range",
        int(df.age.min()), int(df.age.max()),
        (int(df.age.min()), int(df.age.max()))
    )
else:
    min_age, max_age = None, None

# -----------------------------------------------------------
# APPLY FILTERS SAFELY
# -----------------------------------------------------------
filtered = df.copy()

if selected_depts is not None:
    filtered = filtered[filtered["department"].isin(selected_depts)]

if gender_filter != "All":
    filtered = filtered[filtered["gender"] == gender_filter]

if min_age is not None:
    filtered = filtered[(filtered["age"] >= min_age) & (filtered["age"] <= max_age)]

# -----------------------------------------------------------
# TABS
# -----------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Visualizations", "AI Predictions", "Heatmap", "Download"
])

# -----------------------------------------------------------
# TAB 1 – OVERVIEW
# -----------------------------------------------------------
with tab1:
    st.subheader("📊 Key KPIs")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Employees", len(filtered))

    if "salary" in filtered.columns:
        k2.metric("Avg Salary", f"${filtered.salary.mean():,.0f}")
    else:
        k2.metric("Avg Salary", "N/A")

    if "promoted" in filtered.columns:
        k3.metric("Promotion Rate", f"{filtered.promoted.mean()*100:.1f}%")
    else:
        k3.metric("Promotion Rate", "N/A")

    if "length_of_service" in filtered.columns:
        k4.metric("Avg Service (yrs)", f"{filtered.length_of_service.mean():.1f}")
    else:
        k4.metric("Avg Service (yrs)", "N/A")

    st.write("### Data Preview")
    st.dataframe(filtered.head())

# -----------------------------------------------------------
# TAB 2 – VISUALIZATIONS
# -----------------------------------------------------------
with tab2:
    st.subheader("📈 Visual Insights")

    # Gender Pie
    if "gender" in filtered.columns:
        fig = px.pie(filtered, names="gender", title="Gender Distribution")
        st.plotly_chart(fig)

    # Department Count
    if "department" in filtered.columns:
        fig = px.bar(filtered, x="department", title="Employee Count by Dept")
        st.plotly_chart(fig)

    # Age Distribution
    if "age" in filtered.columns:
        fig = px.histogram(filtered, x="age", color="department",
                           title="Age Distribution by Department")
        st.plotly_chart(fig)

    # Salary Violin
    if "salary" in filtered.columns:
        fig = px.violin(filtered, x="department", y="salary", box=True,
                        title="Salary Distribution by Department")
        st.plotly_chart(fig)

# -----------------------------------------------------------
# TAB 3 – AI MODEL
# -----------------------------------------------------------
with tab3:
    st.subheader("🤖 Promotion Prediction")

    required = ["age", "length_of_service", "previous_year_rating", "no_of_trainings", "promoted"]

    if all(col in filtered.columns for col in required):

        if filtered["promoted"].nunique() > 1:

            X = filtered[required[:-1]]
            y = filtered["promoted"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            model = LogisticRegression()
            model.fit(X_train, y_train)

            acc = accuracy_score(y_test, model.predict(X_test))
            st.success(f"Model Accuracy: {acc*100:.2f}%")

            age = st.number_input("Age", 20, 60, 30)
            service = st.number_input("Years of Service", 1, 20, 5)
            rating = st.slider("Rating (Last Year)", 1, 5, 3)
            training = st.number_input("No. of Trainings", 1, 10, 2)

            if st.button("Predict Promotion"):
                pred = model.predict([[age, service, rating, training]])[0]
                if pred:
                    st.success("Employee is likely to be promoted!")
                else:
                    st.warning("Employee is unlikely to be promoted.")
        else:
            st.warning("Column 'promoted' has only 1 unique value.")

    else:
        st.warning("Missing required columns for ML model.")

# -----------------------------------------------------------
# TAB 4 – HEATMAP
# -----------------------------------------------------------
with tab4:
    st.subheader("📊 Correlation Heatmap")

    num_cols = filtered.select_dtypes(include=['int', 'float'])
    if len(num_cols.columns) > 1:
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(num_cols.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns.")

# -----------------------------------------------------------
# TAB 5 – DOWNLOAD
# -----------------------------------------------------------
with tab5:
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "filtered_data.csv", "text/csv")
