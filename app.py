import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

st.set_page_config(
    page_title="Student Depression Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Load model
model = joblib.load("model_depression.pkl")

# Dataset
df = pd.read_csv("depression.csv")

# CSS
st.markdown("""
<style>
.stApp {
    background-color: #000000;
}

.main-title {
    font-size: 42px;
    font-weight: bold;
    color: white;
}

.subtitle {
    font-size: 18px;
    color: #D1D5DB;
}

.metric-box {
    background-color: #111827;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #374151;
    text-align: center;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🧠 Navigation")
st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning Project")
menu = st.sidebar.radio(
    "Go to",
    ["Home", "Analytics Dashboard", "Prediction"]
)

# ================= HOME =================
if menu == "Home":
    st.markdown('<p class="main-title">Student Depression Prediction Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Machine Learning-based dashboard to analyze and predict depression risk among students.</p>', unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">📂 Dataset Size</div>
            <div class="metric-value">{len(df)}</div>
            <small>Total student records</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">📊 Total Features</div>
            <div class="metric-value">8</div>
            <small>Selected variables</small>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">🌳 Decision Tree</div>
            <div class="metric-value">69%</div>
            <small>Baseline model</small>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">🌲 Random Forest</div>
            <div class="metric-value">77%</div>
            <small>Best model</small>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    st.subheader("📊 Model Accuracy Comparison")

    comparison_df = pd.DataFrame({
        "Model": ["Decision Tree", "Random Forest"],
        "Accuracy": [69.01, 76.94]
    })

    fig = px.bar(
        comparison_df,
        x="Model",
        y="Accuracy",
        text="Accuracy",
        title="Accuracy Comparison"
    )

    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)
    st.write("")
    st.subheader("📋 Model Evaluation")

    with st.expander("🌳 Decision Tree Evaluation"):
        st.code("""
    Accuracy: 69.01%

            Precision  Recall  F1-score
    Class 0:   0.63     0.64     0.64
    Class 1:   0.73     0.73     0.73

    Macro Avg: 0.68     0.68     0.68
    Weighted Avg: 0.69  0.69     0.69
        """)
    
    with st.expander("🌲 Random Forest Evaluation"):
        st.code("""
    Accuracy: 76.94%

            Precision  Recall  F1-score
    Class 0:   0.74     0.69     0.72
    Class 1:   0.79     0.82     0.80

    Macro Avg: 0.77     0.76     0.76
    Weighted Avg: 0.77  0.77     0.77
        """)

# ================= ANALYTICS =================
elif menu == "Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    st.write("Interactive exploratory data analysis and model evaluation")

    # -----------------------------
    # Depression Distribution
    # -----------------------------
    st.subheader("Depression Distribution")

    dep_count = df["Depression"].value_counts().reset_index()
    dep_count.columns = ["Depression", "Count"]
    dep_count["Depression"] = dep_count["Depression"].map({0: "No Depression", 1: "Depression"})

    fig = px.pie(
        dep_count,
        names="Depression",
        values="Count",
        hole=0.4,
        title="Depression Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Age Distribution
    # -----------------------------
    st.subheader("Age Distribution")

    fig = px.histogram(
        df,
        x="Age",
        nbins=25,
        title="Age Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Gender vs Depression
    # -----------------------------
    st.subheader("Gender vs Depression")

    gender_dep = pd.crosstab(df["Gender"], df["Depression"])
    gender_dep.columns = ["No Depression", "Depression"]

    fig = px.bar(
        gender_dep,
        barmode="group",
        title="Gender vs Depression"
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Academic Pressure
    # -----------------------------
    st.subheader("Academic Pressure Distribution")

    fig = px.histogram(
        df,
        x="Academic Pressure",
        title="Academic Pressure Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Financial Stress
    # -----------------------------
    st.subheader("Financial Stress Distribution")

    fig = px.histogram(
        df,
        x="Financial Stress",
        title="Financial Stress Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Feature Importance
    # -----------------------------
    st.subheader("Feature Importance (Random Forest)")

    feature_df = pd.DataFrame({
        "Feature": [
            "Gender",
            "Age",
            "Academic Pressure",
            "Study Satisfaction",
            "Sleep Duration",
            "Dietary Habits",
            "Financial Stress",
            "Work/Study Hours"
        ],
        "Importance": [
            0.033569,
            0.200831,
            0.228892,
            0.090936,
            0.080420,
            0.060143,
            0.137595,
            0.167614
        ]
    })

    fig = px.bar(
        feature_df.sort_values(by="Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Confusion Matrix DT
    # -----------------------------
    st.subheader("Decision Tree Confusion Matrix")

    cm_dt = np.array([
        [1471, 831],
        [853, 2279]
    ])

    fig = px.imshow(
        cm_dt,
        text_auto=True,
        title="Decision Tree Confusion Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Confusion Matrix RF
    # -----------------------------
    st.subheader("Random Forest Confusion Matrix")

    cm_rf = np.array([
        [1599, 703],
        [550, 2582]
    ])

    fig = px.imshow(
        cm_rf,
        text_auto=True,
        title="Random Forest Confusion Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)

# ================= PREDICTION =================
elif menu == "Prediction":
    st.title("🔍 Predict Depression Risk")
    st.write("Input student information below to predict depression risk.")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=15, max_value=40, value=20)
        academic_pressure = st.slider("Academic Pressure", 0, 5, 3)
        study_satisfaction = st.slider("Study Satisfaction", 0, 5, 3)

    with col2:
        sleep = st.selectbox(
            "Sleep Duration",
            ["5-6 hours", "Less than 5 hours", "7-8 hours", "More than 8 hours", "Others"]
        )

        diet = st.selectbox(
            "Dietary Habits",
            ["Healthy", "Moderate", "Unhealthy", "Others"]
        )

        financial_stress = st.slider("Financial Stress", 0, 5, 3)
        study_hours = st.slider("Study Hours", 0, 12, 4)

    if st.button("Predict", use_container_width=True):

        gender_map = {
            "Male": 0,
            "Female": 1
        }

        sleep_map = {
            "5-6 hours": 0,
            "Less than 5 hours": 1,
            "7-8 hours": 2,
            "More than 8 hours": 3,
            "Others": 4
        }

        diet_map = {
            "Healthy": 0,
            "Moderate": 1,
            "Unhealthy": 2,
            "Others": 3
        }

        input_data = np.array([[
            gender_map[gender],
            age,
            academic_pressure,
            study_satisfaction,
            sleep_map[sleep],
            diet_map[diet],
            financial_stress,
            study_hours
        ]])

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        st.write("")
        st.subheader("Prediction Result")

        if prediction[0] == 1:
            risk_score = int(probability[0][1] * 100)
            st.error(f"⚠️ High Risk of Depression ({risk_score}%)")

            st.warning("""
Recommendations:
- Improve sleep quality
- Reduce academic pressure
- Maintain healthy eating habits
- Consider professional support
            """)

        else:
            safe_score = int(probability[0][0] * 100)
            st.success(f"✅ Low Risk of Depression ({safe_score}%)")

            st.info("""
Recommendations:
- Maintain healthy routines
- Keep balanced study hours
- Continue good sleep habits
- Monitor stress regularly
            """)