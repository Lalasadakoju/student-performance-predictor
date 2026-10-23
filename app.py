
import pickle
from pathlib import Path
from datetime import datetime
from io import BytesIO

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


# ==================================================
# CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Student Analytics | ML Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

FEATURES = [
    "Study_Hours",
    "Previous_Marks",
    "Attendance"
]

MODEL_NAMES = ["Linear Regression", "Random Forest"]

MODEL_METRICS = pd.DataFrame({
    "Model": MODEL_NAMES,
    "MAE": [1.04, 1.08],
    "MSE": [1.66, 1.85],
    "R² Score": [0.995, 0.995]
})


# ==================================================
# PROFESSIONAL UI STYLING
# ==================================================

st.markdown("""
<style>
    .stApp {
        background: #0b1120;
        color: #edf3ff;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #101a2b;
        border-right: 1px solid #26354d;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    .brand {
        font-size: 1.4rem;
        font-weight: 800;
        color: #f2f6ff;
    }

    .brand-subtitle {
        font-size: 0.82rem;
        color: #93a5c2;
        margin-top: 4px;
    }

    .sidebar-label {
        color: #8498b7;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin-top: 22px;
        margin-bottom: 10px;
    }

    [data-testid="stSidebar"] div.stButton > button {
        width: 100%;
        min-height: 44px;
        border-radius: 9px;
        border: 1px solid transparent;
        background: transparent;
        color: #b9c8df;
        text-align: left;
        justify-content: flex-start;
        padding: 0.65rem 0.85rem;
        font-size: 0.91rem;
        font-weight: 500;
    }

    [data-testid="stSidebar"] div.stButton > button:hover {
        background: #1b2a42;
        border-color: #31496b;
        color: #ffffff;
    }

    [data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background: #203d70;
        color: #ffffff;
        border: 1px solid #477bd5;
        font-weight: 700;
    }

    .hero {
        padding: 30px;
        border: 1px solid #2c4773;
        border-radius: 17px;
        background: linear-gradient(120deg, #152747, #142039 75%);
        margin-bottom: 25px;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 2rem;
        margin-bottom: 8px;
    }

    .hero p {
        color: #b8c9e5;
        font-size: 1rem;
        line-height: 1.7;
        margin-bottom: 0;
    }

    .eyebrow {
        color: #76a6ff;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        font-size: 0.72rem;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .panel {
        background: #111b2e;
        border: 1px solid #26354d;
        border-radius: 13px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .panel h3 {
        color: #edf3ff;
        font-size: 1.08rem;
        margin-top: 0;
    }

    .panel p {
        color: #9eafca;
        line-height: 1.6;
        font-size: 0.9rem;
    }

    .muted {
        color: #9aaac3;
        font-size: 0.87rem;
    }

    div[data-testid="stMetric"] {
        background: #111b2e;
        border: 1px solid #293953;
        border-radius: 12px;
        padding: 17px;
    }

    div[data-testid="stMetricLabel"] {
        color: #a9bad3;
    }

    div[data-testid="stMetricValue"] {
        color: #edf3ff;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 9px;
        min-height: 42px;
        font-weight: 600;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #283750;
        border-radius: 10px;
        overflow: hidden;
    }

    hr {
        border-color: #283750;
    }

    h1, h2, h3 {
        letter-spacing: -0.4px;
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD DATA AND MODELS
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "data.csv")


@st.cache_resource
def load_models():
    with open(BASE_DIR / "student_performance_model.pkl", "rb") as f:
        lr_model = pickle.load(f)

    with open(BASE_DIR / "random_forest_model.pkl", "rb") as f:
        rf_model = pickle.load(f)

    return lr_model, rf_model


try:
    df = load_data()
    lr_model, rf_model = load_models()

    missing_columns = [
        column for column in FEATURES + ["Final_Marks"]
        if column not in df.columns
    ]

    if missing_columns:
        st.error(f"Missing dataset columns: {missing_columns}")
        st.stop()

except FileNotFoundError as error:
    st.error(
        f"Required file not found: {error.filename}. "
        "Check that data.csv and both model files are in the project folder."
    )
    st.stop()

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def get_model(model_name):
    if model_name == "Linear Regression":
        return lr_model
    return rf_model


def get_category(marks):
    if marks >= 75:
        return "High"
    elif marks >= 50:
        return "Moderate"
    return "Needs Improvement"


def get_suggestion(category):
    suggestions = {
        "High": (
            "Continue regular study habits and revise topics consistently."
        ),
        "Moderate": (
            "Practise regularly and focus on topics that need improvement."
        ),
        "Needs Improvement": (
            "Follow a study timetable, revise basic concepts, "
            "and seek help when needed."
        )
    }
    return suggestions[category]


def draw_chart(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def validate_features(data):
    """Validate input features before making predictions."""
    if data.empty:
        return False, "The dataset is empty."

    if not all(column in data.columns for column in FEATURES):
        return False, "One or more required input columns are missing."

    numeric = data[FEATURES].apply(pd.to_numeric, errors="coerce")

    if numeric.isnull().any().any():
        return False, "Input features must contain valid numbers without missing values."

    if (
        (numeric["Study_Hours"] < 0).any()
        or (numeric["Study_Hours"] > 24).any()
    ):
        return False, "Study hours must be between 0 and 24."

    for column in ["Previous_Marks", "Attendance"]:
        if ((numeric[column] < 0).any() or (numeric[column] > 100).any()):
            return False, f"{column} must be between 0 and 100."

    return True, ""


def add_prediction_history(record):
    """Keep prediction history for the current Streamlit session."""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    st.session_state.prediction_history.insert(0, record)
    st.session_state.prediction_history = (
        st.session_state.prediction_history[:100]
    )


# ==================================================
# SESSION STATE
# ==================================================

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None

if "prediction_form_reset" not in st.session_state:
    st.session_state.prediction_form_reset = 0


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

PAGES = [
    ("Home", "⌂"),
    ("Student Prediction", "◎"),
    ("Analytics Dashboard", "▥"),
    ("Model Comparison", "⇄"),
    ("Batch Prediction", "▤"),
    ("Prediction History", "◷"),
    ("About Project", "ⓘ")
]

with st.sidebar:
    st.markdown(
        '<div class="brand">🎓 Student Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-subtitle">Performance Intelligence</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-label">WORKSPACE</div>',
        unsafe_allow_html=True
    )

    for page_name, icon in PAGES:
        if st.button(
            f"{icon}   {page_name}",
            key=f"nav_{page_name}",
            type=(
                "primary"
                if st.session_state.current_page == page_name
                else "secondary"
            ),
            use_container_width=True
        ):
            st.session_state.current_page = page_name
            st.rerun()

    st.divider()

    st.markdown(
        '<div class="sidebar-label">PROJECT STACK</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="muted">Python · Pandas · Scikit-learn</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="muted">Matplotlib · Seaborn · Streamlit</div>',
        unsafe_allow_html=True
    )

    st.divider()
    st.caption("Educational ML demonstration")

page = st.session_state.current_page


# ==================================================
# HOME PAGE
# ==================================================

if page == "Home":
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">MACHINE LEARNING PROJECT</div>
        <h1>Student Performance Analytics</h1>
        <p>
            Explore student data, understand learning patterns,
            compare prediction models, and estimate examination marks
            through one interactive dashboard.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Overview")
    st.markdown(
        '<p class="muted">A quick summary of the sample dataset and project.</p>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Dataset Records", len(df))
    c2.metric("Input Features", len(FEATURES))
    c3.metric("ML Models", 2)
    c4.metric("Prediction Target", "Final Marks")

    st.divider()
    st.markdown("## Explore the Workspace")

    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="panel">
            <h3>◎ Student Prediction</h3>
            <p>Enter study hours, previous marks, and attendance to estimate final marks.</p>
        </div>
        <div class="panel">
            <h3>⇄ Model Comparison</h3>
            <p>Compare the recorded evaluation results of two regression models.</p>
        </div>
        <div class="panel">
            <h3>◷ Prediction History</h3>
            <p>Review predictions generated during the current app session.</p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="panel">
            <h3>▥ Analytics Dashboard</h3>
            <p>Explore the dataset, descriptive statistics, charts, and correlation heatmap.</p>
        </div>
        <div class="panel">
            <h3>▤ Batch Prediction</h3>
            <p>Upload a CSV, generate estimates for multiple records, and download results.</p>
        </div>
        <div class="panel">
            <h3>ⓘ About the Project</h3>
            <p>Review the objective, technologies, algorithms, and limitations.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    dataset_bytes = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Sample Dataset",
        data=dataset_bytes,
        file_name="student_performance_dataset.csv",
        mime="text/csv",
        key="home_dataset_download",
        use_container_width=True
    )

    st.warning(
        "This project uses only 20 synthetic records. "
        "Predictions are educational estimates and must not be used "
        "for actual academic decisions."
    )


# ==================================================
# STUDENT PREDICTION PAGE
# ==================================================

elif page == "Student Prediction":
    st.markdown("## Student Prediction")
    st.markdown(
        '<p class="muted">Enter student details to estimate final examination marks.</p>',
        unsafe_allow_html=True
    )

    st.download_button(
        "⬇️ Download Sample Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="student_performance_dataset.csv",
        mime="text/csv",
        key="prediction_dataset_download"
    )

    reset_version = st.session_state.prediction_form_reset

    with st.form(key=f"prediction_form_{reset_version}"):
        model_name = st.selectbox(
            "Machine Learning Model",
            MODEL_NAMES,
            key=f"prediction_model_{reset_version}"
        )

        st.markdown("### Student Details")

        c1, c2, c3 = st.columns(3)

        with c1:
            study_hours = st.number_input(
                "Daily Study Hours",
                min_value=0.0,
                max_value=24.0,
                value=5.0,
                step=0.5,
                key=f"study_hours_{reset_version}"
            )

        with c2:
            previous_marks = st.number_input(
                "Previous Marks",
                min_value=0.0,
                max_value=100.0,
                value=65.0,
                step=1.0,
                key=f"previous_marks_{reset_version}"
            )

        with c3:
            attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=85.0,
                step=1.0,
                key=f"attendance_{reset_version}"
            )

        submitted = st.form_submit_button(
            "✨ Generate Prediction",
            type="primary",
            use_container_width=True
        )

    reset_col, spacer = st.columns([1, 3])

    with reset_col:
        if st.button("↻ Reset Inputs", use_container_width=True):
            st.session_state.prediction_form_reset += 1
            st.rerun()

    if submitted:
        input_data = pd.DataFrame({
            "Study_Hours": [study_hours],
            "Previous_Marks": [previous_marks],
            "Attendance": [attendance]
        })

        try:
            model = get_model(model_name)
            prediction = float(model.predict(input_data)[0])
            prediction = max(0.0, min(100.0, prediction))
            category = get_category(prediction)

            result = {
                "Date_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Model": model_name,
                "Study_Hours": study_hours,
                "Previous_Marks": previous_marks,
                "Attendance": attendance,
                "Predicted_Final_Marks": round(prediction, 2),
                "Performance_Category": category
            }

            st.session_state.latest_prediction = result
            add_prediction_history(result)

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    result = st.session_state.latest_prediction

    if result is not None:
        st.divider()
        st.markdown("### Prediction Result")

        a, b, c = st.columns(3)

        a.metric(
            "Estimated Marks",
            f"{result['Predicted_Final_Marks']:.2f} / 100"
        )

        b.metric("Performance Category", result["Performance_Category"])
        c.metric("Selected Model", result["Model"])

        st.progress(result["Predicted_Final_Marks"] / 100)

        st.markdown("### General Suggestion")
        st.info(get_suggestion(result["Performance_Category"]))

        individual_csv = pd.DataFrame([result]).to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download This Prediction",
            data=individual_csv,
            file_name="student_prediction_result.csv",
            mime="text/csv",
            key="individual_prediction_download"
        )

        st.caption(
            "This estimate is generated from a small synthetic dataset "
            "and is not an official academic assessment."
        )


# ==================================================
# ANALYTICS DASHBOARD
# ==================================================

elif page == "Analytics Dashboard":
    st.markdown("## Analytics Dashboard")
    st.markdown(
        '<p class="muted">Explore the sample data and visual relationships.</p>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)
    a.metric("Student Records", len(df))
    b.metric("Input Variables", len(FEATURES))
    c.metric("Output Variable", "Final Marks")

    st.download_button(
        "⬇️ Download Dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="student_performance_dataset.csv",
        mime="text/csv",
        key="analytics_dataset_download"
    )

    st.divider()

    with st.expander("View Dataset", expanded=True):
        st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Descriptive Statistics"):
        st.dataframe(df.describe().round(2), use_container_width=True)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("### Study Hours vs Final Marks")
        fig1, ax1 = plt.subplots(figsize=(7, 4))

        sns.scatterplot(
            data=df,
            x="Study_Hours",
            y="Final_Marks",
            s=80,
            ax=ax1
        )

        ax1.set_xlabel("Study Hours per Day")
        ax1.set_ylabel("Final Marks")
        ax1.set_ylim(0, 100)
        ax1.grid(alpha=0.25)

        draw_chart(fig1)

    with right:
        st.markdown("### Attendance vs Final Marks")
        fig2, ax2 = plt.subplots(figsize=(7, 4))

        sns.scatterplot(
            data=df,
            x="Attendance",
            y="Final_Marks",
            s=80,
            ax=ax2
        )

        ax2.set_xlabel("Attendance (%)")
        ax2.set_ylabel("Final Marks")
        ax2.set_ylim(0, 100)
        ax2.grid(alpha=0.25)

        draw_chart(fig2)

    st.divider()
    st.markdown("### Feature Correlation")

    fig3, ax3 = plt.subplots(figsize=(9, 5))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax3
    )

    draw_chart(fig3)

    st.caption(
        "Correlations describe associations in this synthetic dataset; "
        "they do not establish cause and effect."
    )


# ==================================================
# MODEL COMPARISON
# ==================================================

elif page == "Model Comparison":
    st.markdown("## Model Comparison")
    st.markdown(
        '<p class="muted">Compare the recorded test-set evaluation metrics.</p>',
        unsafe_allow_html=True
    )

    st.dataframe(
        MODEL_METRICS,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="panel">
            <h3>MAE</h3>
            <p>Mean Absolute Error. Lower values indicate smaller average absolute errors.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="panel">
            <h3>MSE</h3>
            <p>Mean Squared Error. Lower values indicate smaller average squared errors.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="panel">
            <h3>R² Score</h3>
            <p>Measures how well predictions fit the test-set target values.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Error Metric Comparison")

    fig, ax = plt.subplots(figsize=(8, 4))

    MODEL_METRICS.set_index("Model")[["MAE", "MSE"]].plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Error")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    draw_chart(fig)

    st.success(
        "In the recorded test results, Linear Regression has slightly "
        "lower MAE and MSE. Both models have the same reported R² score."
    )

    st.warning(
        "Evaluation used only five test records from a synthetic dataset. "
        "These results do not demonstrate real-world predictive performance."
    )


# ==================================================
# BATCH PREDICTION
# ==================================================

elif page == "Batch Prediction":
    st.markdown("## Batch Student Prediction")
    st.markdown(
        '<p class="muted">Upload a CSV file to estimate marks for multiple records.</p>',
        unsafe_allow_html=True
    )

    st.download_button(
        "⬇️ Download Sample Dataset",
        data=df[FEATURES].to_csv(index=False).encode("utf-8"),
        file_name="student_prediction_template.csv",
        mime="text/csv",
        key="batch_template_download"
    )

    model_name = st.selectbox(
        "Machine Learning Model",
        MODEL_NAMES,
        key="batch_model_choice"
    )

    st.markdown("### Required CSV Format")

    st.code(
        "Study_Hours,Previous_Marks,Attendance\n"
        "2,40,60\n"
        "4,60,75\n"
        "6,80,90",
        language="csv"
    )

    uploaded_file = st.file_uploader(
        "Upload Student CSV",
        type=["csv"],
        key="batch_csv_upload"
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            valid, message = validate_features(batch_df)

            if not valid:
                st.error(message)

            else:
                numeric_df = batch_df[FEATURES].apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                model = get_model(model_name)
                raw_predictions = model.predict(numeric_df)

                results = batch_df.copy()

                results["Predicted_Final_Marks"] = [
                    round(max(0.0, min(100.0, float(value))), 2)
                    for value in raw_predictions
                ]

                results["Performance_Category"] = (
                    results["Predicted_Final_Marks"].apply(get_category)
                )

                st.success(
                    f"Predictions generated for {len(results)} records."
                )

                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "⬇️ Download Prediction Results",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="student_prediction_results.csv",
                    mime="text/csv",
                    key="batch_results_download"
                )

        except (pd.errors.ParserError, UnicodeDecodeError):
            st.error("Unable to read this file. Please upload a valid CSV.")

        except Exception as error:
            st.error(f"Unable to generate batch predictions: {error}")


# ==================================================
# PREDICTION HISTORY
# ==================================================

elif page == "Prediction History":
    st.markdown("## Prediction History")
    st.markdown(
        '<p class="muted">Predictions generated during the current app session.</p>',
        unsafe_allow_html=True
    )

    history = st.session_state.prediction_history

    if history:
        history_df = pd.DataFrame(history)

        c1, c2 = st.columns(2)
        c1.metric("Predictions This Session", len(history_df))
        c2.metric(
            "Average Estimated Marks",
            f"{history_df['Predicted_Final_Marks'].mean():.2f}"
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Prediction History",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
            key="history_download"
        )

        if st.button("🗑️ Clear Prediction History"):
            st.session_state.prediction_history = []
            st.session_state.latest_prediction = None
            st.rerun()

    else:
        st.info(
            "No predictions yet. Open Student Prediction to generate your first result."
        )


# ==================================================
# ABOUT PROJECT
# ==================================================

elif page == "About Project":
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">PROJECT PROFILE</div>
        <h1>Student Performance Analytics</h1>
        <p>
            A beginner-level Data Science and Machine Learning demonstration
            that explores student-related data and estimates examination marks.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Project Objective")

    st.write(
        "To demonstrate data loading, exploratory data analysis, "
        "visualization, regression model training, model evaluation, "
        "and prediction through an interactive interface."
    )

    st.divider()
    st.markdown("### Technologies")

    technology_df = pd.DataFrame({
        "Technology": [
            "Python",
            "Pandas",
            "NumPy",
            "Matplotlib",
            "Seaborn",
            "Scikit-learn",
            "Streamlit"
        ],
        "Purpose": [
            "Programming language",
            "Data loading and manipulation",
            "Numerical operations",
            "Charts and visualization",
            "Correlation heatmap",
            "Machine Learning models",
            "Interactive web interface"
        ]
    })

    st.dataframe(
        technology_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.markdown("### Input Features")

    st.markdown("""
    - **Study Hours:** Daily study time.
    - **Previous Marks:** Marks obtained in a previous examination.
    - **Attendance:** Attendance percentage.
    - **Target:** Final examination marks.
    """)

    st.markdown("### Algorithms")

    st.markdown("""
    **Linear Regression:** Estimates a numeric result by learning a linear
    relationship between input variables and final marks.

    **Random Forest Regression:** Combines predictions from multiple
    decision trees to estimate a numeric result.
    """)

    st.divider()
    st.markdown("### Limitations")

    st.warning("""
    - The dataset contains only 20 synthetic records.
    - Model evaluation is based on a test set of five records.
    - The sample data is not taken from real students.
    - Predictions are educational estimates, not official academic results.
    """)


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.markdown("""
<p style="text-align:center; color:#8291aa; font-size:0.82rem;">
    STUDENT PERFORMANCE ANALYTICS
    &nbsp; • &nbsp; EDUCATIONAL ML PROJECT
</p>
""", unsafe_allow_html=True)
