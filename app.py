"""Streamlit demo for the exploratory AI Dependency Index prediction model.

Run from the project folder with: python -m streamlit run app.py
The application does not store questionnaire responses.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "outputs" / "models"
RESULTS = ROOT / "outputs" / "results_summary.json"
MODEL_PATH = MODELS / "best_regression_pipeline.joblib"
METADATA_PATH = MODELS / "model_metadata.json"

RESPONSE_OPTIONS = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5,
}

USAGE_ITEMS = [
    "I use generative AI tools regularly for academic purposes.",
    "I use AI tools for assignment writing.",
    "I use AI tools for study planning.",
    "I use AI tools for research-related tasks.",
    "I depend on AI tools to complete academic work faster.",
]
TRUST_ITEMS = [
    "I trust AI-generated academic responses.",
    "AI tools usually provide reliable information.",
    "I rarely verify AI-generated answers before using them.",
    "I believe AI tools make better academic suggestions than I can myself.",
    "I trust AI recommendations for academic decision-making.",
]
OFFLOADING_ITEMS = [
    "Using AI tools reduces my need to think deeply.",
    "I analyze less when using AI-generated responses.",
    "I prefer AI explanations over conducting my own research.",
    "AI tools reduce my independent problem-solving efforts.",
    "I rely more on AI-generated ideas than my own thinking.",
]
DECISION_ITEMS = [
    "AI tools influence my academic decisions.",
    "I use AI suggestions for study-related decisions.",
    "AI tools help me choose research topics or project ideas.",
    "I use AI recommendations to improve academic performance.",
    "AI tools positively affect my academic decision-making process.",
]

# These are the observed categories in the source survey data.
DEPARTMENTS = [
    "Accounting", "Architecture", "BBA", "CSE", "Chemical Engineering", "Chemistry",
    "Civil Engineering", "Computer Science and Engineering", "Development Studies", "EEE",
    "Economics", "Electrical and Electronic Engineering", "English", "Environmental Science",
    "Finance", "Human Resource Management", "Industrial and Production Engineering",
    "Information Technology", "International Business", "Journalism and Media Studies", "Law",
    "Management", "Marketing", "Mass Communication", "Mathematics", "Mechanical Engineering",
    "Pharmacy", "Physics", "Political Science", "Psychology", "Public Administration", "Sociology",
    "Software Engineering", "Statistics", "Textile Engineering",
]


def survey_columns(prefix: str, items: list[str]) -> list[str]:
    """Return source-data column names for a block of survey statements."""
    return [f"{prefix} [{item}]" for item in items]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def likert_question(question: str, key: str) -> int:
    """Show one Likert item and return its 1--5 numeric coding."""
    answer = st.radio(
        question,
        options=list(RESPONSE_OPTIONS),
        index=2,
        horizontal=True,
        key=key,
    )
    return RESPONSE_OPTIONS[answer]


def item_block(title: str, caption: str, items: list[str], key_prefix: str) -> list[int]:
    st.markdown(f"#### {title}")
    st.caption(caption)
    return [likert_question(item, f"{key_prefix}_{number}") for number, item in enumerate(items)]


def mean(values: list[int]) -> float:
    return sum(values) / len(values)


st.set_page_config(page_title="AI Dependency Index Demo", page_icon="📊", layout="wide")

if not MODEL_PATH.is_file() or not METADATA_PATH.is_file() or not RESULTS.is_file():
    st.error(
        "Model artifacts are missing. Run all cells in "
        "AI_Dependency_Study_anaconda_compatible.ipynb before opening this demo."
    )
    st.stop()

metadata = load_json(METADATA_PATH)
summary = load_json(RESULTS)

st.markdown(
    """
    <style>
      .hero {
        padding: 1.6rem 2rem;
        border-radius: 1rem;
        color: white;
        background: linear-gradient(110deg, #3867d6, #7c3fb7);
        margin-bottom: 1.4rem;
      }
      .hero h1 { color: white; margin: 0 0 .35rem 0; font-size: 2rem; }
      .hero p { margin: 0; font-size: 1rem; opacity: .95; }
      div[data-testid="stMetric"] {
        background: #f5f8ff;
        border: 1px solid #dce7ff;
        border-radius: .7rem;
        padding: .8rem;
      }
      div[data-testid="stFormSubmitButton"] > button {
        width: 100%; background: #3867d6; color: white; border: 0; font-weight: 600;
      }
    </style>
    <div class="hero">
      <h1>📊 AI Dependency Index Demo</h1>
      <p>Answer the self-report questions to receive an exploratory model estimate on the 1–5 survey scale.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model information")
    st.success("Exploratory portfolio model")
    st.caption("Selected model")
    st.write(f"**{summary['best_model']}**")
    st.caption("Held-out test performance")
    st.write(f"RMSE: **{summary['test_metrics']['RMSE']:.3f}**")
    st.write(f"R²: **{summary['test_metrics']['R2']:.3f}**")
    st.divider()
    st.caption("Response scale")
    st.write("1 = Strongly Disagree\n\n5 = Strongly Agree")
    st.divider()
    st.caption("Privacy")
    st.write("Your answers are used only in this browser session and are not saved by this app.")

st.info(
    "This is an exploratory educational demonstration. The result is not a psychological or clinical diagnosis, "
    "a measure of intelligence or cognitive ability, or a basis for decisions about a person."
)
st.caption(
    "The model has one internal test-set evaluation only. External validation, calibration analysis, and subgroup "
    "prediction-performance checks have not yet been established; no risk category or action should be inferred."
)

form_column, result_column = st.columns([1.65, 1], gap="large")

with form_column:
    st.subheader("Complete the questionnaire")
    st.caption("Select the answer that best reflects your current academic use of generative AI tools.")

    with st.form("prediction_questionnaire", border=False):
        with st.expander("About you", expanded=True):
            demographic_left, demographic_right = st.columns(2)
            with demographic_left:
                gender = st.selectbox("Gender", ["Female", "Male"])
                age = st.number_input("Age", min_value=18, max_value=28, value=22, step=1)
                university_type = st.selectbox(
                    "University Type", ["Private University", "Public University", "Other"]
                )
            with demographic_right:
                academic_level = st.selectbox("Academic Level", ["Undergraduate", "Postgraduate"])
                department = st.selectbox("Department/Discipline", DEPARTMENTS)

        usage_values = item_block(
            "1. Academic AI use", "Tell us how you use generative AI for your academic work.", USAGE_ITEMS, "usage"
        )
        trust_values = item_block(
            "2. Trust in AI", "Tell us how much you trust AI in academic contexts.", TRUST_ITEMS, "trust"
        )
        offloading_values = item_block(
            "3. Thinking while using AI", "Tell us about your learning and problem-solving experience.",
            OFFLOADING_ITEMS, "offloading"
        )
        decision_values = item_block(
            "4. Academic decisions", "Tell us how AI affects your study-related decisions.", DECISION_ITEMS, "decision"
        )
        submitted = st.form_submit_button("Generate my exploratory estimate", type="primary")

    if submitted:
        predictor_row = dict(zip(survey_columns("AI Usage", USAGE_ITEMS), usage_values))
        predictor_row.update(
            {
                "Gender": gender,
                "Age": int(age),
                "University Type": university_type,
                "Academic Level": academic_level,
                "Department/Discipline": department,
                "AI_Trust_Index": mean(trust_values),
                "Cognitive_Offloading_Index": mean(offloading_values),
                "Academic_Decision_Index": mean(decision_values),
            }
        )
        feature_frame = pd.DataFrame([predictor_row]).reindex(columns=metadata["features"])
        prediction = float(load_model().predict(feature_frame)[0])
        st.session_state["prediction"] = prediction
        st.session_state["input_indices"] = {
            "AI Usage": mean(usage_values),
            "AI Trust": mean(trust_values),
            "Cognitive Offloading": mean(offloading_values),
            "Academic Decision-Making": mean(decision_values),
        }

with result_column:
    st.subheader("Your prediction")
    if "prediction" not in st.session_state:
        st.info("Complete the questionnaire and select **Generate my exploratory estimate** to view the model result.")
    else:
        prediction = st.session_state["prediction"]
        st.success(f"Based on your responses, the model estimates an AI Dependency Index of {prediction:.1f}/5.")
        st.metric("Estimated AI Dependency Index", f"{prediction:.2f} / 5")
        st.progress(min(100, max(0, round(prediction / 5 * 100))), text=f"Estimated position on the 1–5 survey scale: {prediction:.2f}")
        st.caption("This bar displays the model estimate on the survey response scale. It does not define severity levels.")
        st.divider()
        st.markdown("**Questionnaire averages used by the model**")
        for name, value in st.session_state["input_indices"].items():
            st.write(f"{name}: **{value:.2f}/5**")
        st.divider()
        st.caption(
            "The model predicts the dependency-item average without asking the five dependency questions that construct "
            "that target. This avoids direct target leakage, but the estimate remains exploratory."
        )

with st.expander("About this demo and its limits"):
    st.markdown(
        """
        This application uses the saved Gradient Boosting regression pipeline from the accompanying portfolio study.
        It uses AI-use responses, demographics, and averages from the trust, cognitive-offloading, and academic-decision
        questions. It does not collect or save individual answers.

        The source survey is cross-sectional and self-reported. The dependency composite had weak internal consistency
        in the primary analysis, and the model has not been externally validated. Therefore, the result should be read
        only as an illustrative model estimate, not as evidence of a condition or a fact about an individual.
        """
    )
