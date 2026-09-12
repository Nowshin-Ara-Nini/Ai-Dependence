"""Local Streamlit dashboard for the exploratory AI-dependency portfolio study.

Run from the project root with: python -m streamlit run app.py
It reads aggregate results and the saved pipeline, and does not save form inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
MODELS = OUT / "models"

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
DEPARTMENTS = [
    "CSE", "EEE", "Economics", "Pharmacy", "Law", "Architecture", "Mathematics", "English",
    "Civil Engineering", "BBA", "Software Engineering", "Journalism and Media Studies", "Physics",
    "Sociology", "Mass Communication", "Accounting", "Chemical Engineering", "Chemistry",
    "Computer Science and Engineering", "Development Studies", "Electrical and Electronic Engineering",
    "Environmental Science", "Finance", "Human Resource Management", "Industrial and Production Engineering",
    "Information Technology", "International Business", "Management", "Marketing", "Mechanical Engineering",
    "Political Science", "Psychology", "Public Administration", "Statistics", "Textile Engineering",
]


def survey_columns(prefix: str, statements: list[str]) -> list[str]:
    return [f"{prefix} [{statement}]" for statement in statements]


def label(statement: str) -> str:
    return statement.removeprefix("I ").rstrip(".")


def available(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


@st.cache_data
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data
def load_table(filename: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / filename)


@st.cache_resource
def load_model():
    return joblib.load(MODELS / "best_regression_pipeline.joblib")


def show_figure(filename: str, caption: str) -> None:
    path = FIGURES / filename
    if available(path):
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Run the notebook to create {filename}.")


def item_slider(statement: str, key: str) -> int:
    return st.slider(label(statement), 1, 5, 3, key=key)


st.set_page_config(page_title="AI Dependency Study", page_icon="📊", layout="wide")
st.title("AI Dependency and Cognitive Offloading Among University Students")
st.caption("Exploratory portfolio dashboard based on self-reported survey responses")

summary_path = OUT / "results_summary.json"
metadata_path = MODELS / "model_metadata.json"
model_path = MODELS / "best_regression_pipeline.joblib"
if not available(summary_path) or not available(metadata_path):
    st.error("Analysis artifacts are missing. Run all cells in AI_Dependency_Study_anaconda_compatible.ipynb first.")
    st.stop()

summary = load_json(str(summary_path))
metadata = load_json(str(metadata_path))
st.warning(
    "This dashboard is exploratory. It does not diagnose AI dependency, cognitive impairment, "
    "mental-health conditions, intelligence, memory, or cognitive ability."
)
page = st.sidebar.radio("Navigate", ["Overview", "Study findings", "Exploratory estimate", "Methods and limits"])

if page == "Overview":
    st.subheader("Study snapshot")
    columns = st.columns(4)
    columns[0].metric("Raw survey rows", f"{summary['n_raw']:,}")
    columns[1].metric("Primary response patterns", f"{summary['n_primary']:,}")
    columns[2].metric("Selected model", summary["best_model"])
    columns[3].metric("Held-out R²", f"{summary['test_metrics']['R2']:.3f}")
    st.write(
        "The primary analysis gives each exact response pattern equal weight because 451 rows repeat complete "
        "response patterns. This does not prove that the records came from repeated respondents."
    )
    first, second = st.columns(2)
    with first:
        show_figure("03_spearman_heatmap.png", "Spearman correlations among exploratory item averages")
    with second:
        show_figure("08_model_comparison.png", "Cross-validation and held-out regression performance")

elif page == "Study findings":
    st.subheader("Association and model results")
    relationships = load_table("spearman_relationships.csv")
    st.dataframe(relationships.query("sample == 'primary_distinct'"), use_container_width=True, hide_index=True)
    first, second = st.columns(2)
    with first:
        show_figure("11_permutation_importance.png", "Held-out permutation importance for the selected model")
    with second:
        show_figure("16_cluster_profiles.png", "Exploratory cluster centroids on the original 1–5 scale")
    st.subheader("Supplementary enhancement checks")
    first, second = st.columns(2)
    with first:
        st.caption("Training-only hyperparameter search; no test-set reuse")
        st.dataframe(load_table("training_only_hyperparameter_search.csv"), use_container_width=True, hide_index=True)
    with second:
        st.caption("Alternative clustering comparison")
        st.dataframe(load_table("alternative_clustering_comparison.csv"), use_container_width=True, hide_index=True)
    show_figure("18_alternative_clustering.png", "Internal clustering criteria across methods")

elif page == "Exploratory estimate":
    st.subheader("Estimate the model output")
    st.write(
        "Use the survey coding: 1 = Strongly Disagree, 2 = Disagree, 3 = Neutral, 4 = Agree, and 5 = Strongly Agree. "
        "No form values are written to disk."
    )
    st.info(
        "The output is an exploratory prediction from the saved Model B pipeline. It is not a validated score, "
        "diagnosis, eligibility decision, or student ranking."
    )
    if not available(model_path):
        st.error("The saved model is missing. Run the notebook first.")
        st.stop()

    with st.form("prediction_form"):
        demographics, usage = st.columns(2)
        with demographics:
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input("Age", min_value=18, max_value=28, value=22, step=1)
            university_type = st.selectbox("University Type", ["Private University", "Public University", "Other"])
            academic_level = st.selectbox("Academic Level", ["Undergraduate", "Postgraduate"])
            department = st.selectbox("Department/Discipline", DEPARTMENTS)
        with usage:
            st.markdown("**AI usage items**")
            usage_values = [item_slider(item, f"usage_{number}") for number, item in enumerate(USAGE_ITEMS)]

        trust, offloading, decisions = st.columns(3)
        with trust:
            st.markdown("**Trust in AI items**")
            trust_values = [item_slider(item, f"trust_{number}") for number, item in enumerate(TRUST_ITEMS)]
        with offloading:
            st.markdown("**Cognitive-offloading items**")
            offloading_values = [item_slider(item, f"offloading_{number}") for number, item in enumerate(OFFLOADING_ITEMS)]
        with decisions:
            st.markdown("**Academic-decision items**")
            decision_values = [item_slider(item, f"decision_{number}") for number, item in enumerate(DECISION_ITEMS)]
        submitted = st.form_submit_button("Generate exploratory estimate")

    if submitted:
        predictor_row = dict(zip(survey_columns("AI Usage", USAGE_ITEMS), usage_values))
        predictor_row.update({
            "Gender": gender,
            "Age": int(age),
            "University Type": university_type,
            "Academic Level": academic_level,
            "Department/Discipline": department,
            "AI_Trust_Index": sum(trust_values) / len(trust_values),
            "Cognitive_Offloading_Index": sum(offloading_values) / len(offloading_values),
            "Academic_Decision_Index": sum(decision_values) / len(decision_values),
        })
        feature_frame = pd.DataFrame([predictor_row]).reindex(columns=metadata["features"])
        prediction = float(load_model().predict(feature_frame)[0])
        st.success(f"Based on your responses, the model estimates an AI Dependency Index of {prediction:.1f}/5.")
        st.metric("Exploratory predicted AI Dependency Index", f"{prediction:.2f} / 5")
        st.bar_chart(pd.DataFrame({"Index value": [prediction]}, index=["Model estimate"]))
        st.caption("The bar shows the estimated value on the survey's 1–5 response scale; it is not a severity category.")
        st.caption(
            "The model excludes the five dependency items that construct the target. Its prediction can still use "
            "related self-reported constructs and must not be used to make decisions about a person."
        )

else:
    st.subheader("Methods, privacy, and limitations")
    st.markdown(
        """
        - The survey is cross-sectional and self-reported. Associations and feature importance do not establish causation.
        - Primary internal-consistency estimates were weak, so the composite averages are exploratory rather than validated scales.
        - The selected model has held-out RMSE of about 0.504 and R² of about 0.292. It is not externally validated.
        - The dashboard loads aggregate outputs and a saved model. It does not display respondent-level records or save form submissions.
        - Before public deployment, review survey permissions, privacy, security, model governance, and external validation.
        """
    )
    st.subheader("Run locally")
    st.code("python -m pip install -r requirements.txt\npython -m streamlit run app.py", language="powershell")
