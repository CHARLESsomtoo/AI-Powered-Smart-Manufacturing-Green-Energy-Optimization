import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Powered Smart Manufacturing & Green Energy Optimization",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "project_outputs"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD COMPRESSED MODEL
# ============================================================

MODEL_PATH = MODELS_DIR / "predictive_maintenance_model_small.pkl"

model = None

if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(
            f"Could not load compressed model: {e}"
        )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_csv_file(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def find_file(folder, possible_names):

    if not folder.exists():
        return None

    for name in possible_names:
        file_path = folder / name

        if file_path.exists():
            return file_path

    for file_path in folder.iterdir():

        if file_path.is_file():

            file_name = file_path.name.lower()

            for name in possible_names:

                if file_name == name.lower():
                    return file_path

    return None


def get_machine_column(df):

    possible_columns = [
        "machine_id",
        "Machine_ID",
        "machine",
        "Machine"
    ]

    for column in possible_columns:

        if column in df.columns:
            return column

    return None


def get_numeric_column(df, possible_columns):

    for column in possible_columns:

        if column in df.columns:
            return column

    return None


def safe_value(df, column, default=0):

    if column in df.columns and len(df) > 0:
        return df[column].mean()

    return default


# ============================================================
# LOAD PROJECT FILES
# ============================================================

maintenance_file = find_file(
    OUTPUT_DIR,
    [
        "maintenance_profile.csv"
    ]
)

energy_profile_file = find_file(
    OUTPUT_DIR,
    [
        "energy_profile.csv"
    ]
)

model_comparison_file = find_file(
    OUTPUT_DIR,
    [
        "model_comparison.csv"
    ]
)

smart_manufacturing_file = find_file(
    DATA_DIR,
    [
        "smart_manufacturing_data.csv"
    ]
)

energy_data_file = find_file(
    DATA_DIR,
    [
        "Energy_dataset.csv",
        "energy_dataset.csv"
    ]
)


# ============================================================
# LOAD DATASETS
# ============================================================

maintenance_df = (
    load_csv_file(maintenance_file)
    if maintenance_file is not None
    else pd.DataFrame()
)

energy_profile_df = (
    load_csv_file(energy_profile_file)
    if energy_profile_file is not None
    else pd.DataFrame()
)

model_comparison_df = (
    load_csv_file(model_comparison_file)
    if model_comparison_file is not None
    else pd.DataFrame()
)

smart_manufacturing_df = (
    load_csv_file(smart_manufacturing_file)
    if smart_manufacturing_file is not None
    else pd.DataFrame()
)

energy_df = (
    load_csv_file(energy_data_file)
    if energy_data_file is not None
    else pd.DataFrame()
)


# ============================================================
# APP TITLE
# ============================================================

st.title(
    "AI-Powered Smart Manufacturing & Green Energy Optimization"
)

st.caption(
    "An AI-powered decision-support framework combining "
    "predictive maintenance and industrial energy optimization."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Executive Dashboard",
        "Predictive Maintenance",
        "Energy Optimization",
        "Recommendations"
    ]
)


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":

    st.header("Executive Dashboard")

    maintenance_rate = safe_value(
        maintenance_df,
        "av
