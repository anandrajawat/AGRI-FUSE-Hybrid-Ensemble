"""
Data Loading and Artifact Helper for AGRI-FUSE Streamlit Dashboard.
Caches and safely loads reports, dataset previews, and evaluation metrics.
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st

REPORTS_DIR = "agri-fuse/reports"
METRICS_DIR = "agri-fuse/artifacts/metrics"
RAW_DIR = "agri-fuse/data/raw"
PLOTS_DIR = "agri-fuse/artifacts/plots"

@st.cache_data
def load_inventory():
    path = f"{REPORTS_DIR}/dataset_inventory.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_quality_report():
    path = f"{REPORTS_DIR}/dataset_quality_report.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_feature_dictionary():
    path = f"{REPORTS_DIR}/feature_dictionary.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_compatibility_matrix():
    path = f"{REPORTS_DIR}/dataset_compatibility_matrix.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_task_metrics(task_file):
    path = f"{METRICS_DIR}/{task_file}"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_raw_dataset(key):
    paths = {
        "D1": f"{RAW_DIR}/crop_recommendation/Crop_recommendation.csv",
        "D2": f"{RAW_DIR}/indian_crop_yield/crop_yield.csv",
        "D3": f"{RAW_DIR}/eastern_wheat/Wheat_Yld_Eastern_IND_Data.xlsx",
        "D4": f"{RAW_DIR}/icrisat/icrisat_district_level_1990_2015.xls",
        "D5": f"{RAW_DIR}/sf24/Crop_recommendationV2.csv",
        "D6": f"{RAW_DIR}/smart_farming_iot/Smart_Farming_Crop_Yield_2024.csv",
        "D7": f"{RAW_DIR}/soil_nutrients/dataset.csv",
        "D9": f"{RAW_DIR}/agriyield/train.csv"
    }
    path = paths.get(key)
    if not path or not os.path.exists(path):
        return None
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)
    return None

