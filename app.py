# Tennis Match Predictor with Elo Ratings, Head-to-Head, XGBoost + Web App

import pandas as pd
import numpy as np
import glob
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Streamlit UI setup
st.title("🎾 Tennis Match Predictor")
st.markdown("This app uses ATP match data from 1981–2024 to predict match outcomes using Elo ratings and machine learning.")

# Download full ATP match data if not already available
DATA_URL = "https://github.com/JeffSackmann/tennis_atp/archive/refs/heads/master.zip"

if not os.path.exists("atp_data"):
    with st.spinner("Downloading ATP dataset (~15MB)..."):
        import zipfile, requests, io, shutil
        r = requests.get(DATA_URL)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        temp_dir = "_temp_tennis_data"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        z.extractall(temp_dir)
        if os.path.exists(os.path.join(temp_dir, "tennis_atp-master")):
            if os.path.exists("atp_data"):
                shutil.rmtree("atp_data")
            os.rename(os.path.join(temp_dir, "tennis_atp-master"), "atp_data")
        shutil.rmtree(temp_dir)

# Load all ATP match files from 1981 to 2024
all_files = sorted(glob.glob(os.path.join("atp_data", "atp_matches_*.csv")))
all_files = [f for f in all_files if f[-8:-4].isdigit() and 1981 <= int(f[-8:-4]) <= 2024]

if not all_files:
    st.error("ATP match files not found after download. Please try again later.")
    st.stop()

# Combine all datasets
valid_files = []
for f in all_files:
    try:
        df_check = pd.read_csv(f, nrows=1)
        valid_files.append(f)
    except Exception as e:
        st.warning(f"Skipping {f}: {e}")

if not valid_files:
    st.error("No valid CSV files found for loading.")
    st.stop()

# Load valid data
df = pd.concat([pd.read_csv(f) for f in valid_files], ignore_index=True)

# Keep relevant columns
# (This is a placeholder for actual feature engineering and prediction code)
st.success(f"Successfully loaded {len(df):,} match records.")
