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
import stat

# New for Kaggle API integration
import subprocess
import zipfile
import shutil

# Streamlit UI setup
st.title("🎾 Tennis Match Predictor")
st.markdown("This app uses ATP match data from 1981–2024 to predict match outcomes using Elo ratings and machine learning.")

# Helper: remove read-only flag to allow folder deletion
def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

# Set up Kaggle credentials from Streamlit secrets
if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
    os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
    os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

# Download full ATP match data from Jeff Sackmann if not already available
DATA_URL = "https://github.com/JeffSackmann/tennis_atp/archive/refs/heads/master.zip"
if not os.path.exists("atp_data") or not os.listdir("atp_data"):
    with st.spinner("Downloading ATP dataset (~15MB)..."):
        import requests, io
        r = requests.get(DATA_URL)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        temp_dir = "_temp_tennis_data"

        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, onerror=remove_readonly)
            except Exception as e:
                st.warning(f"Failed to clean temp directory: {e}")

        z.extractall(temp_dir)

        extracted_dir = os.path.join(temp_dir, "tennis_atp-master")
        if os.path.exists(extracted_dir):
            try:
                if os.path.exists("atp_data"):
                    shutil.rmtree("atp_data", onerror=remove_readonly)
                os.rename(extracted_dir, "atp_data")
            except Exception as e:
                st.error(f"Failed to move ATP data directory: {e}")
        else:
            st.error("ATP data folder not found in extracted zip!")
            st.stop()

        try:
            shutil.rmtree(temp_dir, onerror=remove_readonly)
        except Exception as e:
            st.warning(f"Could not fully delete temp folder (harmless): {e}")

# 🔄 Download Kaggle dataset via API if available
kaggle_dir = "kaggle_data"
kaggle_file = os.path.join(kaggle_dir, "atp_matches.csv")
if not os.path.exists(kaggle_file):
    try:
        st.info("Downloading updated ATP data from Kaggle...")
        os.makedirs(kaggle_dir, exist_ok=True)
        kaggle_dataset = "dissfya/atp-tennis-2000-2023daily-pull"
        subprocess.run(["kaggle", "datasets", "download", "-d", kaggle_dataset, "-p", kaggle_dir], check=True)
        zip_path = os.path.join(kaggle_dir, kaggle_dataset.split("/")[-1] + ".zip")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(kaggle_dir)
    except Exception as e:
        st.warning(f"Could not download from Kaggle: {e}")

# Load all ATP match files from 1981 to 2024
all_files = sorted(glob.glob(os.path.join("atp_data", "atp_matches_*.csv")))
all_files = [f for f in all_files if "doubles" not in f and "futures" not in f and "qual_chall" not in f]
all_files = [f for f in all_files if os.path.exists(f) and f[-8:-4].isdigit() and 1981 <= int(f[-8:-4]) <= 2024]

# Add Kaggle file if it exists
if os.path.exists(kaggle_file):
    all_files.append(kaggle_file)

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

st.success(f"Successfully loaded {len(df):,} match records.")

# Add Elo and helper features for prediction

def compute_elo(df, k=32):
    elo = {}
    surface_elo = {}
    matches = []

    for i, row in df.iterrows():
        w = row['winner_name']
        l = row['loser_name']
        s = row['surface']

        if pd.isnull(w) or pd.isnull(l) or pd.isnull(s):
            continue

        w_elo = elo.get(w, 1500)
        l_elo = elo.get(l, 1500)
        w_surface = surface_elo.get((w, s), 1500)
        l_surface = surface_elo.get((l, s), 1500)

        w_comb = 0.5 * w_elo + 0.5 * w_surface
        l_comb = 0.5 * l_elo + 0.5 * l_surface

        expected_win = 1 / (1 + 10 ** ((l_comb - w_comb) / 400))
        delta = k * (1 - expected_win)

        elo[w] = w_elo + delta
        elo[l] = l_elo - delta
        surface_elo[(w, s)] = w_surface + delta
        surface_elo[(l, s)] = l_surface - delta

    return elo, surface_elo

elo, surface_elo = compute_elo(df)

@st.cache_resource
def train_model(df, elo, surface_elo):
    df_train = df.dropna(subset=['winner_name', 'loser_name'])
    df_train = df_train[df_train['winner_name'] != df_train['loser_name']]

    df_train['elo_diff'] = df_train['winner_name'].apply(lambda x: elo.get(x, 1500)) - df_train['loser_name'].apply(lambda x: elo.get(x, 1500))
    df_train['surface_elo_diff'] = df_train.apply(lambda r: surface_elo.get((r['winner_name'], r['surface']), 1500) - surface_elo.get((r['loser_name'], r['surface']), 1500), axis=1)
    df_train['h2h_diff'] = df_train.apply(lambda r: df[(df['winner_name'] == r['winner_name']) & (df['loser_name'] == r['loser_name'])].shape[0] - df[(df['winner_name'] == r['loser_name']) & (df['loser_name'] == r['winner_name'])].shape[0], axis=1)

    model_df = df_train[['elo_diff', 'surface_elo_diff', 'h2h_diff']].copy()
    model_df['label'] = 1  # winner is first
    model_df = model_df.dropna()

    X = model_df.drop('label', axis=1)
    y = model_df['label']

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)

    return model

model = train_model(df, elo, surface_elo)

# Add single match prediction UI
st.header("🔮 Predict a Single Match")

players = sorted(set(df['winner_name'].dropna().unique()) | set(df['loser_name'].dropna().unique()))
player_a = st.selectbox("Select Player A", players, index=players.index("Novak Djokovic") if "Novak Djokovic" in players else 0)
player_b = st.selectbox("Select Player B", players, index=players.index("Carlos Alcaraz") if "Carlos Alcaraz" in players else 1)
surface = st.selectbox("Select Surface", ["Hard", "Clay", "Grass"])

if player_a == player_b:
    st.warning("Please select two different players.")
else:
    if st.button("Predict Winner"):
        features = {}
        features['elo_diff'] = elo.get(player_a, 1500) - elo.get(player_b, 1500)
        features['surface_elo_diff'] = surface_elo.get((player_a, surface), 1500) - surface_elo.get((player_b, surface), 1500)

        h2h_wins_a = df[(df['winner_name'] == player_a) & (df['loser_name'] == player_b)].shape[0]
        h2h_wins_b = df[(df['winner_name'] == player_b) & (df['loser_name'] == player_a)].shape[0]
        features['h2h_diff'] = h2h_wins_a - h2h_wins_b

        recent_a = df[(df['winner_name'] == player_a) | (df['loser_name'] == player_a)].sort_values('tourney_date', ascending=False).head(50)
        recent_b = df[(df['winner_name'] == player_b) | (df['loser_name'] == player_b)].sort_values('tourney_date', ascending=False).head(50)
        recent_a_wins = recent_a[recent_a['winner_name'] == player_a].shape[0]
        recent_b_wins = recent_b[recent_b['winner_name'] == player_b].shape[0]
        features['recent_win_diff'] = recent_a_wins - recent_b_wins

        X_input = pd.DataFrame([features])

        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0]

        predicted_winner = player_a if pred == 1 else player_b
        st.success(f"🏆 Predicted Winner: **{predicted_winner}**")
        st.write(f"Confidence: {prob[1 if pred == 1 else 0]*100:.2f}%")
