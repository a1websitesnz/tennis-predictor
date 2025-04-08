# Tennis Match Predictor with Elo Ratings, Head-to-Head, XGBoost + Web App + Tennis API

import pandas as pd
import numpy as np
import glob
import os
import joblib
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import stat
import time
import re
import fitz  # PyMuPDF for PDF parsing
import unidecode  # For fuzzy matching player names

# Streamlit UI setup
st.title("🎾 Tennis Match Predictor")
st.markdown("This app uses ATP match data from 1981–2024 to predict match outcomes using Elo ratings, machine learning, and live ranking data.")

# Helper: Normalize names for fuzzy matching
def normalize_name(name):
    return unidecode.unidecode(name.lower().replace(".", "").replace("-", "").strip())

# Load live top 300 ATP players from RapidAPI: tennis-api-atp-wta-itf
@st.cache_data(ttl=86400)
def fetch_top_300_from_api():
    url = "https://tennis-api-atp-wta-itf.p.rapidapi.com/tennis/v2/atp/ranking/singles/"
    headers = {
        "X-RapidAPI-Key": st.secrets["RAPIDAPI_KEY"],
        "X-RapidAPI-Host": "tennis-api-atp-wta-itf.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.warning("⚠️ Failed to fetch rankings from API. Falling back.")
        return []

    try:
        data = response.json()

        if isinstance(data, dict) and "data" in data and "rankings" in data["data"]:
            players_raw = data["data"]["rankings"][:300]
        elif isinstance(data, dict) and "response" in data:
            players_raw = data["response"][:300]
        elif isinstance(data, list):
            players_raw = data[:300]
        else:
            st.error("⚠️ Unexpected API format.")
            return []

        return [f"{p['lastname']} {p['firstname'][0]}." for p in players_raw if 'firstname' in p and 'lastname' in p]

    except Exception as e:
        st.error(f"Error parsing API response: {e}")
        return []

manual_top_300_names = fetch_top_300_from_api()

# Debug: export top 300 list to file
with open("debug_top_300_names.txt", "w") as debug_f:
    for name in manual_top_300_names:
        debug_f.write(name + "\n")

# 🔄 Load Kaggle dataset manually if uploaded
kaggle_dir = "kaggle_data"
kaggle_file = os.path.join(kaggle_dir, "atp_tennis.csv")

# Define expected columns for the raw CSV (no header)
kaggle_columns = [
    "tourney_name", "tourney_date", "tourney_level", "Court", "surface", "Round", "best_of",
    "player1", "player2", "winner_name", "winner_rank", "loser_rank", "winner_seed", "loser_seed",
    "winner_elo", "loser_elo", "score"
]

kaggle_df = pd.read_csv(kaggle_file, names=kaggle_columns, header=None, low_memory=False) if os.path.exists(kaggle_file) else pd.DataFrame()

# If loser_name is missing, infer it from player1/player2/winner_name
if "winner_name" in kaggle_df.columns and "player1" in kaggle_df.columns and "player2" in kaggle_df.columns:
    kaggle_df["loser_name"] = kaggle_df.apply(lambda row: row["player2"] if row["winner_name"] == row["player1"] else row["player1"], axis=1)

# Load and combine ATP data
all_files = sorted(glob.glob(os.path.join("atp_data", "atp_matches_*.csv")))

# Read additional match data
dfs = []
for f in all_files:
    try:
        dfs.append(pd.read_csv(f, low_memory=False))
    except Exception as e:
        st.warning(f"Skipping {f}: {e}")

if kaggle_df.shape[0] > 0:
    dfs.append(kaggle_df)

if not dfs:
    st.error("No valid match data found.")
    st.stop()

df = pd.concat(dfs, ignore_index=True)
st.success(f"✅ Successfully loaded {len(df):,} match records.")

# --- Filter active top 300 players only ---
df_players = pd.unique(df[['winner_name', 'loser_name']].values.ravel())
df_players_clean = [p.strip() for p in df_players if isinstance(p, str)]

normalized_top_300 = {normalize_name(name): name for name in manual_top_300_names}
normalized_dataset_players = {normalize_name(name): name for name in df_players_clean}

matched_names = sorted(set(normalized_top_300.keys()) & set(normalized_dataset_players.keys()))
filtered_top_players = [normalized_dataset_players[name] for name in matched_names]

if not filtered_top_players:
    st.warning("⚠️ No active top 300 players found in dataset after matching.")
else:
    st.info("✅ Active top 300 players filtered and matched to dataset.")

# --- Streamlit UI for Player Selection ---
col1, col2 = st.columns(2)
with col1:
    player1 = st.selectbox("Select Player 1", options=filtered_top_players)
with col2:
    player2 = st.selectbox("Select Player 2", options=filtered_top_players)

surface = st.selectbox("Surface", options=df["surface"].dropna().unique())
level = st.selectbox("Tournament Level", options=df["tourney_level"].dropna().unique())

if st.button("Predict Winner"):
    match_data = df[
        (df["winner_name"].isin([player1, player2])) &
        (df["loser_name"].isin([player1, player2])) &
        (df["surface"] == surface) &
        (df["tourney_level"] == level)
    ]

    if match_data.empty:
        st.warning("No past matches found between selected players on specified surface and level.")
    else:
        match_data["elo_diff"] = match_data["winner_elo"] - match_data["loser_elo"]
        mean_diff = match_data["elo_diff"].mean()

        predicted_winner = player1 if mean_diff > 0 else player2
        st.markdown(f"### 🧠 Predicted Winner: **{predicted_winner}**")

        st.dataframe(match_data[["winner_name", "loser_name", "surface", "tourney_level", "winner_elo", "loser_elo", "score"]])
