# Tennis Match Predictor with Elo Ratings, Head-to-Head, XGBoost + Web App

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

# Streamlit UI setup
st.title("🎾 Tennis Match Predictor")
st.markdown("This app uses ATP match data from 1981–2024 to predict match outcomes using Elo ratings and machine learning.")

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

# 📥 Download and parse ATP current numerical rankings PDF
@st.cache_data(ttl=86400)
def extract_top_300_names_from_official_pdf():
    url = "https://www.protennislive.com/posting/ramr/singles_entry_numerical.pdf"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Failed to download official ATP rankings PDF.")
        return []

    with open("singles_entry_numerical.pdf", "wb") as f:
        f.write(response.content)

    top_300_names = []
    doc = fitz.open("singles_entry_numerical.pdf")

    for page in doc:
        text = page.get_text()
        for line in text.split("\n"):
            match = re.match(r"^\s*(\d{1,3})\s+([A-Za-z'\-]+)\s+([A-Za-z]+)\s+\(\w{3}\)", line)
            if match:
                last, first = match.groups()[1:3]
                full = f"{last} {first[0]}."
                top_300_names.append(full)
            if len(top_300_names) >= 300:
                break
        if len(top_300_names) >= 300:
            break

    doc.close()
    return top_300_names

manual_top_300_names = extract_top_300_names_from_official_pdf()

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

# 🛠 Ensure required columns exist before filtering
required_cols = ["winner_name", "loser_name", "surface", "tourney_level"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"Missing required columns in dataset: {', '.join(missing_cols)}")
    st.stop()

# Basic preprocessing for modeling
df = df[df["winner_name"].notnull() & df["loser_name"].notnull()]
df = df[(df["surface"].notnull()) & (df["tourney_level"].notnull())]

# Normalize player names for fuzzy matching
def normalize_for_matching(name):
    if pd.isna(name): return ""
    return re.sub(r'[^a-z0-9]', '', name.lower().strip())

all_players = sorted(set(df['winner_name']).union(set(df['loser_name'])))
all_players = [p.strip() for p in all_players if isinstance(p, str) and p.strip() != ""]
norm_map = {normalize_for_matching(p): p for p in all_players}

# Match top 300 players to dataset
top_300_clean = [normalize_for_matching(p) for p in manual_top_300_names]
matched_players = [norm_map[n] for n in norm_map if n in top_300_clean]
players = sorted(set(matched_players))

if not players:
    st.warning("⚠️ No active top 300 players found in dataset after fuzzy matching.")
    st.write("🪪 Sample from match dataset:", all_players[:10])
    st.write("📊 Sample top 300 list:", manual_top_300_names[:10])
    st.stop()

def_idx1 = 0
while def_idx1 < len(players) and (players[def_idx1] == "" or pd.isna(players[def_idx1])):
    def_idx1 += 1

def_idx2 = def_idx1 + 1 if def_idx1 + 1 < len(players) else 0

player1 = st.selectbox("Select Player 1", players, index=def_idx1)
player2 = st.selectbox("Select Player 2", players, index=def_idx2)

# Match context inputs
surface = st.selectbox("Surface", df['surface'].dropna().unique())
tourney_level = st.selectbox("Tournament Level", df['tourney_level'].dropna().unique())

if st.button("Predict Winner"):
    match_data = df[(df["winner_name"].isin([player1, player2])) & (df["loser_name"].isin([player1, player2]))]
    if match_data.empty:
        st.warning("No head-to-head data between selected players.")
    else:
        match_data["p1_win"] = match_data["winner_name"] == player1

        match_data["elo_diff"] = match_data["winner_elo"] - match_data["loser_elo"]
        match_data["rank_diff"] = match_data["loser_rank"] - match_data["winner_rank"]
        match_data["seed_diff"] = match_data["loser_seed"] - match_data["winner_seed"]

        features = match_data[["surface", "tourney_level", "elo_diff", "rank_diff", "seed_diff"]]
        features = pd.get_dummies(features)
        X = features
        y = match_data["p1_win"]

        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X, y)

        test_input = pd.DataFrame([[surface, tourney_level, 0, 0, 0]], columns=["surface", "tourney_level", "elo_diff", "rank_diff", "seed_diff"])
        test_input = pd.get_dummies(test_input).reindex(columns=X.columns, fill_value=0)

        prediction = model.predict(test_input)[0]
        proba = model.predict_proba(test_input)[0][int(prediction)]

        winner = player1 if prediction else player2
        st.success(f"🎯 Predicted Winner: **{winner}** with {proba*100:.2f}% confidence")
