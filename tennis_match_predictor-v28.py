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
import time

# Streamlit UI setup
st.title("🎾 Tennis Match Predictor")
st.markdown("This app uses ATP match data from 1981–2024 to predict match outcomes using Elo ratings and machine learning.")

# 🔄 Load Kaggle dataset manually if uploaded
kaggle_dir = "kaggle_data"
kaggle_file = os.path.join(kaggle_dir, "atp_tennis.csv")

kaggle_df = pd.read_csv(kaggle_file, low_memory=False) if os.path.exists(kaggle_file) else pd.DataFrame()

# ✅ Only rename columns if data is loaded
if not kaggle_df.empty:
    rename_map = {
        "Winner": "winner_name",
        "Loser": "loser_name",
        "Surface": "surface",
        "Tournament": "tourney_name",
        "Date": "tourney_date",
        "Series": "tourney_level"
    }
    kaggle_df.rename(columns=rename_map, inplace=True)

# Load and combine ATP data
all_files = sorted(glob.glob(os.path.join("atp_data", "atp_matches_*.csv")))

# 🎯 Load current players from Jeff Sackmann's ranking file
ranking_file = os.path.join("atp_data", "atp_rankings_current.csv")
current_players = set()
top_300_names = []
if os.path.exists(ranking_file):
    rankings_df = pd.read_csv(ranking_file, header=None, names=["ranking_date", "rank", "player_id", "points"])
    rankings_df["rank"] = pd.to_numeric(rankings_df["rank"], errors="coerce")
    rankings_df = rankings_df.dropna(subset=["rank"])
    rankings_df = rankings_df[rankings_df['rank'] <= 300]
    player_id_name_map = pd.read_csv(os.path.join("atp_data", "atp_players.csv"), header=None, names=["player_id", "first_name", "last_name", "hand", "birth_date", "country_code"])
    player_id_name_map["full_name"] = player_id_name_map["first_name"].fillna("").astype(str) + " " + player_id_name_map["last_name"].fillna("").astype(str)
    merged = pd.merge(rankings_df, player_id_name_map, on="player_id")
    current_players = set(merged["full_name"].dropna().astype(str).unique())
    top_300_names = list(merged.sort_values("rank")["full_name"])

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

# Player select inputs using current top 300 players only
all_players = sorted(set(df['winner_name']).union(set(df['loser_name'])))
all_players = [p for p in all_players if isinstance(p, str) and p.strip() != ""]
players = [p for p in top_300_names if p in all_players]

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

        features = match_data[["surface", "tourney_level"]]
        features = pd.get_dummies(features)
        X = features
        y = match_data["p1_win"]

        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X, y)

        test_input = pd.DataFrame([[surface, tourney_level]], columns=["surface", "tourney_level"])
        test_input = pd.get_dummies(test_input).reindex(columns=X.columns, fill_value=0)

        prediction = model.predict(test_input)[0]
        proba = model.predict_proba(test_input)[0][int(prediction)]

        winner = player1 if prediction else player2
        st.success(f"🎯 Predicted Winner: **{winner}** with {proba*100:.2f}% confidence")
