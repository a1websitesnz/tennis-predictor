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
st.markdown("Upload your ATP matches dataset or use the full 1981–2024 dataset.")

# Load all ATP match files from 1981 to 2024
data_dir = "./atp_data"
all_files = sorted(glob.glob(os.path.join(data_dir, "atp_matches_*.csv")))
all_files = [f for f in all_files if "1968" not in f and int(os.path.basename(f)[12:16]) >= 1981]

# Combine all datasets
df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

# Keep relevant columns
df = df[['winner_name', 'loser_name', 'surface', 'winner_rank', 'loser_rank',
         'winner_age', 'loser_age', 'winner_ht', 'loser_ht']].dropna()

# Elo ratings stored by player and surface
elo_dict = {'all': {}, 'Clay': {}, 'Hard': {}, 'Grass': {}, 'Carpet': {}}

# Head-to-head win tracking
h2h_wins = {}

def get_elo(player, surface):
    return elo_dict[surface].get(player, 1500)

def update_elo(w_elo, l_elo, k=32):
    expected = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
    return w_elo + k * (1 - expected), l_elo - k * (1 - expected)

def get_h2h(winner, loser):
    return h2h_wins.get((winner, loser), 0) - h2h_wins.get((loser, winner), 0)

def update_elos(row):
    surface = row['surface']
    w, l = row['winner_name'], row['loser_name']
    w_elo = get_elo(w, surface)
    l_elo = get_elo(l, surface)
    new_w_elo, new_l_elo = update_elo(w_elo, l_elo)
    elo_dict[surface][w] = new_w_elo
    elo_dict[surface][l] = new_l_elo
    h2h_wins[(w, l)] = h2h_wins.get((w, l), 0) + 1
    return pd.Series({'winner_elo': new_w_elo, 'loser_elo': new_l_elo, 'h2h_diff': get_h2h(w, l)})

elo_h2h = df.apply(update_elos, axis=1)
df = df.join(elo_h2h)

# Feature engineering
df['elo_diff'] = df['winner_elo'] - df['loser_elo']
df['rank_diff'] = df['winner_rank'] - df['loser_rank']
df['age_diff'] = df['winner_age'] - df['loser_age']
df['ht_diff'] = df['winner_ht'] - df['loser_ht']
df['surface_code'] = df['surface'].astype('category').cat.codes
df['label'] = 1

# Flip rows to augment data
df_flip = df.copy()
df_flip['elo_diff'] *= -1
df_flip['rank_diff'] *= -1
df_flip['age_diff'] *= -1
df_flip['ht_diff'] *= -1
df_flip['h2h_diff'] *= -1
df_flip['label'] = 0

# Combine and shuffle
df_all = pd.concat([df, df_flip], ignore_index=True).sample(frac=1, random_state=42)

# Train/test split
features = ['elo_diff', 'rank_diff', 'age_diff', 'ht_diff', 'h2h_diff', 'surface_code']
X = df_all[features]
y = df_all['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

# Save model
joblib.dump(model, "xgb_tennis_model.pkl")

# Show accuracy
st.success(f"✅ Model Accuracy: {acc * 100:.2f}%")

# Plot feature importance
importances = model.feature_importances_
fig, ax = plt.subplots()
sns.barplot(x=importances, y=features, ax=ax)
ax.set_title("Feature Importances")
st.pyplot(fig)

# Match predictor UI
st.subheader("🎯 Predict Match Outcome")
player1_elo = st.number_input("Player 1 Elo", 1200, 3000, 1500)
player2_elo = st.number_input("Player 2 Elo", 1200, 3000, 1500)
rank_diff = st.number_input("Ranking Difference (P1 - P2)", -1000, 1000, 0)
age_diff = st.number_input("Age Difference (P1 - P2)", -30, 30, 0.0)
ht_diff = st.number_input("Height Difference (P1 - P2)", -50, 50, 0)
h2h_diff = st.number_input("Head-to-Head Difference (P1 - P2)", -20, 20, 0)
surface_input = st.selectbox("Surface", ['Hard', 'Clay', 'Grass', 'Carpet'])
surface_code = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}[surface_input]

if st.button("Predict Winner"):
    input_df = pd.DataFrame([[player1_elo - player2_elo, rank_diff, age_diff, ht_diff, h2h_diff, surface_code]], columns=features)
    model = joblib.load("xgb_tennis_model.pkl")
    prediction = model.predict(input_df)[0]
    st.markdown(f"### 🏆 Predicted Winner: {'Player 1' if prediction == 1 else 'Player 2'}")
