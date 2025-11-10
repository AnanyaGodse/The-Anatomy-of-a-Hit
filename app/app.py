from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

# initialize Flask app
app = Flask(__name__)

# load model, scaler, and label encoder
model = joblib.load("model/xgb_best_model.pkl")
scaler = joblib.load("model/scaler.pkl")
le = joblib.load("model/label_encoder.pkl")

# get all genre labels for dropdown
genres = list(le.classes_)

# numeric columns to scale
numeric_cols = [
    'danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness','valence', 'tempo', 'duration_min',
    'energy_dance_ratio', 'loudness_norm'
]

@app.route('/')
def home():
    return render_template('index.html', genres=genres)

@app.route('/predict', methods=['POST'])
def predict():
    # read inputs from form
    data = {
        'track_genre': request.form['track_genre'],
        'explicit': int(request.form['explicit']),
        'danceability': float(request.form['danceability']),
        'energy': float(request.form['energy']),
        'loudness': float(request.form['loudness']),
        'speechiness': float(request.form['speechiness']),
        'acousticness': float(request.form['acousticness']),
        'instrumentalness': float(request.form['instrumentalness']),
        'valence': float(request.form['valence']),
        'tempo': float(request.form['tempo']),
        'duration_min': float(request.form['duration_min'])
    }

    # derived features
    data['energy_dance_ratio'] = data['energy'] / (data['danceability'] + 1e-6)
    data['loudness_norm'] = (data['loudness'] - (-60)) / (0 - (-60))  # assuming loudness in [-60,0]
    data['tempo_bin'] = pd.qcut([data['tempo']], 5, labels=False, duplicates='drop')[0]
    data['track_genre_encoded'] = le.transform([data['track_genre']])[0]
    data['key'] = 5  # placeholder 
    data['mode'] = 1
    data['liveness'] = 0.2
    data['time_signature'] = 4

    # make into DataFrame
    X_input = pd.DataFrame([data])

    # drop track_genre
    X_input = X_input.drop(columns=['track_genre'])
    
    # scale numeric columns
    X_input[numeric_cols] = scaler.transform(X_input[numeric_cols])

    feature_order = [
        'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
        'valence', 'tempo', 'time_signature', 'duration_min', 'energy_dance_ratio',
        'loudness_norm', 'tempo_bin', 'track_genre_encoded']

    X_input = X_input[feature_order]

    # predict
    y_proba = model.predict_proba(X_input)[:, 1][0]
    threshold = 0.611
    result = "🌟 Likely a Hit!" if y_proba >= threshold else "💤 Probably Not a Hit"

    return render_template(
        'index.html',
        genres=genres,
        prediction_text=f"{result} (Probability: {y_proba:.2f})"
    )

if __name__ == "__main__":
    app.run(debug=True)
