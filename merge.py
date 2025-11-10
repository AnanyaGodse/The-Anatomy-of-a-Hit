import pandas as pd

# Load shap values and corresponding features
shap_df = pd.read_csv("shap_values_sample.csv")        # each feature’s SHAP value
X_test = pd.read_csv("spotify_preprocessed.csv")  # your preprocessed features
preds = pd.read_csv("model_predictions.csv")            # contains predicted_label, predicted_proba, etc.

# Merge to align features and predictions
merged = X_test.copy()
merged["predicted_label"] = preds["predicted_label"]
merged["predicted_proba"] = preds["predicted_proba"]

# Add shap values (rename them to shap_feature)
for col in shap_df.columns:
    merged[f"shap_{col}"] = shap_df[col]

# Save for Tableau
merged.to_csv("shap_per_song_full.csv", index=False)
