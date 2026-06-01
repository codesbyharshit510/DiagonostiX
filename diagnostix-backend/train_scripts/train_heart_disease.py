import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

BASE = os.path.dirname(os.path.dirname(__file__))

csv_path = os.path.join(BASE, "models_training_full", "tabular", "heart", "heart.csv")
model_path = os.path.join(BASE, "models", "tabular", "heart.pkl")
background_path = os.path.join(BASE, "models", "tabular", "heart_background.npy")

os.makedirs(os.path.dirname(model_path), exist_ok=True)

print("📄 Loading dataset:", csv_path)
df = pd.read_csv(csv_path)

X = df.drop("target", axis=1)
y = df["target"]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

acc = model.score(X_test_scaled, y_test)
print(f"✅ Heart Disease Model Accuracy: {acc}")

# Save model
joblib.dump({"model": model, "scaler": scaler}, model_path)
print("💾 Model saved at:", model_path)

# Save SHAP background data
background = X_train_scaled[np.random.choice(X_train_scaled.shape[0], 100, replace=False)]
np.save(background_path, background)
print("💾 SHAP background saved at:", background_path)
