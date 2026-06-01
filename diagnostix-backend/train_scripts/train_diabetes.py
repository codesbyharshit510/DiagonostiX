import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

BASE = os.path.dirname(os.path.dirname(__file__))

csv_path = os.path.join(BASE, "models_training_full", "tabular", "diabetes", "diabetes.csv")
model_path = os.path.join(BASE, "models", "tabular", "diabetes.pkl")
background_path = os.path.join(BASE, "models", "tabular", "diabetes_background.npy")

os.makedirs(os.path.dirname(model_path), exist_ok=True)

print("📄 Loading dataset:", csv_path)
df = pd.read_csv(csv_path)

# Auto-detect the target column
target_col = None
for col in ["Outcome", "target", "label", "Target"]:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    raise Exception("❌ Target column not found. Rename your label column to 'Outcome' or 'target'.")

X = df.drop(target_col, axis=1)
y = df[target_col]

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
print(f"✅ Diabetes Model Accuracy: {acc}")

# Save model
joblib.dump({"model": model, "scaler": scaler}, model_path)
print("💾 Model saved at:", model_path)

# Save SHAP background data
background = X_train_scaled[np.random.choice(X_train_scaled.shape[0], 100, replace=False)]
np.save(background_path, background)
print("💾 SHAP background saved at:", background_path)
