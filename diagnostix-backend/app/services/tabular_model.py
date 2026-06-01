import joblib
import numpy as np

def load_tabular_model(path: str):
    return joblib.load(path)

def convert_to_np(data: dict, feature_order: list):
    arr = [data.get(feat, 0) for feat in feature_order]
    return np.array(arr).reshape(1, -1)
