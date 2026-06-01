import joblib
import numpy as np
from fastapi import APIRouter, HTTPException
from app.utils.supabase import supabase

router = APIRouter(prefix="/tabular", tags=["Tabular Models"])

# -------------------------------------------------------------------
#  MODEL PATHS
# -------------------------------------------------------------------

TABULAR_MODEL_PATHS = {
    "heart_disease": "models/tabular/heart.pkl",
    "diabetes": "models/tabular/diabetes.pkl"
}

_loaded_models = {}  # cache


# -------------------------------------------------------------------
#  LOAD MODEL
# -------------------------------------------------------------------

def load_tabular_model(disease: str):
    if disease not in TABULAR_MODEL_PATHS:
        raise HTTPException(400, f"Unknown tabular disease '{disease}'")

    if disease in _loaded_models:
        return _loaded_models[disease]

    try:
        bundle = joblib.load(TABULAR_MODEL_PATHS[disease])
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {e}")

    if "model" not in bundle or "scaler" not in bundle:
        raise HTTPException(
            500,
            "Invalid model bundle. Expected {'model': ..., 'scaler': ...}"
        )

    _loaded_models[disease] = bundle
    return bundle


# -------------------------------------------------------------------
#  PREDICT (USED BY FRONTEND)
# -------------------------------------------------------------------

@router.post("/predict")
def predict_tabular(
    disease: str,
    sample: dict,
    user_id: str | None = None
):
    """
    Predict using a trained tabular ML model.
    Saves prediction history to Supabase.
    """

    bundle = load_tabular_model(disease)
    model = bundle["model"]
    scaler = bundle["scaler"]

    try:
        # Convert input dict → numpy array
        X = np.array([list(sample.values())], dtype=float)

        # Scale
        X_scaled = scaler.transform(X)

        # Prediction
        pred = int(model.predict(X_scaled)[0])

        # Probability
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[0].tolist()
            confidence = round(max(probs) * 100, 2)
        else:
            probs = None
            confidence = None

    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    # ----------------------------------------------------------------
    #  SAVE TO SUPABASE
    # ----------------------------------------------------------------
    if user_id:
        supabase.table("predictions").insert({
            "user_id": user_id,
            "disease_type": disease,
            "prediction_result": {
                "predicted_class": pred,
                "input_features": sample,
                "probabilities": probs
            },
            "confidence_score": confidence,
            "explainability_data": None
        }).execute()

    return {
        "disease": disease,
        "prediction": pred,
        "probabilities": probs,
        "confidence": confidence
    }