# app/tabular_explain_routes.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from app.utils.supabase import supabase

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

import json
import base64
import io
import os
from datetime import datetime
import pandas as pd

router = APIRouter()

# ---------------------------
# Paths (adjust if your layout differs)
# ---------------------------
TABULAR_MODEL_PATHS = {
    "heart": "models/tabular/heart.pkl",
    "diabetes": "models/tabular/diabetes.pkl",
}

# optional: training CSVs (used to infer feature order if not saved in bundle)
TABULAR_CSV_PATHS = {
    "heart": "models_training_full/tabular/heart/heart.csv",
    "diabetes": "models_training_full/tabular/diabetes/diabetes.csv",
}

TABULAR_BACKGROUND_PATHS = {
    "heart": "models/tabular/heart_background.npy",
    "diabetes": "models/tabular/diabetes_background.npy",
}

CLASS_LABELS = {
    "heart": ["No Heart Disease", "Has Heart Disease"],
    "diabetes": ["No Diabetes", "Diabetes"]
}

MEDICAL_EXPLANATIONS = {
    "heart": {
        "No Heart Disease": "Model predicts low risk of heart disease. Maintain routine healthy lifestyle and consult clinician if symptoms arise.",
        "Has Heart Disease": "Model predicts elevated risk of heart disease. Recommend cardiology follow-up, ECG and clinical correlation."
    },
    "diabetes": {
        "No Diabetes": "Low likelihood of diabetes based on provided features. Maintain healthy diet and follow-up as needed.",
        "Diabetes": "High probability of diabetes. Recommend blood glucose/HbA1c testing and clinical follow-up."
    }
}

_loaded_tabular_models = {}


# ---------------------------
# Utility: load saved bundle
# ---------------------------
def load_tabular_bundle(disease: str):
    if disease not in TABULAR_MODEL_PATHS:
        raise HTTPException(400, f"Unknown tabular disease '{disease}'")

    if disease in _loaded_tabular_models:
        return _loaded_tabular_models[disease]

    path = TABULAR_MODEL_PATHS[disease]
    if not os.path.exists(path):
        raise HTTPException(500, f"Tabular model file not found: {path}")

    try:
        bundle = joblib.load(path)
    except Exception as e:
        raise HTTPException(500, f"Failed to load model bundle: {e}")

    # expected bundle form: {"model":..., "scaler":..., optionally "feature_names": [...]}
    if not isinstance(bundle, dict) or "model" not in bundle or "scaler" not in bundle:
        raise HTTPException(500, "Model bundle format invalid. Expected dict with 'model' and 'scaler' keys.")

    _loaded_tabular_models[disease] = bundle
    return bundle


# ---------------------------
# Utility: infer feature order
# ---------------------------
def infer_feature_order(disease: str, bundle: dict, sample_keys: list):
    # 1) If bundle contains 'feature_names', use it
    if "feature_names" in bundle and isinstance(bundle["feature_names"], (list, tuple)):
        return list(bundle["feature_names"])

    # 2) Try training CSV
    csv_path = TABULAR_CSV_PATHS.get(disease)
    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # infer label column
            for label_col in ("target", "Outcome", "label", "Target"):
                if label_col in df.columns:
                    feature_names = [c for c in df.columns if c != label_col]
                    return feature_names
            # fallback: if sample keys match most columns, return df.columns as-is minus last column
            return [c for c in df.columns if c in sample_keys]
        except Exception:
            pass

    # 3) Fallback: use sample_keys order (best-effort)
    return list(sample_keys)


# ---------------------------
# Utility: fig -> base64
# ---------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------------------------
# Robust SHAP normalizer (handles many shapes)
# ---------------------------
def normalize_shap_values(shap_values, pred_index):
    """
    Normalize shap_values into a 1D array of length n_features corresponding to the
    predicted class. Accepts the various SHAP output shapes:
      - list/tuple (multiclass): list[class_index] -> (n_samples, n_features)
      - ndarray shapes seen in practice:
          (1, n_features)
          (n_features,)
          (1, n_features, 1)
          (1, n_features, 2)
          (n_features, 2)
          (n_features, )
          (n_samples, n_features, C)
    Returns: 1D numpy array of length n_features
    """
    arr = None

    # If list/tuple: multiclass typical case (list of arrays)
    if isinstance(shap_values, (list, tuple)):
        # pick class-specific array
        arr = np.array(shap_values[pred_index])
    else:
        arr = np.array(shap_values)

    # squeeze size-1 dims
    arr = np.squeeze(arr)

    # If now 1D, that's the per-feature vector
    if arr.ndim == 1:
        return arr.astype(float)

    # If 2D:
    if arr.ndim == 2:
        # possible shapes:
        # (n_features, n_classes)  -> choose column pred_index
        # (n_samples, n_features)  -> if n_samples==1, take row 0
        # (n_features, ) handled above
        r0, r1 = arr.shape
        # case: (n_features, n_classes)
        if r1 > 1 and r0 != 1 and r0 == arr.shape[0]:
            # arr is (n_features, n_classes) -> select column
            try:
                return arr[:, pred_index].astype(float)
            except Exception:
                pass
        # case: (n_samples, n_features)
        if r0 == 1:
            return arr[0].astype(float)
        # case: (n_samples, n_features) with r0>1 (multiple samples) -> take first sample
        return arr[0].astype(float)

    # If 3D (e.g., (1, n_features, C) or (n_samples, n_features, C))
    if arr.ndim == 3:
        # try sel = arr[0, :, pred_index] if last dim = classes
        if arr.shape[-1] > 1:
            try:
                return arr[0, :, pred_index].astype(float)
            except Exception:
                pass
        # fallback: reduce by squeezing and recursively normalize
        arr2 = np.squeeze(arr)
        if arr2.ndim == 2:
            return normalize_shap_values(arr2, pred_index)

    raise ValueError(f"SHAP reduced to unexpected shape {arr.shape}")


# ---------------------------
# Build SHAP PNGs (robust)
# ---------------------------
def build_shap_pngs(model, background, X_sample, feature_names, pred_index):
    # create explainer (prefer TreeExplainer for tree models)
    try:
        explainer = shap.TreeExplainer(model, data=background)
    except Exception:
        explainer = shap.Explainer(model, background)

    raw_shap = explainer.shap_values(X_sample)

    try:
        sv = normalize_shap_values(raw_shap, pred_index)  # per-feature 1D vector
    except Exception as e:
        raise RuntimeError(f"SHAP reduced to unexpected shape: {e}")

    # compute global mean |shap| (works for list or ndarray)
    if isinstance(raw_shap, (list, tuple)):
        global_sv = np.mean(np.abs(np.array(raw_shap[pred_index])), axis=0)
    else:
        arr = np.array(raw_shap)
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            global_sv = np.abs(arr)
        elif arr.ndim == 2:
            # maybe shape (n_samples, n_features) or (n_features, n_classes)
            if arr.shape[0] == 1:
                global_sv = np.mean(np.abs(arr), axis=0)
            else:
                # if features x classes, choose class column if exists else mean abs
                if arr.shape[1] > 1:
                    try:
                        global_sv = np.abs(arr[:, pred_index])
                    except Exception:
                        global_sv = np.mean(np.abs(arr), axis=0)
                else:
                    global_sv = np.mean(np.abs(arr), axis=0)
        else:
            # fallback
            global_sv = np.mean(np.abs(arr.reshape(-1, arr.shape[-1])), axis=0)

    global_sv = np.squeeze(global_sv)
    if global_sv.ndim != 1:
        # try to flatten safely
        global_sv = global_sv.flatten()

    feature_names = [str(f) for f in feature_names]

    # -------------------------
    # Feature importance plot
    # -------------------------
    order = np.argsort(global_sv)[::-1][:min(20, len(feature_names))]
    order = [int(i) for i in np.array(order).flatten()]

    fig1, ax1 = plt.subplots(figsize=(7, max(2, len(order) * 0.35) + 1))
    labels1 = [feature_names[i] for i in order][::-1]
    vals1 = [float(global_sv[i]) for i in order][::-1]
    ax1.barh(labels1, vals1)
    ax1.set_title("Feature Importance (mean |SHAP|)")
    ax1.set_xlabel("mean |SHAP value|")
    fig1.tight_layout()
    feat_imp_png = fig_to_base64(fig1)

    # -------------------------
    # Per-sample contributions (waterfall-like)
    # -------------------------
    w_order = np.argsort(np.abs(sv))[::-1][:min(20, len(feature_names))]
    w_order = [int(i) for i in np.array(w_order).flatten()]
    w_vals = [float(sv[i]) for i in w_order]
    w_labels = [feature_names[i] for i in w_order]
    colors = ["#10b981" if v > 0 else "#ef4444" for v in w_vals]

    fig2, ax2 = plt.subplots(figsize=(7, max(2, len(w_order) * 0.35) + 1))
    ax2.barh(w_labels[::-1], w_vals[::-1], color=colors[::-1])
    ax2.set_title("Per-sample SHAP Contributions")
    ax2.set_xlabel("SHAP value (impact on prediction)")
    fig2.tight_layout()
    waterfall_png = fig_to_base64(fig2)

    # -------------------------
    # Force-like sorted bar
    # -------------------------
    sorted_vals = np.sort(w_vals)
    sorted_colors = ["#10b981" if v > 0 else "#ef4444" for v in sorted_vals]
    fig3, ax3 = plt.subplots(figsize=(7, 3))
    ax3.barh(range(len(sorted_vals)), sorted_vals, color=sorted_colors)
    ax3.set_yticks(range(len(sorted_vals)))
    # reuse labels (showing top features)
    ax3.set_yticklabels(w_labels)
    ax3.set_title("Force-like SHAP Contributions")
    fig3.tight_layout()
    force_png = fig_to_base64(fig3)

    return feat_imp_png, waterfall_png, force_png


# ---------------------------
# Main route: /tabular/explain
# ---------------------------
@router.post("/explain")
async def explain_tabular(disease: str, sample: dict, user_id: str | None = None):
    """
    Expects:
      - disease (query param): 'heart' or 'diabetes'
      - sample (JSON body): mapping feature_name -> value (prefer training feature names/order)
    Returns an HTML report (interactive client-side)
    """
    # validate
    if disease not in TABULAR_MODEL_PATHS:
        raise HTTPException(400, "Unknown disease")

    if not isinstance(sample, dict):
        raise HTTPException(400, "Sample must be a JSON object mapping feature_name -> value")

    # load model bundle
    bundle = load_tabular_bundle(disease)
    model = bundle["model"]
    scaler = bundle["scaler"]

    # infer feature order (bundle may contain feature_names; else CSV; else sample order)
    feature_order = infer_feature_order(disease, bundle, list(sample.keys()))

    # Build ordered vector from feature_order
    try:
        ordered_values = [sample[f] for f in feature_order]
    except KeyError as e:
        # helpful error: missing feature
        raise HTTPException(400, f"Input missing required feature: {e}")

    X = np.array([ordered_values], dtype=float)

    # scale
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        # Some sklearn scalers warn about feature names; attempt transform with plain array
        try:
            X_scaled = scaler.transform(np.asarray(X))
        except Exception as ee:
            raise HTTPException(400, f"Feature scaling failed: {ee}") from e

    # predict
    try:
        probs = model.predict_proba(X_scaled)[0]
        pred_idx = int(model.predict(X_scaled)[0])
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    # load background for SHAP (fallback to X_scaled)
    bg_path = TABULAR_BACKGROUND_PATHS.get(disease)
    if bg_path and os.path.exists(bg_path):
        try:
            background = np.load(bg_path)
        except Exception:
            background = X_scaled
    else:
        background = X_scaled

    # build SHAP PNGs
    try:
        feat_imp_png, waterfall_png, force_png = build_shap_pngs(
            model, background, X_scaled, feature_order, pred_idx
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to build SHAP PNGs: {e}")

    class_names = CLASS_LABELS.get(disease, [f"class_{i}" for i in range(len(probs))])
    explanation_text = MEDICAL_EXPLANATIONS.get(disease, {}).get(class_names[pred_idx], "")
    normalized_disease = "heart_disease" if disease == "heart" else disease

    if user_id:
        supabase.table("predictions").insert({
            "user_id": user_id,
            "disease_type": normalized_disease,
            "prediction_result": {
                "predicted_class": pred_idx,
                "predicted_label": class_names[pred_idx],
                "input_features": dict(zip(feature_order, ordered_values)),
                "probabilities": [float(x) for x in probs],
            },
            "confidence_score": round(float(max(probs)) * 100, 2),
            "explainability_data": {
                "type": "shap",
                "available": True,
                "feature_order": feature_order,
                "explanation": explanation_text,
            },
        }).execute()

    # Inline HTML report (self-contained)
    html = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset='utf-8'>
        <title>DiagnostiX - Tabular SHAP Report</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
        <style>
          body{{font-family:Arial,Helvetica,sans-serif;background:#f7f8fb;margin:0;padding:20px;color:#0f172a}}
          .wrap{{max-width:1100px;margin:0 auto}}
          .card{{background:white;border-radius:12px;padding:16px;margin-top:12px;box-shadow:0 6px 18px rgba(11,22,50,0.06)}}
          img{{max-width:100%;height:auto;border-radius:8px}}
          pre{{background:#f3f6fb;padding:10px;border-radius:8px}}
          .class-item{{padding:8px;border-radius:8px;border:1px solid #eef2ff;margin-bottom:8px;cursor:pointer}}
          button{{padding:8px 12px;border-radius:8px;background:#2563eb;color:#fff;border:0;cursor:pointer}}
          .header{{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap}}
          .header-actions{{display:flex;gap:10px;align-items:center}}
          .subtle{{color:#475569}}
        </style>
      </head>
      <body>
        <div class='wrap'>
          <div class='card'>
            <div class="header">
              <div>
                <h2>DiagnostiX - Tabular SHAP Report</h2>
                <div>Disease: <b>{disease}</b> | Predicted: <b id="predLabel">{class_names[pred_idx]}</b> (class {pred_idx})</div>
                <div class="subtle">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
                <div style="margin-top:8px">Explanation: {explanation_text}</div>
              </div>
              <div class="header-actions">
                <button onclick="downloadPDF()">Download PDF</button>
              </div>
            </div>
          </div>

          <div class='card'>
            <h3>Input (ordered by training)</h3>
            <div class="small">Feature order: {feature_order}</div>
            <pre>{json.dumps(dict(zip(feature_order, ordered_values)), indent=2)}</pre>
          </div>

          <div class='card'>
            <h3>Predicted probabilities</h3>
            <pre>{json.dumps([float(x) for x in probs], indent=2)}</pre>
          </div>

          <div class='card'>
            <h3>Feature importance (mean |SHAP|)</h3>
            <img src='data:image/png;base64,{feat_imp_png}'/>
          </div>

          <div class='card'>
            <h3>Per-sample SHAP contributions</h3>
            <img src='data:image/png;base64,{waterfall_png}'/>
          </div>

          <div class='card'>
            <h3>Force-like contributions</h3>
            <img src='data:image/png;base64,{force_png}'/>
          </div>

        </div>
        <script>
          function downloadPDF() {{
            const element = document.querySelector('.wrap');
            const opt = {{
              margin: 0.4,
              filename: 'diagnostix-{normalized_disease}-report.pdf',
              image: {{ type: 'jpeg', quality: 0.98 }},
              html2canvas: {{ scale: 2, useCORS: true }},
              jsPDF: {{ unit: 'in', format: 'a4', orientation: 'portrait' }}
            }};

            html2pdf().set(opt).from(element).save();
          }}
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html)
