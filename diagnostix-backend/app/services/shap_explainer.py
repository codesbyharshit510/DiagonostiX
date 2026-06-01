import shap
import numpy as np

def create_shap_explainer(model, background_data):
    # If background_data is None, SHAP will try to use model-specific defaults,
    # but it's recommended to provide a small sample from your training data.
    if background_data is None:
        try:
            return shap.Explainer(model)
        except Exception:
            return None
    # Prefer TreeExplainer for tree models
    try:
        expl = shap.TreeExplainer(model, data=background_data)
    except Exception:
        expl = shap.Explainer(model, masker=background_data)
    return expl

def get_shap_values(explainer, sample):
    if explainer is None:
        return None
    vals = explainer(sample)
    # vals.values -> numpy array or list depending on model
    return vals.values if hasattr(vals, "values") else vals
