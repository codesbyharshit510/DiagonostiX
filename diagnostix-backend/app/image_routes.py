from app.utils.supabase import supabase
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import torch
import base64

from app.services.image_model import load_image_model, preprocess_image
from app.services.gradcam_custom import generate_gradcam


router = APIRouter(prefix="/image", tags=["Image Models"])


# -------------------------------------------------------------------
#  MODEL CONFIGURATION
# -------------------------------------------------------------------

MODEL_PATHS = {
    "pneumonia": "models/image/pneumonia.pt",
    "alzheimers": "models/image/alzheimers.pt",
    "brain_tumor": "models/image/brain_tumor.pt",
}

NUM_CLASSES = {
    "pneumonia": 2,
    "alzheimers": 4,
    "brain_tumor": 4,
}

_loaded_models = {}


# -------------------------------------------------------------------
#  LOAD MODEL
# -------------------------------------------------------------------

def get_model(name: str):
    if name not in MODEL_PATHS:
        raise HTTPException(
            400, f"Unknown disease '{name}'. Valid: {list(MODEL_PATHS.keys())}"
        )

    if name in _loaded_models:
        return _loaded_models[name]

    try:
        model = load_image_model(MODEL_PATHS[name], NUM_CLASSES[name])
        model.eval()
    except Exception as e:
        raise HTTPException(500, f"Model loading failed: {e}")

    _loaded_models[name] = model
    return model


# -------------------------------------------------------------------
#  PREDICT (API ONLY)
# -------------------------------------------------------------------

@router.post("/predict")
async def predict_image(
    disease: str,
    file: UploadFile = File(...),
    user_id: str | None = None
):
    img_bytes = await file.read()
    tensor = preprocess_image(img_bytes)

    model = get_model(disease)

    with torch.no_grad():
        output = model(tensor)
        if output.dim() != 2:
            raise HTTPException(500, "Invalid model output")

        probs = torch.softmax(output, dim=1)[0].cpu().numpy().tolist()
        pred_class = int(torch.argmax(output, dim=1).item())
        confidence = round(max(probs) * 100, 2)

    # -------------------------------
    # SAVE TO SUPABASE
    # -------------------------------
    if user_id:
        supabase.table("predictions").insert({
            "user_id": user_id,
            "disease_type": disease,
            "prediction_result": {
                "predicted_class": pred_class,
                "probabilities": probs
            },
            "confidence_score": confidence,
            "explainability_data": None
        }).execute()

    return {
        "disease": disease,
        "predicted_class": pred_class,
        "probabilities": probs,
        "confidence": confidence
    }


# -------------------------------------------------------------------
#  GRAD-CAM (API ONLY)
# -------------------------------------------------------------------

@router.post("/gradcam")
async def gradcam_image(
    disease: str,
    file: UploadFile = File(...),
    target_class: int | None = None
):
    img_bytes = await file.read()
    tensor = preprocess_image(img_bytes)

    model = get_model(disease)

    with torch.no_grad():
        out = model(tensor)
        pred_class = int(out.argmax(1).item())

    chosen_class = target_class if target_class is not None else pred_class

    try:
        target_layer = model.layer4[-1]
    except Exception:
        convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
        target_layer = convs[-1]

    gradcam_b64 = generate_gradcam(model, tensor, target_layer, chosen_class)

    return {
        "disease": disease,
        "predicted_class": pred_class,
        "gradcam_base64": gradcam_b64,
    }


# -------------------------------------------------------------------
#  FULL HTML REPORT (FRONTEND)
# -------------------------------------------------------------------

@router.post("/report")
async def image_report(
    disease: str,
    file: UploadFile = File(...),
    user_id: str | None = None
):
    img_bytes = await file.read()
    tensor = preprocess_image(img_bytes)

    model = get_model(disease)

    with torch.no_grad():
        out = model(tensor)
        if out.dim() != 2:
            raise HTTPException(500, "Invalid model output")

        probs = torch.softmax(out, dim=1)[0].cpu().numpy().tolist()
        pred_class = int(torch.argmax(out, dim=1).item())
        confidence = round(max(probs) * 100, 2)

    try:
        target_layer = model.layer4[-1]
    except Exception:
        convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
        target_layer = convs[-1]

    gradcam_b64 = generate_gradcam(model, tensor, target_layer, pred_class)
    input_b64 = base64.b64encode(img_bytes).decode()

    # -------------------------------
    # SAVE TO SUPABASE
    # -------------------------------
    if user_id:
        supabase.table("predictions").insert({
            "user_id": user_id,
            "disease_type": disease,
            "prediction_result": {
                "predicted_class": pred_class,
                "probabilities": probs
            },
            "confidence_score": confidence,
            "explainability_data": {
                "gradcam": True
            }
        }).execute()

    try:
        with open("app/templates/image_report_template.html", "r", encoding="utf-8") as f:
            template = f.read()
    except FileNotFoundError:
        raise HTTPException(500, "Missing image_report_template.html")

    html = (
        template.replace("{{DISEASE}}", disease.upper())
                .replace("{{PRED_CLASS}}", str(pred_class))
                .replace("{{PROBS}}", str(probs))
                .replace("{{CONFIDENCE}}", str(confidence))
                .replace("{{INPUT_IMAGE}}", input_b64)
                .replace("{{GRAD_CAM}}", gradcam_b64)
    )

    return HTMLResponse(content=html, status_code=200)
