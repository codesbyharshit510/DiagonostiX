from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from app.services.image_model import preprocess_image
from app.image_routes import MODEL_PATHS, get_model
from app.services.gradcam_custom import generate_gradcam
from app.utils.supabase import supabase
import torch
import base64
import json
from datetime import datetime

router = APIRouter()

# =====================================================
# Class Labels
# =====================================================
CLASS_LABELS = {
    "pneumonia": ["NORMAL", "PNEUMONIA"],
    "alzheimers": [
        "Mild Demented",
        "Moderate Demented",
        "Non Demented",
        "Very Mild Demented"
    ],
    "brain_tumor": [
        "Glioma Tumor",
        "Meningioma Tumor",
        "Pituitary Tumor",
        "No Tumor"
    ]
}

# =====================================================
# Medical Explanations (Professional)
# =====================================================
MEDICAL_EXPLANATIONS = {
    "pneumonia": {
        "NORMAL": (
            "The chest radiograph demonstrates clear lung fields without evidence of focal consolidation, "
            "interstitial infiltrates, or pleural effusion. These findings are consistent with normal pulmonary "
            "parenchyma and absence of active lower respiratory tract infection.\n\n"
            "Clinical correlation is advised if respiratory symptoms persist, as early or viral infections may "
            "precede radiographic changes."
        ),
        "PNEUMONIA": (
            "The image demonstrates areas of increased pulmonary opacity and consolidation, suggestive of pneumonia. "
            "These findings reflect alveolar inflammation commonly associated with infectious etiologies such as "
            "bacterial or viral pneumonia.\n\n"
            "Prompt clinical evaluation, laboratory testing, and antimicrobial therapy may be required depending "
            "on patient symptoms and oxygenation status."
        )
    },

    "alzheimers": {
        "Non Demented": (
            "Neuroimaging does not demonstrate structural abnormalities suggestive of neurodegenerative dementia. "
            "Cortical volume and hippocampal structures appear preserved for age.\n\n"
            "If cognitive symptoms persist, further neuropsychological assessment may be warranted."
        ),
        "Very Mild Demented": (
            "Subtle neuroanatomical changes are observed, consistent with very early cognitive decline. "
            "These findings may correspond to prodromal Alzheimer’s disease.\n\n"
            "Early intervention and regular monitoring are recommended."
        ),
        "Mild Demented": (
            "Findings are consistent with mild Alzheimer’s disease, characterized by early hippocampal and temporal "
            "lobe involvement.\n\n"
            "Patients may experience memory impairment with preserved functional independence."
        ),
        "Moderate Demented": (
            "Significant cortical and hippocampal atrophy is observed, consistent with moderate Alzheimer’s disease.\n\n"
            "Patients typically require assistance with daily activities and structured clinical care."
        )
    },

    "brain_tumor": {
        "Glioma Tumor": (
            "The Grad-CAM highlights infiltrative regions within the brain parenchyma, suggestive of glioma. "
            "Gliomas arise from glial cells and may demonstrate aggressive biological behavior.\n\n"
            "Further evaluation with contrast-enhanced MRI and oncological consultation is recommended."
        ),
        "Meningioma Tumor": (
            "The model localizes extra-axial regions consistent with meningioma, a typically slow-growing tumor "
            "originating from the meninges.\n\n"
            "Management may range from surveillance to surgical intervention depending on tumor size and symptoms."
        ),
        "Pituitary Tumor": (
            "Activation is noted in the sellar region, suggestive of a pituitary lesion. Such tumors may affect "
            "hormonal regulation and optic pathways.\n\n"
            "Endocrinological evaluation and dedicated pituitary imaging are advised."
        ),
        "No Tumor": (
            "No focal intracranial mass or abnormal enhancement is detected. Brain structures appear within normal "
            "anatomical limits.\n\n"
            "Clinical follow-up is advised if neurological symptoms persist."
        )
    }
}

# =====================================================
# HTML + PDF Template
# =====================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>DiagnostiX – Radiology Report</title>

<script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>

<style>
body{font-family:Arial,Helvetica,sans-serif;background:#f3f4f6;margin:0;padding:24px}
.container{max-width:1100px;margin:auto}
.card{background:#fff;border-radius:14px;padding:22px;margin-bottom:20px;
box-shadow:0 8px 24px rgba(0,0,0,0.08)}
h2,h3{margin-top:0}
.badge{display:inline-block;padding:6px 16px;border-radius:999px;font-size:13px;font-weight:bold}
.low{background:#e0f2fe;color:#0369a1}
.medium{background:#fef3c7;color:#92400e}
.high{background:#fee2e2;color:#991b1b}
.img-wrap img{width:100%;height:auto;border-radius:12px}
pre{background:#f1f5f9;padding:14px;border-radius:10px}
button{padding:10px 18px;border:none;border-radius:10px;
background:#2563eb;color:white;font-weight:bold;cursor:pointer}
</style>
</head>

<body>
<div class="container" id="report">

<div class="card">
<h2>DiagnostiX – Interactive Image Report</h2>
<p>
<b>Disease:</b> {{disease}}<br>
<b>Prediction:</b> {{pred_label}} (Class {{pred_index}})<br>
<b>Confidence Level:</b> <span class="badge {{confidence_class}}">{{confidence_text}}</span><br>
<b>Generated:</b> {{time}}
</p>
<button onclick="downloadPDF()">Download Hospital PDF</button>
</div>

<div class="card">
<h3>Image & Grad-CAM Visualization</h3>
<div class="img-wrap">
<img src="data:image/png;base64,{{overlay_b64}}">
</div>
</div>

<div class="card">
<h3>Medical Interpretation</h3>
<p style="line-height:1.8">{{explanation}}</p>
</div>

<div class="card">
<h3>Prediction Probabilities</h3>
<p>Class Order: {{classes_str}}</p>
<pre>{{probs}}</pre>
</div>

</div>

<script>
function downloadPDF(){
  const element = document.getElementById('report');
  html2pdf().from(element).set({
    margin: 0.5,
    filename: 'DiagnostiX_Hospital_Report.pdf',
    image: { type: 'jpeg', quality: 0.98 },
    html2canvas: { scale: 2 },
    jsPDF: { unit: 'in', format: 'a4', orientation: 'portrait' }
  }).save();
}
</script>

</body>
</html>
"""

# =====================================================
# Endpoint
# =====================================================
@router.post("/report")
async def image_report(
    disease: str,
    file: UploadFile = File(...),
    user_id: str | None = None,
):

    if disease not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail="Unknown disease")

    img_bytes = await file.read()
    tensor = preprocess_image(img_bytes)

    model = get_model(disease)
    device = next(model.parameters()).device

    with torch.no_grad():
        out = model(tensor.to(device))
        probs = torch.softmax(out, dim=1)[0].cpu().numpy().tolist()
        pred_index = int(torch.argmax(out, dim=1).item())

    class_names = CLASS_LABELS[disease]
    pred_label = class_names[pred_index]
    explanation = MEDICAL_EXPLANATIONS[disease][pred_label]
    normalized_disease = disease

    top = max(probs)
    if top >= 0.8:
        confidence_text, confidence_class = "HIGH CONFIDENCE", "high"
    elif top >= 0.55:
        confidence_text, confidence_class = "MODERATE CONFIDENCE", "medium"
    else:
        confidence_text, confidence_class = "LOW CONFIDENCE", "low"

    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    gradcam_b64 = generate_gradcam(model, tensor.to(device), convs[-1], pred_index)

    if user_id:
        supabase.table("predictions").insert({
            "user_id": user_id,
            "disease_type": normalized_disease,
            "prediction_result": {
                "predicted_class": pred_index,
                "predicted_label": pred_label,
                "probabilities": probs,
            },
            "confidence_score": round(top * 100, 2),
            "explainability_data": {
                "type": "gradcam",
                "available": True,
                "confidence_text": confidence_text,
                "explanation": explanation,
            },
        }).execute()

    html = (
        HTML_TEMPLATE
        .replace("{{disease}}", disease)
        .replace("{{pred_label}}", pred_label)
        .replace("{{pred_index}}", str(pred_index))
        .replace("{{confidence_text}}", confidence_text)
        .replace("{{confidence_class}}", confidence_class)
        .replace("{{time}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        .replace("{{overlay_b64}}", gradcam_b64)
        .replace("{{explanation}}", explanation)
        .replace("{{classes_str}}", ", ".join(class_names))
        .replace("{{probs}}", json.dumps(probs, indent=2))
    )

    return HTMLResponse(content=html)
