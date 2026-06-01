# generate_image_report.py
"""
Generate an enhanced interactive HTML report for an image using your local backend.
Includes:
 - Diagnosis explanation (human-friendly)
 - Probability interpretation (risk text)
 - Disease-specific medical notes

Usage:
    python generate_image_report.py --image path/to/test.png --disease brain_tumor --out report.html --server http://127.0.0.1:8000
"""

import argparse
import base64
import requests
import webbrowser
import os
from pathlib import Path
import datetime
import json
import sys

# -------------------------
# Default image (from conversation upload)
# -------------------------
DEFAULT_IMAGE = "/mnt/data/4664770a-5c3c-4e2e-8780-825701949f4e.png"

# -------------------------
# HTTP helpers (predict + gradcam)
# -------------------------
def post_predict(server_url, image_path, disease):
    url = server_url.rstrip("/") + "/image/predict"
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {"disease": disease}
        r = requests.post(url, files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.json()

def post_gradcam(server_url, image_path, disease):
    url = server_url.rstrip("/") + "/image/gradcam"
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {"disease": disease}
        r = requests.post(url, files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.json()

# -------------------------
# Small utilities for the report
# -------------------------
def interpret_probability(probs, class_idx):
    """
    Convert a probability numeric value into a human-friendly risk string.
    """
    p = float(probs[class_idx])
    if p >= 0.90:
        return "Very high likelihood (urgent clinical review recommended)"
    if p >= 0.75:
        return "High likelihood — clinical correlation recommended"
    if p >= 0.50:
        return "Moderate likelihood — consider follow-up testing"
    if p >= 0.25:
        return "Low likelihood — monitor and follow-up if symptoms persist"
    return "Very low likelihood"

# disease specific textual explanations (A + C)
DISEASE_TEXT = {
    "brain_tumor": {
        "classes": ["No tumor", "Glioma", "Meningioma", "Pituitary tumor"],
        "explanations": {
            "No tumor": "No imaging signs strongly suggestive of a tumor on the supplied slice/scan. Clinical correlation and further imaging may still be required if symptoms persist.",
            "Glioma": "Gliomas are intrinsic brain tumors arising from glial cells. They can be infiltrative; further neuro-imaging, specialist referral and possible biopsy are recommended.",
            "Meningioma": "Meningiomas arise from the meninges and often have a distinct appearance. Many are benign but may need neurosurgical review depending on size and symptoms.",
            "Pituitary tumor": "Pituitary tumors occur in the sellar region and can affect hormone production. Endocrinological and neurosurgical evaluation is recommended."
        },
        "notes": "MRI appearance may vary with acquisition sequence. This model uses axial slices resized to the training input. False positives/negatives can occur — use clinical judgment."
    },
    "pneumonia": {
        "classes": ["No pneumonia", "Bacterial pneumonia", "Viral pneumonia", "Other lung opacity"],
        "explanations": {
            "No pneumonia": "No clear consolidation or infiltrate detected on the provided X-ray image. Consider other causes of symptoms if present.",
            "Bacterial pneumonia": "Pattern suggests lobar/segmental consolidation consistent with bacterial infection. Correlate with clinical signs and consider antibiotics as per clinical guidelines.",
            "Viral pneumonia": "Diffuse interstitial patterns may suggest viral infection. Consider viral testing and clinical context.",
            "Other lung opacity": "Non-specific lung opacity that may reflect atelectasis, scarring, or other pathology. Clinical correlation and further imaging recommended."
        },
        "notes": "Chest X-ray appearance depends on exposure and position. For suspected pneumonia, combine with clinical exam and lab tests (e.g., CBC, cultures)."
    },
    "alzheimers": {
        "classes": ["No AD pattern", "Early", "Moderate", "Advanced"],
        "explanations": {
            "No AD pattern": "No clear structural MRI features consistent with Alzheimer's disease on this image. Clinical assessment and cognitive testing remain primary.",
            "Early": "Subtle structural changes consistent with early neurodegenerative change. Consider neuropsychological testing and specialist referral.",
            "Moderate": "Structural atrophy patterns consistent with moderate neurodegeneration; correlate with cognitive function and consider management planning.",
            "Advanced": "More pronounced atrophy typical of advanced disease; discuss care planning and specialist management."
        },
        "notes": "Structural MRI alone cannot diagnose Alzheimer's disease — combine with cognitive testing, biomarkers, and clinical history."
    }
}

# -------------------------
# HTML template (enhanced)
# -------------------------
HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>DiagnostiX — Enhanced Image Report</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; padding:18px; background:#f7f8fb; color:#0f172a}
    .container{max-width:1100px;margin:0 auto}
    header{display:flex;gap:12px;align-items:center}
    h1{margin:0;font-size:22px}
    .meta{color:#6b7280;font-size:13px}
    .card{background:white;border-radius:12px;padding:16px;margin-top:12px;box-shadow:0 6px 18px rgba(11,22,50,0.06)}
    img{max-width:100%;height:auto;border-radius:8px}
    .row{display:flex;gap:16px;align-items:flex-start}
    .left{flex:1}
    .right{width:360px}
    .controls{display:flex;gap:8px;margin-top:12px}
    .badge{display:inline-block;padding:6px 10px;border-radius:999px;background:#eef2ff;color:#1e3a8a;font-weight:600}
    pre{background:#f3f6fb;padding:10px;border-radius:8px;overflow:auto}
    .section-title{font-weight:700;margin-bottom:8px}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>DiagnostiX — Enhanced Image Report</h1>
        <div class="meta">Disease: <strong>{disease}</strong> &nbsp;•&nbsp; Generated: {time}</div>
      </div>
      <div style="margin-left:auto">
        <button onclick="downloadPNG()" style="padding:8px 12px;border-radius:8px;background:#2563eb;color:#fff;border:0;cursor:pointer">Download PNG</button>
        <button onclick="downloadPDF()" style="padding:8px 12px;border-radius:8px;background:#e6eefc;color:#11438a;border:0;cursor:pointer;margin-left:8px">Download PDF</button>
      </div>
    </header>

    <div class="row" style="margin-top:16px">
      <div class="left">
        <div class="card">
          <div style="display:flex;gap:12px;align-items:flex-start">
            <div style="flex:1">
              <div class="section-title">Image & Grad-CAM</div>
              <div id="imageWrap" style="position:relative;border-radius:8px;overflow:hidden">
                <img id="orig" src="data:image/png;base64,{orig_b64}" alt="original" />
              </div>
              <div class="controls">
                <button onclick="showOriginal()" class="badge">Original</button>
                <button onclick="showOverlay()" class="badge" style="background:#fff2e8;color:#873800">Overlay</button>
                <button onclick="showSideBySide()" class="badge" style="background:#e6f5ea;color:#0b7a3f">Side-by-side</button>
              </div>
            </div>

            <div style="width:320px">
              <div class="section-title">Probabilities & Interpretation</div>
              <canvas id="probChart" style="background:#fff;border-radius:8px;display:block"></canvas>
              <div style="margin-top:8px;font-size:13px;color:#475569">Class order: <code>{classes_list}</code></div>
              <div style="margin-top:10px">
                <div style="font-weight:700">Probability interpretation</div>
                <div id="probInterpret" style="margin-top:6px">{prob_interpret_html}</div>
              </div>
            </div>
          </div>
        </div>

        <div class="card" style="margin-top:12px">
          <div class="section-title">Diagnosis explanation</div>
          <div style="font-size:14px;color:#0f172a" id="diagExpl">{diag_explanation}</div>
          <div style="margin-top:12px;color:#6b7280;font-size:13px">{disease_notes}</div>
        </div>

        <div class="card" style="margin-top:12px">
          <div class="section-title">Raw model outputs</div>
          <pre id="rawJson">{raw_json}</pre>
        </div>
      </div>

      <div class="right">
        <div class="card">
          <div style="font-weight:700">Predicted</div>
          <div style="margin-top:8px">
            <div style="font-size:18px;font-weight:800">{pred_label}</div>
            <div style="color:#6b7280;margin-top:4px">Class index: {pred_index}</div>
            <div style="margin-top:10px"><strong>Final interpretation:</strong> <span id="finalInterp" style="font-weight:700">{final_interpretation}</span></div>
          </div>
          <div style="margin-top:12px">
            <div style="font-weight:700">Quick actions</div>
            <div style="margin-top:8px">
              <button onclick="downloadPNG()" style="padding:8px 12px;border-radius:8px;background:#2563eb;color:#fff;border:0;cursor:pointer;width:100%">Download PNG</button>
            </div>
          </div>
        </div>

        <div class="card" style="margin-top:12px">
          <div style="font-weight:700">Advice & next steps (general)</div>
          <ul style="margin-top:8px;color:#334155">
            <li>If the result indicates high likelihood, arrange clinical follow-up immediately.</li>
            <li>Combine imaging with clinical exam, labs and specialist referral as needed.</li>
            <li>AI outputs are probabilistic — do not use as sole basis for definitive diagnosis.</li>
          </ul>
        </div>
      </div>
    </div>

    <footer style="margin-top:18px;color:#94a3b8">Generated by DiagnostiX · Local · Model: {model_info}</footer>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>

  <script>
    const origB64 = "{orig_b64}";
    const overlayB64 = "{overlay_b64}";
    const classes = {classes_json};
    const probs = {probs_json};
    const pred_index = {pred_index};
    const pred_label = "{pred_label}";

    function showOriginal(){
      document.getElementById("imageWrap").innerHTML =
        `<img id="orig" src="data:image/png;base64,${origB64}" />`;
    }
    function showOverlay(){
      document.getElementById("imageWrap").innerHTML =
        `<div style="position:relative"><img src="data:image/png;base64,${origB64}" style="opacity:1"/><img src="data:image/png;base64,${overlayB64}" style="position:absolute;left:0;top:0;width:100%;height:100%;mix-blend-mode:multiply;opacity:0.8"/></div>`;
    }
    function showSideBySide(){
      document.getElementById("imageWrap").innerHTML =
        `<div style="display:flex;gap:8px"><img src="data:image/png;base64,${origB64}" style="width:50%"/><img src="data:image/png;base64,${overlayB64}" style="width:50%"/></div>`;
    }

    // chart
    (function(){
      const ctx = document.getElementById('probChart').getContext('2d');
      new Chart(ctx, {
        type: 'bar',
        data: {
          labels: classes,
          datasets: [{
            label: 'Probability',
            data: probs,
            backgroundColor: probs.map(p => p > 0.5 ? '#2563eb' : '#93c5fd')
          }]
        },
        options: {
          scales: { y: { beginAtZero:true, max:1 } },
          plugins: { legend: { display: false } }
        }
      });
    })();

    // build human-friendly interpretation
    function probToText(p){
      if (p >= 0.90) return "Very high likelihood (urgent review recommended)";
      if (p >= 0.75) return "High likelihood — clinical correlation recommended";
      if (p >= 0.50) return "Moderate likelihood — consider follow-up testing";
      if (p >= 0.25) return "Low likelihood — monitor / follow-up if symptoms persist";
      return "Very low likelihood";
    }
    // populate interpretation box
    const interpDiv = document.getElementById('probInterpret');
    interpDiv.innerHTML = classes.map((c, i) => {
      return `<div style="margin-bottom:6px"><strong>${c}:</strong> ${(probs[i]*100).toFixed(1)}% — ${probToText(probs[i])}</div>`;
    }).join('');

    // download functions
    async function downloadPNG(){
      // make canvas of the big container
      const el = document.querySelector('.container');
      const canvas = await html2canvas(el, {scale:2});
      const a = document.createElement('a');
      a.href = canvas.toDataURL('image/png');
      a.download = 'diagnostix_report_{disease}.png';
      a.click();
    }
    async function downloadPDF(){
      const { jsPDF } = window.jspdf;
      const el = document.querySelector('.container');
      const canvas = await html2canvas(el, {scale:2});
      const img = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p','mm','a4');
      const pageWidth = pdf.internal.pageSize.getWidth();
      const imgHeight = canvas.height * (pageWidth / canvas.width);
      pdf.addImage(img, 'PNG', 0, 0, pageWidth, imgHeight);
      pdf.save('diagnostix_report_{disease}.pdf');
    }

    // By default show overlay if available
    if (overlayB64 && overlayB64.length > 50) showOverlay();
  </script>
</body>
</html>
"""

# -------------------------
# Build & write report
# -------------------------
def build_report(server, image_path, disease, out_path):
    print(f"[+] Uploading '{image_path}' for prediction to {server} ...")
    pred = post_predict(server, image_path, disease)
    print("[+] Predict response received. Requesting Grad-CAM ...")
    grad = post_gradcam(server, image_path, disease)
    print("[+] Grad-CAM response received. Building report ...")

    # extract probabilities (support multiple possible key names)
    if "probabilities" in pred:
        probs = pred["probabilities"]
    elif "probs" in pred:
        probs = pred["probs"]
    else:
        # fallback: try 'predictions' or 'scores'
        probs = pred.get("predictions", pred.get("scores", []))

    # guard
    if not isinstance(probs, (list, tuple)):
        # sometimes nested numpy -> convert
        try:
            probs = list(map(float, probs))
        except Exception:
            raise RuntimeError("Unexpected probabilities format: " + str(probs))

    # gradcam key
    overlay_b64 = grad.get("gradcam_base64") or grad.get("gradcam") or grad.get("overlay")
    if overlay_b64 is None:
        raise RuntimeError("No gradcam returned from server response: " + str(grad))

    # read original image and encode base64
    with open(image_path, "rb") as f:
        orig_bytes = f.read()
    orig_b64 = base64.b64encode(orig_bytes).decode("ascii")

    # classes labels: try predict response, else disease mapping, else numeric
    classes = pred.get("classes") or pred.get("labels") or DISEASE_TEXT.get(disease, {}).get("classes") or [f"class_{i}" for i in range(len(probs))]
    # ensure list of strings
    classes = [str(c) for c in classes]

    # predicted index and label
    pred_index = int(pred.get("predicted_class", pred.get("pred_index", probs.index(max(probs)) if probs else 0)))
    pred_label = classes[pred_index] if 0 <= pred_index < len(classes) else str(pred_index)

    # build diagnosis explanation & disease notes (A + C)
    diag_explanation = DISEASE_TEXT.get(disease, {}).get("explanations", {}).get(pred_label, "No textual explanation available for this class.")
    disease_notes = DISEASE_TEXT.get(disease, {}).get("notes", "")

    # probability interpretation HTML (B)
    prob_interpret_html = ""
    for i, c in enumerate(classes):
        p = float(probs[i]) if i < len(probs) else 0.0
        prob_interpret_html += f"<div><strong>{c}</strong>: {(p*100):.1f}% — {interpret_probability(probs, i)}</div>"

    # model info (if provided)
    model_info = pred.get("model", pred.get("model_version", "local model"))

    raw_json = json.dumps({ "predict": pred, "gradcam": grad }, indent=2)

    html = HTML_TEMPLATE.format(
        disease=disease,
        time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        orig_b64=orig_b64,
        overlay_b64=overlay_b64,
        classes_list=", ".join(classes),
        classes_json=json.dumps(classes),
        probs_json=json.dumps([float(x) for x in probs]),
        probs=str(probs),
        pred_index=pred_index,
        pred_label=pred_label,
        diag_explanation=diag_explanation,
        disease_notes=disease_notes,
        prob_interpret_html=prob_interpret_html,
        final_interpretation=interpret_probability(probs, pred_index),
        model_info=model_info,
        raw_json=raw_json
    )

    Path(out_path).write_text(html, encoding="utf-8")
    print("[+] Report written to", out_path)
    # open in browser
    try:
        webbrowser.open("file://" + str(Path(out_path).resolve()))
    except Exception:
        print("[!] Could not open browser automatically; open the file manually:", out_path)

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", required=False, default=DEFAULT_IMAGE, help="Path to image file to analyze")
    parser.add_argument("--disease", "-d", required=True, help="disease key (e.g. brain_tumor, pneumonia, alzheimers)")
    parser.add_argument("--out", "-o", default="diagnostix_report.html", help="Output HTML file")
    parser.add_argument("--server", "-s", default="http://127.0.0.1:8000", help="Backend server URL")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] image file not found: {args.image}", file=sys.stderr)
        sys.exit(2)

    try:
        build_report(args.server, args.image, args.disease, args.out)
    except Exception as e:
        print("[ERROR] Failed to build report:", e, file=sys.stderr)
        sys.exit(1)
