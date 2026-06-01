# DiagnostiX Backend (scaffold)

This repository contains a ready-to-run FastAPI backend scaffold for:
- Image model inference + Grad-CAM
- Tabular model inference + SHAP

**Important:** This scaffold does NOT include trained model binaries.
Place your trained models under `models/image/` and `models/tabular/`.

## Expected model files
- Image models (PyTorch `.pt` or `.pth`) under `models/image/`:
  - `pneumonia.pt`
  - `malaria.pt`
  - `brain_tumor.pt`

- Tabular models (joblib / pickle) under `models/tabular/`:
  - `diabetes.pkl`
  - `heart.pkl`
  - optionally: `background.npy` (numpy array of training rows for SHAP)

## Quick start (local)
1. Create virtual env: `python -m venv venv && source venv/bin/activate`
2. Install: `pip install -r requirements.txt`
3. Place your models in the `models/` directory.
4. Run: `./run.sh`
5. Open: `http://localhost:8000/docs` to use Swagger UI.



