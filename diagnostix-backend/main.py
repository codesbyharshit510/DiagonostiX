from dotenv import load_dotenv
import os
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.image_routes import router as image_router
from app.tabular_routes import router as tabular_router
from app.report_routes import router as report_router
from app.tabular_explain_routes import router as tabular_explain_router
from app.llm_routes import router as llm_router

app = FastAPI(title="DiagnostiX Backend")

# -------------------------
# CORS
# -------------------------
origins = [
    # Local development
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    # Production
    "https://diagonosti-x.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "DiagnostiX Backend Running"}

# -------------------------
# ROUTERS
# -------------------------
app.include_router(image_router, prefix="/image")
app.include_router(report_router, prefix="/image")
app.include_router(tabular_router, prefix="/tabular")
app.include_router(tabular_explain_router, prefix="/tabular")
app.include_router(llm_router)
