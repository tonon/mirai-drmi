from fastapi import FastAPI, HTTPException,  UploadFile, File
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import shutil
import uuid

import cv2
import numpy as np
from collections import Counter


app = FastAPI(
    title="Lesion Inference API",
    description="API para exposição dos descritores de inferência clínica da lesão retinal",
    version="1.0"
)

import os
os.makedirs("temp", exist_ok=True)
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caminho para o CSV gerado pelo pipeline
CSV_PATH = BASE_DIR / "data" / "processed" / "inference" / "lesion_inference.csv"

if not CSV_PATH.exists():
    raise RuntimeError(f"Arquivo CSV não encontrado: {CSV_PATH}")

# Carrega o CSV uma única vez em memória
df = pd.read_csv(CSV_PATH)


@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(url='/docs')

@app.get("/lesions")
def list_lesions():
    """
    Retorna todos os registros de inferência.
    """
    return df.to_dict(orient="records")


@app.get("/lesions/{image_id}")
def get_lesion(image_id: str):
    """
    Retorna os descritores de uma imagem específica.
    """
    row = df[df["image_id"] == image_id]

    if row.empty:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")

    return row.iloc[0].to_dict()


@app.get("/lesions/by_class/{lesion_class}")
def get_by_class(lesion_class: str):
    """
    Filtra registros por classe (ex.: amd, normal).
    """
    filtered = df[df["class"] == lesion_class]

    if filtered.empty:
        raise HTTPException(status_code=404, detail="Nenhum registro encontrado")

    return filtered.to_dict(orient="records")

from app.pipelines.inference_pipeline import analyze_by_component

def load_ensemble(image: np.ndarray) -> np.ndarray:
    """Mock temporário de segmentação"""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2 - 40, h // 2), 18, 255, -1)
    cv2.circle(mask, (w // 2 + 40, h // 2), 22, 255, -1)
    return mask


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    # --------------------------------------------------
    # 1. Salvar imagem temporária
    # --------------------------------------------------
    tmp_path = Path("temp") / file.filename
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    image = cv2.imread(str(tmp_path))
    if image is None:
        return {"error": "Imagem inválida"}

    # --------------------------------------------------
    # 2. Rodar ensemble
    # --------------------------------------------------
    ensemble_mask = load_ensemble(image)
    binary_mask = (ensemble_mask > 0).astype(np.uint8)

    # --------------------------------------------------
    # 3. Análise Clínica
    # --------------------------------------------------
    drusas = analyze_by_component(binary_mask)

    from collections import Counter
    counts = Counter([d["morphology_class"] for d in drusas])

    # --------------------------------------------------
    # 4. Salvar máscara para visualização
    # --------------------------------------------------
    mask_path = Path("temp") / f"{tmp_path.stem}_mask.png"
    cv2.imwrite(str(mask_path), ensemble_mask)
    mask_url = f"/temp/{mask_path.name}"

    return {
        "image_id": tmp_path.stem,
        "mask_url": mask_url,
        "drusas": drusas,
        "num_small_drusas": counts.get("small_druse", 0),
        "num_medium_drusas": counts.get("medium_druse", 0),
        "num_large_regions": counts.get("large_region", 0),
    }