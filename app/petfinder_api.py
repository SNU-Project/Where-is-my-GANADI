"""demo/petfinder 프론트엔드(신기훈님 PR #3, koosamuel 원본 이관)가 기대하는 API 계약을
우리 E4 모델 + demo_cache 갤러리로 구현한 백엔드.

프론트(`demo/petfinder/app.js`)가 실제로 호출하는 엔드포인트 3개만 구현한다:
  - GET  /api/public/health                       -> {"status":"ok","data_classification":"public_notice_demo"}
  - GET  /api/public/animals/{animal_id}/image     -> 후보 사진 파일
  - POST /api/public/search (multipart: file, top_k, ...) -> {"items":[...]}

사용:
    uvicorn app.petfinder_api:app --port 8787
"""
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from src.data.transforms import build_eval_transform
from src.models.backbones import BNNeckModel, get_device
from src.retrieval.color import color_histogram

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "demo_cache"
SHELTER_ROOT = PROJECT_ROOT / "Data" / "shelter"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "E1_resnet50_bnneck_breedpretrain_triplet.pt"
COLOR_ALPHA = 0.75

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_state: dict = {}


def _clean(v):
    """pandas NaN -> None (표준 JSON엔 NaN이 없음)."""
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _load():
    if _state:
        return _state
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    model = BNNeckModel(num_classes=ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    device = get_device()
    model.to(device)

    embeddings = np.load(CACHE_DIR / "gallery_embeddings.npy")
    color_hists = np.load(CACHE_DIR / "gallery_color_hists.npy")
    index = pd.read_csv(CACHE_DIR / "gallery_index.csv")
    meta = pd.read_csv(CACHE_DIR / "gallery_meta.csv")
    index = index.merge(meta, on="desertion_no", how="left")

    _state.update(model=model, transform=build_eval_transform(), device=device,
                   embeddings=embeddings, color_hists=color_hists, index=index)
    return _state


@app.get("/api/public/health")
def health():
    return {"status": "ok", "data_classification": "public_notice_demo"}


@app.get("/api/public/animals/{animal_id}/image")
def animal_image(animal_id: str, slot: str = "popfile1"):
    state = _load()
    rows = state["index"][state["index"].desertion_no.astype(str) == str(animal_id)]
    if rows.empty:
        return JSONResponse({"detail": "not found"}, status_code=404)
    idx = 1 if (slot == "popfile2" and len(rows) > 1) else 0
    relpath = rows.iloc[idx].relpath
    return FileResponse(SHELTER_ROOT / relpath)


@app.post("/api/public/search")
async def search(
    file: UploadFile = File(...),
    top_k: int = Form(5),
    feature_text: str = Form(""),
    location_text: str = Form(""),
    missing_date: str = Form(""),
    exclude_exact_image: str = Form("false"),
):
    state = _load()
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")

    x = state["transform"](img).unsqueeze(0).to(state["device"])
    with torch.no_grad():
        feat = state["model"](x).cpu().numpy()[0]
    feat = feat / (np.linalg.norm(feat) + 1e-12)
    hist = color_histogram(img)

    emb_sim = state["embeddings"] @ feat
    color_sim = np.minimum(state["color_hists"], hist).sum(axis=1)
    final = COLOR_ALPHA * emb_sim + (1 - COLOR_ALPHA) * color_sim

    cand = state["index"].copy()
    cand["similarity"] = final
    top = (cand.sort_values("similarity", ascending=False)
                .drop_duplicates("desertion_no")
                .head(top_k))

    items = [{
        "animal_id": str(r.desertion_no),
        "image_slot": "popfile1",
        "kind_name": _clean(r.kind_nm),
        "sex": _clean(r.sex_cd),
        "happen_place": _clean(r.happen_place),
        "happen_date": str(r.happen_dt),
        "rationale": "이미지 임베딩(E4) + 색상 히스토그램 유사도",
        "final_score": float(r.similarity),
    } for _, r in top.iterrows()]
    return {"items": items}
