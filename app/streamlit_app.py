"""찾아줘, 가나디 — 실종견 찾기 데모.

사용자가 실종견 사진을 올리면, 국가동물보호정보시스템(animal.go.kr) 실제 공고 갤러리 중에서
비슷한 개체 Top-10을 찾아 보여준다. 학습된 임베딩 모델(E4: ResNet50+BNNeck+Triplet)의 코사인 유사도에
색상 히스토그램 유사도를 섞어(alpha=0.75) 순위를 매긴다 — 임베딩은 "진짜 정답"을 잘 찾아내지만
오답 후보끼리의 순서(예: 색깔이 완전히 다른 개가 상위권에 나오는 문제)는 보정이 안 돼서,
전통적인 색상 비교를 더해 보완했다 (scripts/eval_color_blend.py로 검증, 실제 Rank-1도 개선됨).

실행:
    streamlit run app/streamlit_app.py
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.data.transforms import build_eval_transform
from src.models.backbones import BNNeckModel, get_device
from src.retrieval.color import color_histogram

CACHE_DIR = PROJECT_ROOT / "demo_cache"
SHELTER_ROOT = PROJECT_ROOT / "Data" / "shelter"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "E1_resnet50_bnneck_breedpretrain_triplet.pt"
COLOR_ALPHA = 0.75  # 임베딩 75% + 색상 25% (scripts/eval_color_blend.py로 검증한 값)
TOP_K = 10  # 실제 사진 재현 테스트: 진짜 정답이 순위 100위 안쪽까지는 종종 들어와, 5보다 넉넉하게 보여준다

st.set_page_config(page_title="찾아줘, 가나디", page_icon="🐕", layout="wide")


@st.cache_resource
def load_model():
    import torch
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    model = BNNeckModel(num_classes=ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    device = get_device()
    model.to(device)
    return model, build_eval_transform(), device


@st.cache_data
def load_gallery():
    embeddings = np.load(CACHE_DIR / "gallery_embeddings.npy")
    color_hists = np.load(CACHE_DIR / "gallery_color_hists.npy")
    index = pd.read_csv(CACHE_DIR / "gallery_index.csv")
    meta = pd.read_csv(CACHE_DIR / "gallery_meta.csv")
    index = index.merge(meta, on="desertion_no", how="left")
    return embeddings, color_hists, index


def embed_query(img: Image.Image, model, transform, device) -> np.ndarray:
    import torch
    x = transform(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x).cpu().numpy()[0]
    return feat / (np.linalg.norm(feat) + 1e-12)


def embed_query_multi(imgs: list, model, transform, device) -> np.ndarray:
    """여러 장의 쿼리 사진을 각각 임베딩한 뒤 평균 -> 재정규화.
    사진이 여러 장이면 자세·조명에 따른 임베딩 흔들림이 평균으로 상쇄돼 훨씬 안정적이다
    (DogFaceNet 검증: 1장 Rank-1 87.1%/mAP 79.3% -> 2장 95.4%/89.1% -> 3장 95.8%/92.5%)."""
    feats = np.stack([embed_query(img, model, transform, device) for img in imgs])
    avg = feats.mean(axis=0)
    return avg / (np.linalg.norm(avg) + 1e-12)


def region_of(addr) -> str:
    if not isinstance(addr, str) or not addr.strip():
        return "전체"
    return addr.split()[0]


def main():
    st.title("🐕 찾아줘, 가나디")
    st.caption("보호소 유기견 공고와 실종견 사진을 매칭하는 AI 검색 데모 "
               "(국가동물보호정보시스템 실데이터 기반)")
    st.warning("⚠️ 이 결과는 **참고용 후보**입니다. 최종 확인은 반드시 해당 보호소 방문/연락으로 해주세요.")

    if not CHECKPOINT.exists():
        st.error(f"체크포인트가 없습니다: {CHECKPOINT}\n"
                 f"먼저 `python scripts/train_embed.py`로 E1을 학습시켜 주세요.")
        return
    if not (CACHE_DIR / "gallery_embeddings.npy").exists():
        st.error("갤러리 임베딩 캐시가 없습니다. 먼저 `python scripts/build_demo_gallery.py`를 실행하세요.")
        return

    model, transform, device = load_model()
    embeddings, color_hists, index = load_gallery()

    with st.sidebar:
        st.header("검색 조건 (선택)")
        st.caption("모르면 비워두세요 — 조건에 맞는 후보가 없으면 자동으로 전체에서 찾습니다.")

        sexes = ["전체"] + sorted(index.sex_cd.dropna().unique().tolist())
        sex = st.selectbox("성별", sexes)

        regions = ["전체"] + sorted({region_of(a) for a in index.care_addr.dropna()})
        region = st.selectbox("지역(시도)", regions)

        st.divider()
        st.metric("갤러리 규모", f"{index.desertion_no.nunique()}건 / 사진 {len(index)}장")
        st.caption("데이터 출처: [국가동물보호정보시스템](https://www.animal.go.kr)")

    MAX_PHOTOS = 5
    uploaded = st.file_uploader(
        "실종견 사진을 올려주세요 (여러 장일수록 정확해져요, 최대 5장)",
        type=["jpg", "jpeg", "png"], accept_multiple_files=True,
    )

    if not uploaded:
        st.info(f"사진을 올리면 비슷한 개체 Top-{TOP_K}를 보여드립니다. "
                 "**여러 각도·자세의 사진을 함께 올리면 정확도가 크게 올라가요** "
                 "(자체 검증: 1장 대비 2장이면 정확도가 확 뛰고, 3장부터는 거의 최대치예요).")
        return
    if len(uploaded) > MAX_PHOTOS:
        st.warning(f"최대 {MAX_PHOTOS}장까지만 사용해요 — 처음 {MAX_PHOTOS}장만 반영합니다.")
        uploaded = uploaded[:MAX_PHOTOS]

    query_imgs = [Image.open(f) for f in uploaded]
    st.caption(f"업로드한 사진 {len(query_imgs)}장")
    cols = st.columns(len(query_imgs))
    for col, img in zip(cols, query_imgs):
        with col:
            st.image(img, use_container_width=True)

    if st.button("🔍 찾기", type="primary"):
        t0 = time.time()
        with st.spinner("검색 중..."):
            q_feat = embed_query_multi(query_imgs, model, transform, device)
            hists = np.stack([color_histogram(img.convert("RGB")) for img in query_imgs])
            q_hist = hists.mean(axis=0)

            mask = pd.Series(True, index=index.index)
            if sex != "전체":
                mask &= index.sex_cd == sex
            if region != "전체":
                mask &= index.care_addr.apply(region_of) == region
            if mask.sum() == 0:
                st.warning("조건에 맞는 공고가 없어 전체 갤러리에서 검색합니다.")
                mask[:] = True

            cand_idx = index[mask].index.to_numpy()
            emb_sim = embeddings[cand_idx] @ q_feat  # 둘 다 L2 정규화됨 -> 내적 = 코사인 유사도
            color_sim = np.minimum(color_hists[cand_idx], q_hist).sum(axis=1)  # 히스토그램 교집합
            final_sim = COLOR_ALPHA * emb_sim + (1 - COLOR_ALPHA) * color_sim

            cand = index.loc[cand_idx].copy()
            cand["similarity"] = final_sim
            # 같은 개체(desertion_no)는 가장 잘 맞는 사진 하나만 남긴다
            best_per_dog = cand.sort_values("similarity", ascending=False).drop_duplicates("desertion_no")
            top_candidates = best_per_dog.head(TOP_K)

        st.success(f"검색 완료 ({time.time() - t0:.1f}초, 후보 {mask.sum()}건 중에서)")
        st.subheader(f"가장 비슷한 후보 Top-{TOP_K}")

        rows = [top_candidates.iloc[i:i + 5] for i in range(0, len(top_candidates), 5)]
        for row_df in rows:
            cols = st.columns(5)
            for col, (_, r) in zip(cols, row_df.iterrows()):
                with col:
                    st.image(str(SHELTER_ROOT / r.relpath), use_container_width=True)
                    st.metric("유사도", f"{r.similarity*100:.1f}%")
                    st.markdown(f"**{r.kind_nm}** · {r.sex_cd} · {r.age}")
                    st.caption(f"{r.care_nm}\n{r.care_addr}\n☎ {r.care_tel}")
                    if isinstance(r.special_mark, str) and r.special_mark.strip():
                        st.caption(f"특징: {r.special_mark}")
                    st.caption(f"공고번호: {r.notice_no}")


if __name__ == "__main__":
    main()
