"""임베딩(+색상 블렌딩) 랭킹이 실제로 품종·색깔 유사성을 반영하는지 정량 진단.

문제의식: MPDD/DogFaceNet은 품종 라벨이 없어서 "Top-5가 품종/색깔이 완전히 다르다"는
사용자 피드백을 지금까지 벤치마크로 검증한 적이 없다. shelter 메타데이터(kind_nm, color_cd)를
정답 라벨처럼 활용해, "쿼리와 같은 개체를 찾았는가"가 아니라 "쿼리와 비슷해 보이는 개를
찾았는가"를 직접 측정한다.

방법: 갤러리의 각 사진을 쿼리로 삼아(자기 자신의 공고=desertion_no는 제외) Top-K 이웃을
뽑고, 이웃의 품종(kind_nm)·색상(color_cd) 일치율을 무작위 기준선(base rate)과 비교한다.
"믹스견"은 전체의 82%를 차지해 품종 신호가 희석되므로 순종만 따로도 본다.

사용:
    python scripts/diagnose_breed_color_coherence.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

TOP_K = 5
N_QUERY_SAMPLE = 400
SEED = 42


def color_tokens(s: str) -> set:
    if not isinstance(s, str):
        return set()
    return set(s.replace("&", " ").split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="demo_cache")
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir)

    idx = pd.read_csv(cache_dir / "gallery_index.csv")
    meta = pd.read_csv(cache_dir / "gallery_meta.csv")[["desertion_no", "kind_nm", "color_cd"]]
    feats = np.load(cache_dir / "gallery_embeddings.npy")
    hists = np.load(cache_dir / "gallery_color_hists.npy")

    photo_meta = idx.merge(meta, on="desertion_no", how="left")
    n = len(photo_meta)
    assert n == feats.shape[0] == hists.shape[0]

    emb_sim = feats @ feats.T
    color_sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        color_sim[i] = np.minimum(hists, hists[i]).sum(axis=1)

    dog_ids = photo_meta["desertion_no"].to_numpy()
    kind = photo_meta["kind_nm"].to_numpy()
    colors = [color_tokens(c) for c in photo_meta["color_cd"]]

    same_dog = dog_ids[:, None] == dog_ids[None, :]

    rng = np.random.default_rng(SEED)
    query_idx = rng.choice(n, size=min(N_QUERY_SAMPLE, n), replace=False)

    # 전체 갤러리(자기 개 제외) 기준 base rate
    kind_counts = pd.Series(kind).value_counts()
    is_mix = kind == "믹스견"

    def topk_match_rates(alpha, restrict_nonmix=False):
        breed_hits, color_hits, total = 0, 0, 0
        for qi in query_idx:
            if restrict_nonmix and is_mix[qi]:
                continue
            final = alpha * emb_sim[qi] + (1 - alpha) * color_sim[qi]
            final = final.copy()
            final[same_dog[qi]] = -1e9  # 자기 자신(같은 공고) 제외
            top = np.argpartition(-final, TOP_K)[:TOP_K]
            top = top[np.argsort(-final[top])]
            for t in top:
                total += 1
                if kind[t] == kind[qi]:
                    breed_hits += 1
                if colors[t] & colors[qi]:
                    color_hits += 1
        return breed_hits / total, color_hits / total, total

    def base_rate(restrict_nonmix=False):
        breed_hits, color_hits, total = 0, 0, 0
        for qi in query_idx:
            if restrict_nonmix and is_mix[qi]:
                continue
            others = np.where(~same_dog[qi])[0]
            sample = rng.choice(others, size=TOP_K, replace=False)
            for t in sample:
                total += 1
                if kind[t] == kind[qi]:
                    breed_hits += 1
                if colors[t] & colors[qi]:
                    color_hits += 1
        return breed_hits / total, color_hits / total, total

    print(f"[info] 쿼리 샘플 {len(query_idx)}장 / 갤러리 {n}장 / Top-{TOP_K}\n")

    for label, alpha in [("임베딩만 (alpha=1.0)", 1.0), ("임베딩75+색상25 (alpha=0.75, 데모 실제 설정)", 0.75)]:
        b, c, t = topk_match_rates(alpha)
        print(f"[{label}] 전체 — 품종 일치율={b:.1%}  색상 일치율={c:.1%}  (n={t})")
        b2, c2, t2 = topk_match_rates(alpha, restrict_nonmix=True)
        print(f"[{label}] 순종만 — 품종 일치율={b2:.1%}  색상 일치율={c2:.1%}  (n={t2})")

    print()
    b, c, t = base_rate()
    print(f"[무작위 기준선] 전체 — 품종 일치율={b:.1%}  색상 일치율={c:.1%}  (n={t})")
    b2, c2, t2 = base_rate(restrict_nonmix=True)
    print(f"[무작위 기준선] 순종만 — 품종 일치율={b2:.1%}  색상 일치율={c2:.1%}  (n={t2})")


if __name__ == "__main__":
    main()
