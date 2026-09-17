"""animal.go.kr 수집 매니페스트 -> PetFinder 입력 선택지(data/options.json).

지역은 API의 org_nm("경기도 화성시" 형식), 품종은 kind_nm을 그대로 쓴다.
API 이름을 그대로 써야 서버의 장소/특징 텍스트 매칭과 표기가 맞는다.

사용:
    python demo/petfinder/tools/build_options.py \
        --manifest metadata/shelter_manifest_clean.csv \
        --out demo/petfinder/data/options.json
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# 행정구역 표시 순서. 목록에 없는 시도(명칭 변경 등)는 뒤에 가나다순으로 붙는다.
SIDO_ORDER = [
    "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시",
    "울산광역시", "세종특별자치시", "경기도", "강원특별자치도", "충청북도", "충청남도",
    "전북특별자치도", "전라남도", "전남광주통합특별시", "경상북도", "경상남도", "제주특별자치도",
]
PINNED_BREEDS = ["믹스견"]
TRAILING_BREEDS = ["기타"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="metadata/shelter_manifest_clean.csv")
    ap.add_argument("--out", default="demo/petfinder/data/options.json")
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    regions = defaultdict(set)
    for r in rows:
        sido, _, sigungu = (r.get("org_nm") or "").strip().partition(" ")
        if sido:
            regions[sido].add(sigungu.strip())

    order = {name: i for i, name in enumerate(SIDO_ORDER)}
    sido_names = sorted(regions, key=lambda s: (order.get(s, len(order)), s))

    kinds = Counter((r.get("kind_nm") or "").strip() for r in rows)
    kinds.pop("", None)
    middle = sorted(k for k in kinds if k not in PINNED_BREEDS + TRAILING_BREEDS)
    breeds = [k for k in PINNED_BREEDS if k in kinds] + middle + [k for k in TRAILING_BREEDS if k in kinds]

    dates = sorted(r["happen_dt"] for r in rows if r.get("happen_dt"))
    payload = {
        "source": "animal.go.kr abandonmentPublic_v2 (org_nm, kind_nm)",
        "snapshot": {"notices": len(rows), "happen_from": dates[0] if dates else "", "happen_to": dates[-1] if dates else ""},
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "regions": [{"sido": s, "sigungu": sorted(x for x in regions[s] if x)} for s in sido_names],
        "breeds": breeds,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    n_sigungu = sum(len(r["sigungu"]) for r in payload["regions"])
    print(f"[완료] {out}  시도 {len(sido_names)} · 시군구 {n_sigungu} · 품종 {len(breeds)}")


if __name__ == "__main__":
    main()
