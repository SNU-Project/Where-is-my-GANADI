"""Multi-pose Dog Dataset -> manifest CSV.

파일명 규칙 (Market-1501 방식):  <dog_id>_c<camera>s<sequence>_<frame>.jpg
예) 95_c3s2_5.jpg  ->  dog_id=95, camera=3, sequence=2, frame=5

사용:
    python scripts/build_multipose_manifest.py \
        --root "Data/Multi-pose dog dataset/pytorch" \
        --out  metadata/multipose_dog_manifest.csv
"""
import argparse
import csv
import re
from pathlib import Path

from PIL import Image

# 표준형과, query에 1건 있는 변형(146_c1_s3_1.jpg)을 함께 허용
NAME_RE = re.compile(r"^(?P<pid>\d+)_c(?P<cam>\d+)_?s(?P<seq>\d+)_(?P<frame>\d+)\.jpg$", re.I)

SPLITS = ["train", "val", "query", "gallery"]


def parse_name(fname: str):
    m = NAME_RE.match(fname)
    if not m:
        return None
    return (
        int(m.group("pid")),
        int(m.group("cam")),
        int(m.group("seq")),
        int(m.group("frame")),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Data/Multi-pose dog dataset/pytorch")
    ap.add_argument("--out", default="metadata/multipose_dog_manifest.csv")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    unparsed = []
    for split in SPLITS:
        d = root / split
        if not d.is_dir():
            print(f"[warn] 폴더 없음: {d}")
            continue
        for p in sorted(d.glob("*.jpg")):
            parsed = parse_name(p.name)
            if parsed is None:
                unparsed.append(str(p))
                continue
            pid, cam, seq, frame = parsed
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception as e:  # noqa: BLE001
                print(f"[warn] 이미지 열기 실패 {p}: {e}")
                w = h = 0
            rows.append(
                {
                    "split": split,
                    "filename": p.name,
                    "relpath": str(p.relative_to(root)),
                    "dog_id": pid,
                    "camera": cam,
                    "sequence": seq,
                    "frame": frame,
                    "width": w,
                    "height": h,
                    "aspect_ratio": round(w / h, 4) if h else "",
                    "filesize_bytes": p.stat().st_size,
                }
            )

    fields = [
        "split", "filename", "relpath", "dog_id", "camera", "sequence",
        "frame", "width", "height", "aspect_ratio", "filesize_bytes",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # ---- 요약 ----
    def ids(split):
        return {r["dog_id"] for r in rows if r["split"] == split}

    print(f"\n[완료] {out}  (총 {len(rows)} 행)")
    for split in SPLITS:
        srows = [r for r in rows if r["split"] == split]
        if not srows:
            continue
        sids = ids(split)
        cams = sorted({r["camera"] for r in srows})
        print(
            f"  {split:8s} images={len(srows):4d}  ids={len(sids):3d}  "
            f"imgs/id={len(srows)/len(sids):.1f}  cameras={cams}"
        )
    all_ids = {r["dog_id"] for r in rows}
    print(f"  {'TOTAL':8s} images={len(rows):4d}  ids={len(all_ids):3d}")
    print(f"  train∩test ids: {len(ids('train') & (ids('query') | ids('gallery')))}  "
          f"(0이면 올바른 open-set 분할)")
    print(f"  query ids ⊆ gallery ids: {ids('query') <= ids('gallery')}")
    ws = [r["width"] for r in rows if r["width"]]
    hs = [r["height"] for r in rows if r["height"]]
    if ws:
        print(f"  width  min/median/max: {min(ws)}/{sorted(ws)[len(ws)//2]}/{max(ws)}")
        print(f"  height min/median/max: {min(hs)}/{sorted(hs)[len(hs)//2]}/{max(hs)}")
    if unparsed:
        print(f"  [주의] 파싱 실패 {len(unparsed)}건: {unparsed}")


if __name__ == "__main__":
    main()
