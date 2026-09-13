"""국가동물보호정보시스템 구조동물 조회 API 수집 스크립트.

API: 농림축산검역본부_국가동물보호정보시스템 구조동물 조회 서비스 (data.go.kr)
Base URL: https://apis.data.go.kr/1543061/abandonmentPublicService_v2

사용 준비:
    1) 프로젝트 루트에 .env 파일 생성 (git에 커밋되지 않음, .gitignore 처리됨)
       ANIMAL_API_SERVICE_KEY=발급받은_서비스키
    2) pip install requests python-dotenv pillow

사용 예:
    # 최근 60일치 유기견(개) 공고 메타데이터만 수집
    python scripts/fetch_shelter_api.py --days 60 --upkind dog

    # 메타데이터 + 대표사진까지 다운로드
    python scripts/fetch_shelter_api.py --days 60 --upkind dog --download-images
"""
import argparse
import csv
import os
import time
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

BASE_URL = "https://apis.data.go.kr/1543061/abandonmentPublicService_v2/abandonmentPublic_v2"
UPKIND_CODE = {"dog": "417000", "cat": "422400", "etc": "429900"}
POPFILE_FIELDS = [f"popfile{i}" for i in range(1, 9)]

FIELDNAMES = [
    "desertion_no", "notice_no", "process_state", "happen_dt", "happen_place",
    "up_kind_nm", "kind_nm", "kind_full_nm", "color_cd", "age", "weight",
    "sex_cd", "neuter_yn", "special_mark",
    "care_nm", "care_tel", "care_addr", "org_nm", "care_reg_no",
    "notice_sdt", "notice_edt", "upd_tm",
    "photo_urls", "photo_local_paths",
]


def load_service_key() -> str:
    load_dotenv()
    key = os.environ.get("ANIMAL_API_SERVICE_KEY")
    if not key:
        raise SystemExit(
            "ANIMAL_API_SERVICE_KEY가 없습니다. 프로젝트 루트 .env 파일에 "
            "ANIMAL_API_SERVICE_KEY=발급받은_키 를 추가하세요."
        )
    return key


def fetch_page(service_key: str, params: dict, page_no: int, num_of_rows: int = 1000, retries: int = 3):
    q = {
        "serviceKey": service_key,
        "_type": "json",
        "pageNo": page_no,
        "numOfRows": num_of_rows,
        **params,
    }
    for attempt in range(1, retries + 1):
        resp = requests.get(BASE_URL, params=q, timeout=20)
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError as e:
            raise RuntimeError(f"JSON 파싱 실패 (응답 앞부분: {resp.text[:300]!r})") from e
        header = data.get("response", data).get("header", {}) if isinstance(data, dict) else {}
        result_code = header.get("resultCode")
        if result_code not in (None, "00"):
            if attempt < retries:
                time.sleep(1.5 * attempt)
                continue
            raise RuntimeError(f"API 오류: {header}")
        return data
    raise RuntimeError("재시도 초과")


def iter_items(service_key: str, params: dict, num_of_rows: int = 1000, sleep_sec: float = 0.3):
    page_no = 1
    total_count = None
    fetched = 0
    while True:
        data = fetch_page(service_key, params, page_no, num_of_rows)
        body = data.get("response", data).get("body", {})
        if total_count is None:
            total_count = int(body.get("totalCount", 0) or 0)
            print(f"[info] 조건에 맞는 전체 건수: {total_count}")
            if total_count == 0:
                return
        items = body.get("items")
        item_list = []
        if items:
            item = items.get("item")
            if isinstance(item, list):
                item_list = item
            elif isinstance(item, dict):
                item_list = [item]
        for it in item_list:
            yield it
        fetched += len(item_list)
        print(f"[info] page {page_no}: {len(item_list)}건 (누적 {fetched}/{total_count})")
        if fetched >= total_count or not item_list:
            return
        page_no += 1
        time.sleep(sleep_sec)


def to_row(item: dict) -> dict:
    photo_urls = [item.get(f) for f in POPFILE_FIELDS if item.get(f)]
    return {
        "desertion_no": item.get("desertionNo", ""),
        "notice_no": item.get("noticeNo", ""),
        "process_state": item.get("processState", ""),
        "happen_dt": item.get("happenDt", ""),
        "happen_place": item.get("happenPlace", ""),
        "up_kind_nm": item.get("upKindNm", ""),
        "kind_nm": item.get("kindNm", ""),
        "kind_full_nm": item.get("kindFullNm", ""),
        "color_cd": item.get("colorCd", ""),
        "age": item.get("age", ""),
        "weight": item.get("weight", ""),
        "sex_cd": item.get("sexCd", ""),
        "neuter_yn": item.get("neuterYn", ""),
        "special_mark": item.get("specialMark", ""),
        "care_nm": item.get("careNm", ""),
        "care_tel": item.get("careTel", ""),
        "care_addr": item.get("careAddr", ""),
        "org_nm": item.get("orgNm", ""),
        "care_reg_no": item.get("careRegNo", ""),
        "notice_sdt": item.get("noticeSdt", ""),
        "notice_edt": item.get("noticeEdt", ""),
        "upd_tm": item.get("updTm", ""),
        "photo_urls": "|".join(photo_urls),
        "photo_local_paths": "",  # download_images()가 채움
    }


def download_images(rows: list[dict], img_dir: Path, sleep_sec: float = 0.15):
    img_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    for row in rows:
        urls = [u for u in row["photo_urls"].split("|") if u]
        local_paths = []
        for i, url in enumerate(urls):
            ext = Path(urlparse(url).path).suffix or ".jpg"
            fname = f"{row['desertion_no']}_{i}{ext}"
            fpath = img_dir / fname
            if not fpath.exists():
                try:
                    r = session.get(url, timeout=15)
                    r.raise_for_status()
                    fpath.write_bytes(r.content)
                    time.sleep(sleep_sec)
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] 다운로드 실패 {url}: {e}")
                    continue
            local_paths.append(str(fpath.relative_to(img_dir.parent)))
        row["photo_local_paths"] = "|".join(local_paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60, help="구조날짜 기준 최근 N일 (기본 60)")
    ap.add_argument("--upkind", choices=["dog", "cat", "etc"], default="dog")
    ap.add_argument("--state", choices=["", "notice", "protect"], default="",
                     help="공고 상태 (빈값=전체, notice=공고중, protect=보호중)")
    ap.add_argument("--out-dir", default="Data/shelter")
    ap.add_argument("--download-images", action="store_true", help="대표사진까지 다운로드")
    ap.add_argument("--max-items", type=int, default=None, help="테스트용 상한 (전체 다운로드 전 확인용)")
    args = ap.parse_args()

    service_key = load_service_key()

    end = datetime.now()
    start = end - timedelta(days=args.days)
    params = {
        "upkind": UPKIND_CODE[args.upkind],
        "bgnde": start.strftime("%Y%m%d"),
        "endde": end.strftime("%Y%m%d"),
    }
    if args.state:
        params["state"] = args.state

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"

    rows = []
    for item in iter_items(service_key, params):
        rows.append(to_row(item))
        if args.max_items and len(rows) >= args.max_items:
            print(f"[info] --max-items {args.max_items} 도달, 중단")
            break

    print(f"[info] 수집된 공고 수: {len(rows)}")

    if args.download_images and rows:
        print("[info] 이미지 다운로드 시작...")
        download_images(rows, img_dir)

    manifest_path = out_dir / "shelter_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[완료] {manifest_path} ({len(rows)}행)")


if __name__ == "__main__":
    main()
