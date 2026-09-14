# 찾아줘, 가나디 (Where is my GANADI)

보호소 유기견 이미지와 실종견 사진을 매칭하는 **AI 기반 실종견 찾기 서비스**.

실종견 사진 1장을 업로드하면 보호소 유기견 DB에서 **같은 개체일 가능성이 높은 순서로 Top-K 후보**를 제시합니다.
사람 얼굴인식과 유사한 **개체 재식별(Re-Identification) / 이미지 검색** 문제로 접근합니다.

## 문서
- [과제 요건 (마감 9/22, 제출 체크리스트)](docs/02_과제요건.md) — **가장 먼저 볼 문서**
- [7일 스프린트 계획](docs/01_7일_스프린트_계획.md) — 현재 진행 기준 문서
- [전체 기획서 / 설계서 (참고용, 풀스코프)](docs/00_기획서.md)
- [참고 문헌](docs/references/)

## 데이터셋 (실사용 확정분)
| 데이터셋 | 역할 |
|---|---|
| [Multi-pose Dog Dataset](https://data.mendeley.com/datasets/v5j6m8dzhv/1) | 핵심 Re-ID 학습·평가 (전신·포즈 다양성), 191개체/1,657장 |
| [DogFaceNet](https://huggingface.co/datasets/dimidagd/DogFaceNet_224resize) | 핵심 Re-ID 학습·평가 (얼굴 클로즈업), 1,393개체/8,363장 |
| [국가동물보호정보시스템](https://www.animal.go.kr/front/index.do) | 데모 갤러리 + 실도메인 갭 확인 (학습엔 미사용), 공고중 유기견 1,446건(전처리 후) |

PetFace, Kaggle 품종 3종(Dog Breed Identification / Oxford-IIIT Pet / Dogs of the World)은
7일 스프린트 스코프에서 제외 (사유는 `docs/01_7일_스프린트_계획.md` 참고).

> 원본 데이터는 저장소에 커밋하지 않습니다 (`.gitignore` 참고). `Data/` 아래에 배치하세요.

## 환경 설정

```bash
cd "DL PROJECT"
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # 정확히 같은 버전이 필요하면 requirements-lock.txt 사용
```

- MacBook Pro M5 (Apple Silicon) 기준 PyTorch MPS 가속 확인됨 (`torch.backends.mps.is_available() == True`)
- Apple Silicon이 아니면 `torch`가 자동으로 CPU(또는 CUDA)를 쓴다 — 코드 수정 불필요
  (`src/models/backbones.py`의 `get_device()`가 자동 감지)
- 확인: `python scripts/eval_reid.py --dataset mpdd` 실행 후 `Rank-1=...` 출력되면 정상

## 진행 상황
- [x] 프로젝트 기획, 과제 요건 확정
- [x] 데이터 확보 + EDA (MPDD, DogFaceNet, animal.go.kr)
- [x] 평가 지표(CMC/mAP) + E0 베이스라인 (`metadata/results.csv`)
- [ ] E1 임베딩 모델 파인튜닝
- [ ] E2 Prototypical Network 학습
- [ ] 정성분석 + 데모 앱
- [ ] 보고서 · 발표자료
