# 찾아줘, 가나디 (Where is my GANADI)

보호소 유기견 이미지와 실종견 사진을 매칭하는 **AI 기반 실종견 찾기 서비스**.

실종견 사진 1장을 업로드하면 보호소 유기견 DB에서 **같은 개체일 가능성이 높은 순서로 Top-K 후보**를 제시합니다.
사람 얼굴인식과 유사한 **개체 재식별(Re-Identification) / 이미지 검색** 문제로 접근합니다.

## 문서
- [프로젝트 기획서 / 설계서](docs/00_기획서.md)

## 데이터셋
| 데이터셋 | 역할 |
|---|---|
| [Multi-pose Dog Dataset](https://data.mendeley.com/datasets/v5j6m8dzhv/1) | 핵심 Re-ID 학습·평가 |
| [PetFace](https://dahlian00.github.io/PetFacePage/) | 임베딩 사전학습 · 1:1 검증 |
| [Dog Breed Identification](https://www.kaggle.com/competitions/dog-breed-identification/data) | 품종 분류기 |
| [Cats and Dogs Breeds (Oxford-IIIT Pet)](https://www.kaggle.com/datasets/zippyz/cats-and-dogs-breeds-classification-oxford-dataset) | 품종 분류 보조 · 크롭 |
| [Dogs of the World](https://www.kaggle.com/datasets/lextoumbourou/dogs-world) | 품종 분류 보강 |
| [국가동물보호정보시스템](https://www.animal.go.kr/front/index.do) | 실제 보호소 갤러리 · 도메인 테스트 · 데모 DB |

> 원본 데이터는 저장소에 커밋하지 않습니다 (`.gitignore` 참고). `Data/` 아래에 배치하세요.

## 환경
- MacBook Pro M5 (Apple Silicon) · PyTorch MPS
- 설치: `pip install -r requirements.txt` (준비 예정)

## 진행 상황
- [x] 프로젝트 기획 초안
- [ ] 레포 스캐폴딩 (`src/`, `scripts/`, `configs/`)
- [ ] EDA
- [ ] 평가 지표(CMC/mAP) + 베이스라인
- [ ] 임베딩 모델 학습
- [ ] 품종 분류기
- [ ] 데모 앱
