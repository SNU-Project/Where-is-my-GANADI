# 참고 문헌

## Yeon2025_FewShotDogFaceID_MetaLearning.pdf

연수민, 배지호, 부석준, 최상민, 이수원. "메타학습 기반 소수샷 반려견 얼굴 식별"
(Few-Shot Dog Face Identification via Meta-Learning). Journal of KIIT, Vol. 23, No. 10,
pp. 1-9, Oct. 2025. https://doi.org/10.14801/jkiit.2025.23.10.1

**우리 프로젝트와의 관련성**: 우리가 채택한 **DogFaceNet 데이터셋을 동일하게 사용**한다.
개체당 이미지 수가 2~41장(중앙값 5장)으로 극히 적은 상황에서, 일반 전이학습 파인튜닝과
메타러닝(Prototypical Networks, Meta-DeepBDC) 기반 소수샷 학습을 비교한다.

| 방법 | 1-shot | 3-shot | 5-shot |
|---|---|---|---|
| 전이학습 파인튜닝(Baseline) | 18.31% | 23.26% | 26.83% |
| Prototypical Networks | 51.22% | 67.14% | 68.96% |
| **Meta-DeepBDC (최고)** | **64.01%** | **77.51%** | **82.36%** |

(ResNet-12 백본, 20-way 분류 정확도, DogFaceNet 데이터셋 기준)

**결론이 우리 계획에 미친 영향**: `01_7일_스프린트_계획.md`의 E2를 Triplet loss에서
**Prototypical Network 방식 에피소드 학습**으로 변경. 근거는 이 논문이 동일 데이터셋에서
일반 파인튜닝 대비 메타러닝이 압도적으로 우수함(1-shot 기준 3.5배)을 실증했기 때문.
