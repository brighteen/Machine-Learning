# 신경망의 노드 가지치기를 위한 유전 알고리즘 (NNPGA)

논문 "신경망의 노드 가지치기를 위한 유전 알고리즘" (허기수, 오일석, 전북대학교)을 모듈화하여 구현한 시스템입니다.

## 시스템 구조

### 1. 핵심 모듈들

- **`nnpga_core.py`**: NNPGA 알고리즘의 핵심 구현
  - 염색체 생성 및 관리
  - 적합도 평가 (신경망 학습 및 성능 측정)
  - 선택, 교차, 돌연변이 연산자
  - 유전 알고리즘 진화 프로세스

- **`dataset_loader.py`**: UCI 데이터셋 관리
  - 논문에서 사용한 13개 데이터셋 자동 다운로드
  - 데이터 정규화 및 전처리
  - 데이터셋 정보 제공

- **`experiment_runner.py`**: 실험 관리 및 결과 분석
  - 10-fold 교차 검증
  - 다양한 가지치기 비율 실험
  - 결과 집계 및 통계 계산
  - 논문 표 재현

- **`main.py`**: 사용자 인터페이스
  - 메뉴 기반 인터페이스
  - 단일/전체 데이터셋 실험
  - 커스텀 실험 설정

### 2. 지원하는 데이터셋 (논문 표 1)

| Dataset | Samples | Features | Classes | MLP Structure |
|---------|---------|----------|---------|---------------|
| balance-scale | 625 | 4 | 3 | 4-12-3 |
| german | 1000 | 24 | 2 | 24-8-2 |
| glass | 214 | 9 | 7 | 9-28-7 |
| heart-statlog | 270 | 13 | 2 | 13-8-2 |
| ionosphere | 351 | 34 | 2 | 34-8-2 |
| iris | 150 | 4 | 3 | 4-12-3 |
| pima-indians | 768 | 8 | 2 | 8-8-2 |
| segmentation | 2310 | 19 | 7 | 19-24-7 |
| sonar | 208 | 60 | 2 | 60-8-2 |
| vehicle | 846 | 18 | 4 | 18-16-4 |
| vowel | 990 | 10 | 11 | 10-22-11 |
| waveform-noise | 5000 | 40 | 3 | 40-12-3 |
| wine | 178 | 13 | 3 | 13-12-3 |

## 사용법

### 1. 기본 실행
```bash
python main.py
```

메뉴가 나타나면 원하는 옵션을 선택:
1. 단일 데이터셋 실험
2. 전체 데이터셋 실험 (논문 표 4 재현)
3. 데이터셋 정보 확인
4. 커스텀 실험
5. 종료

### 2. 간단한 데모
```bash
python demo.py
```

### 3. 프로그래매틱 사용

```python
from nnpga_core import NNPGA
from dataset_loader import DatasetLoader
from experiment_runner import ExperimentRunner

# 실험 러너 생성
runner = ExperimentRunner()

# 단일 실험 수행
result = runner.run_single_experiment('iris', 0.3)
print(f"압축률: {result['compression_ratio']:.2f}")
print(f"정확도 손실: {result['accuracy_loss']:.4f}")

# 전체 실험 수행 (시간 소요)
results = runner.run_full_experiments()
```

### 4. 커스텀 GA 매개변수 사용

```python
# 커스텀 NNPGA 인스턴스
nnpga = NNPGA(
    population_size=30,
    max_generations=200,
    mutation_prob=0.05
)

# 커스텀 실험 러너
runner = ExperimentRunner(nnpga)
result = runner.run_single_experiment('wine', 0.25)
```

## 알고리즘 매개변수 (논문 기준)

- **해집단 크기 (N)**: 20
- **교차 확률 (pc)**: 1.0 (매번 교차)
- **돌연변이 확률 (pm)**: 0.01
- **선택압 (q)**: 0.25 (순위기반 선택)
- **최대 세대 수 (T)**: 150
- **은닉 노드 수**: 출력 노드 수 × 4

## 결과 분석

시스템은 다음과 같은 메트릭을 제공합니다:

- **원본 정확도**: 가지치기 전 신경망 성능
- **가지치기 후 정확도**: NNPGA 적용 후 성능
- **압축률**: 네트워크 크기 감소 비율
- **정확도 손실**: 성능 저하 정도

모든 결과는 10-fold 교차 검증을 통해 평균과 표준편차로 보고됩니다.

## 확장성

이 모듈화된 구조를 통해 쉽게 확장 가능합니다:

1. **새로운 데이터셋 추가**: `dataset_loader.py`에 로딩 함수 추가
2. **다른 선택/교차/돌연변이 연산자**: `nnpga_core.py`에 메소드 추가
3. **다른 신경망 구조**: 적합도 함수 수정
4. **다른 평가 메트릭**: `experiment_runner.py`에 계산 로직 추가

## 요구사항

- Python 3.7+
- scikit-learn
- pandas
- numpy

```bash
pip install scikit-learn pandas numpy
```

## 논문 재현

전체 실험을 실행하면 논문의 표 4를 재현할 수 있습니다. 각 데이터셋에 대해 다양한 가지치기 비율(10%, 20%, 30%, 40%, 50%)에서의 성능을 측정하고 비교합니다.

실행 시간은 데이터셋 크기와 실험 설정에 따라 달라지며, 전체 실험은 상당한 시간이 소요될 수 있습니다.
