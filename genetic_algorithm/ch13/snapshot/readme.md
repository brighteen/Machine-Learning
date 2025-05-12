# 스냅샷(Snapshot) 최적화 기법

이 폴더는 유전 알고리즘의 실행 상태를 저장하고 복원하는 스냅샷(snapshot) 기법을 구현한 코드를 포함합니다.

## 설명

유전 알고리즘은 실행 시간이 길어질 수 있으며, 중간에 중단되거나 오류가 발생할 경우 처음부터 다시 시작해야 하는 문제가 있습니다. 스냅샷 기법은 알고리즘의 현재 상태를 저장하여 나중에 그 지점부터 다시 시작할 수 있게 해줍니다.

### 주요 기능

- 인구(population)를 JSON 파일로 저장
- 저장된 파일에서 인구를 복원
- 간단하고 직관적인 인터페이스

### 핵심 파일

- `individual.py`: 스냅샷 기능이 구현된 개체 클래스 및 인구 저장/복원 함수
- `test.py`: 저장된 개체 정보를 불러와서 분석하는 예제 코드

## 활용

이 기법은 다음과 같은 상황에서 특히 유용합니다:

- 장시간 실행되는 유전 알고리즘에서 중간 결과 저장
- 시스템 장애 또는 중단 후 계속 진행
- 다양한 매개변수 설정으로 동일한 중간 상태에서 실험

## 스냅샷 과정 상세 설명

유전 알고리즘에서의 스냅샷 과정은 다음과 같은 단계로 이루어집니다:

### 1. 스냅샷 저장 과정

1. **인구 상태 수집**: 현재 세대의 모든 개체에서 유전자 정보를 추출합니다.
2. **직렬화**: 추출한 유전자 정보를 JSON 형식으로 변환합니다.
3. **파일 저장**: 변환된 데이터를 지정된 경로의 파일에 저장합니다.

```python
# 인구 저장 예시 (individual.py의 dump_population 함수 사용)
from individual import dump_population

# 100개의 개체를 가진 인구 생성 (예시)
population = [Individual([random.randint(0, 100)]) for _ in range(100)]

# 저장 경로 설정 (tmp 폴더에 저장)
save_path = os.path.join('tmp', 'population_genes.json')

# 인구 저장
dump_population(population, save_path)
print(f"인구가 {save_path}에 저장되었습니다.")
```

### 2. 스냅샷 복원 과정

1. **파일 로드**: 저장된 JSON 파일을 읽습니다.
2. **역직렬화**: JSON 데이터를 파싱하여 유전자 리스트를 추출합니다.
3. **개체 재생성**: 추출한 유전자 정보로 개체 인스턴스를 생성합니다.
4. **인구 복원**: 생성된 개체들로 인구를 구성합니다.

```python
# 인구 복원 예시 (individual.py의 restore_population 함수 사용)
from individual import restore_population

# 저장된 경로에서 인구 복원
load_path = os.path.join('tmp', 'population_genes.json')
restored_population = restore_population(load_path)

# 복원된 인구 정보 출력
print(f"복원된 인구 크기: {len(restored_population)}")
```

### 3. 실행 주기 설정

실제 유전 알고리즘 실행 시에는 주기적으로 스냅샷을 저장하는 것이 좋습니다:

```python
# 유전 알고리즘 실행 중 주기적 스냅샷 저장 예시 (의사 코드)
generation = 0
max_generations = 1000
snapshot_interval = 50  # 50세대마다 스냅샷 저장

while generation < max_generations:
    # 유전 알고리즘 실행 코드
    # ...
    
    # 주기적 스냅샷 저장
    if generation % snapshot_interval == 0:
        snapshot_path = os.path.join('tmp', f'population_gen_{generation}.json')
        dump_population(population, snapshot_path)
        print(f"세대 {generation}의 스냅샷이 저장되었습니다.")
    
    generation += 1
```

## 성능 고려사항

- **저장 빈도**: 너무 자주 스냅샷을 저장하면 성능 저하가 발생할 수 있습니다. 애플리케이션에 맞게 적절한 간격을 설정하세요.
- **파일 크기**: 인구 크기가 크거나 유전자가 복잡한 경우 파일 크기가 커질 수 있습니다.
- **병렬 처리**: 대규모 병렬 유전 알고리즘을 사용하는 경우, 동시성 이슈를 고려하여 스냅샷 저장 로직을 설계해야 합니다.

## 활용 예제

`test.py` 파일에서는 저장된 스냅샷 파일을 불러와 분석하는 방법의 예시를 제공합니다:

1. 직접 JSON 파일을 읽어 유전자 정보 분석
2. `restore_population()` 함수를 사용하여 개체 복원 및 사용

이 예제를 통해 스냅샷 데이터를 어떻게 활용할 수 있는지 이해할 수 있습니다.
