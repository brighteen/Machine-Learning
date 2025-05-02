# 개선된 유전 알고리즘 실험 결과 #1 (binary 인코딩)

실험 일시: 2025-05-01 16:08:23

## 실험 설정
- encoding_type: binary
- population_size: 100
- max_generations: 200
- initial_crossover_prob: 0.9
- final_crossover_prob: 0.7
- initial_mutation_prob: 0.3
- final_mutation_prob: 0.1
- elite_size: 3
- bits: 24

## 실험 결과

### 실험 1
- 연산자 조합:
  - 선택: selection_rank_with_elite
  - 교차: crossover_one_point
  - 돌연변이: mutation_bit_flip
- 결과:
  - 수렴 세대 수: 23
  - 최소값 f(x): -2.848230
  - 최적 x: -1.823476
  - 이진 표현: 100100011111010001001000

### 실험 2
- 연산자 조합:
  - 선택: selection_tournament
  - 교차: crossover_n_point
  - 돌연변이: mutation_bit_flip
- 결과:
  - 수렴 세대 수: 23
  - 최소값 f(x): -2.848230
  - 최적 x: -1.823476
  - 이진 표현: 100100011111010001001000

### 실험 3
- 연산자 조합:
  - 선택: hybrid
  - 교차: crossover_uniform
  - 돌연변이: mutation_fitness_driven_bit_flip
- 결과:
  - 수렴 세대 수: 18
  - 최소값 f(x): -2.848230
  - 최적 x: -1.823476
  - 이진 표현: 100100011111010001001000

## 통계 분석
- 평균 수렴 세대 수: 21.33
- 전체 최소값: -2.848230
- 최적의 연산자 조합:
  - 선택 연산자: selection_rank_with_elite
  - 교차 연산자: crossover_one_point
  - 돌연변이 연산자: mutation_bit_flip
  - 최적 x: -1.823476
  - 이진 표현: 100100011111010001001000