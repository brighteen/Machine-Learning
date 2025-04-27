# 유전 알고리즘 실험 결과

## 실험 설정
- 인구 크기: 20
- 교차 확률: 0.8
- 돌연변이 확률: 0.2
- 최대 세대 수: 100
- 수렴 임계값: 1e-09
- 수렴 판단 세대 수: 5

## 실험 결과
| 선택 연산자 | 교차 연산자 | 돌연변이 연산자 | 세대 수 | 최소값 f(x) | 최적 x |
|------------|------------|----------------|---------|------------|--------|
| selection_rank | crossover_blend | mutation_random_deviation | 20 | -7.560616 | 11.250000 |
| selection_rank | crossover_linear | mutation_random_deviation | 12 | -7.560616 | 11.250000 |
| selection_rank_with_elite | crossover_blend | mutation_random_deviation | 16 | -7.560616 | 11.250000 |
| selection_stochastic_universal_sampling | crossover_blend | mutation_random_deviation | 29 | -7.560616 | 11.250000 |
| selection_tournament | crossover_blend | mutation_random_deviation | 20 | -7.560616 | 11.250000 |
| selection_stochastic_universal_sampling | crossover_linear | mutation_random_deviation | 29 | -7.560579 | 11.241843 |
| selection_proportional | crossover_uniform | mutation_random_deviation | 19 | -7.560575 | 11.241550 |
| selection_rank_with_elite | crossover_linear | mutation_random_deviation | 15 | -7.560553 | 11.240000 |
| selection_proportional | crossover_linear | mutation_random_deviation | 90 | -7.560486 | 11.260000 |
| selection_stochastic_universal_sampling | crossover_uniform | mutation_random_deviation | 6 | -7.560252 | 11.228810 |
| selection_tournament | crossover_linear | mutation_random_deviation | 23 | -7.560162 | 11.270000 |
| selection_proportional | crossover_blend | mutation_random_deviation | 6 | -7.559844 | 11.220000 |
| selection_tournament | crossover_uniform | mutation_random_deviation | 12 | -7.557161 | 11.188635 |
| selection_rank_with_elite | crossover_uniform | mutation_random_deviation | 6 | -7.555563 | 11.320758 |
| selection_rank | crossover_uniform | mutation_random_deviation | 9 | -5.077163 | 13.608750 |

## 통계 분석
- 평균 수렴 세대 수: 20.80
- 전체 최소값: -7.560616
- 최적의 연산자 조합:
  - 선택 연산자: selection_rank
  - 교차 연산자: crossover_blend
  - 돌연변이 연산자: mutation_random_deviation
  - 최적 x: 11.250000