# 유전 알고리즘 실험 결과 #5

실험 일시: 2025-04-27 17:28:14

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
| selection_proportional | crossover_blend | mutation_random_deviation | 100 | -7.560616 | 11.250000 |
| selection_proportional | crossover_blend | mutation_fitness_driven_random_deviation | 100 | -7.560616 | 11.250000 |
| selection_proportional | crossover_linear | mutation_random_deviation | 28 | -7.560616 | 11.250000 |
| selection_proportional | crossover_linear | mutation_fitness_driven_random_deviation | 10 | -7.560616 | 11.250000 |
| selection_rank | crossover_blend | mutation_random_deviation | 18 | -7.560616 | 11.250000 |
| selection_rank | crossover_linear | mutation_random_deviation | 12 | -7.560616 | 11.250000 |
| selection_rank_with_elite | crossover_blend | mutation_random_deviation | 25 | -7.560616 | 11.250000 |
| selection_rank_with_elite | crossover_blend | mutation_fitness_driven_random_deviation | 17 | -7.560616 | 11.250000 |
| selection_rank_with_elite | crossover_linear | mutation_random_deviation | 50 | -7.560616 | 11.250000 |
| selection_rank_with_elite | crossover_linear | mutation_fitness_driven_random_deviation | 12 | -7.560616 | 11.250000 |
| selection_stochastic_universal_sampling | crossover_blend | mutation_random_deviation | 11 | -7.560616 | 11.250000 |
| selection_stochastic_universal_sampling | crossover_linear | mutation_random_deviation | 15 | -7.560616 | 11.250000 |
| selection_tournament | crossover_blend | mutation_random_deviation | 12 | -7.560616 | 11.250000 |
| selection_tournament | crossover_blend | mutation_fitness_driven_random_deviation | 15 | -7.560616 | 11.250000 |
| selection_stochastic_universal_sampling | crossover_linear | mutation_fitness_driven_random_deviation | 6 | -7.560162 | 11.270000 |
| selection_stochastic_universal_sampling | crossover_blend | mutation_fitness_driven_random_deviation | 16 | -7.559646 | 11.280000 |
| selection_rank | crossover_blend | mutation_fitness_driven_random_deviation | 29 | -7.559330 | 11.284805 |
| selection_rank | crossover_uniform | mutation_fitness_driven_random_deviation | 6 | -7.542427 | 11.386259 |
| selection_stochastic_universal_sampling | crossover_uniform | mutation_fitness_driven_random_deviation | 6 | -7.521335 | 11.048210 |
| selection_tournament | crossover_uniform | mutation_random_deviation | 10 | -7.516023 | 11.035184 |
| selection_rank_with_elite | crossover_uniform | mutation_random_deviation | 7 | -7.508228 | 11.017410 |
| selection_rank | crossover_linear | mutation_fitness_driven_random_deviation | 8 | -7.496151 | 11.510000 |
| selection_tournament | crossover_uniform | mutation_fitness_driven_random_deviation | 6 | -7.481084 | 11.539563 |
| selection_proportional | crossover_uniform | mutation_random_deviation | 9 | -7.407039 | 11.656582 |
| selection_stochastic_universal_sampling | crossover_uniform | mutation_random_deviation | 6 | -7.375170 | 10.815219 |
| selection_rank | crossover_uniform | mutation_random_deviation | 14 | -7.323775 | 10.758982 |
| selection_tournament | crossover_linear | mutation_random_deviation | 17 | -7.317460 | 11.766860 |
| selection_proportional | crossover_uniform | mutation_fitness_driven_random_deviation | 6 | -6.199424 | 15.000000 |
| selection_tournament | crossover_linear | mutation_fitness_driven_random_deviation | 7 | -6.131385 | 10.020000 |
| selection_rank_with_elite | crossover_uniform | mutation_fitness_driven_random_deviation | 11 | -5.296310 | 9.660079 |

## 통계 분석
- 평균 수렴 세대 수: 19.63
- 전체 최소값: -7.560616
- 최적의 연산자 조합:
  - 선택 연산자: selection_proportional
  - 교차 연산자: crossover_blend
  - 돌연변이 연산자: mutation_random_deviation
  - 최적 x: 11.250000