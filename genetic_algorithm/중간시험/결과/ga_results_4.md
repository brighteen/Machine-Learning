# 유전 알고리즘 실험 결과 #4

실험 일시: 2025-04-27 17:27:53

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
| selection_rank | crossover_blend | mutation_fitness_driven_random_deviation | 32 | -7.560616 | 11.250000 |
| selection_rank_with_elite | crossover_blend | mutation_random_deviation | 17 | -7.560616 | 11.250000 |
| selection_rank_with_elite | crossover_linear | mutation_random_deviation | 13 | -7.560616 | 11.250000 |
| selection_rank_with_elite | crossover_linear | mutation_fitness_driven_random_deviation | 29 | -7.560616 | 11.250000 |
| selection_stochastic_universal_sampling | crossover_blend | mutation_fitness_driven_random_deviation | 19 | -7.560616 | 11.250000 |
| selection_stochastic_universal_sampling | crossover_linear | mutation_fitness_driven_random_deviation | 14 | -7.560616 | 11.250000 |
| selection_tournament | crossover_blend | mutation_random_deviation | 19 | -7.560616 | 11.250000 |
| selection_tournament | crossover_blend | mutation_fitness_driven_random_deviation | 13 | -7.560616 | 11.250000 |
| selection_tournament | crossover_linear | mutation_random_deviation | 13 | -7.560616 | 11.250000 |
| selection_stochastic_universal_sampling | crossover_blend | mutation_random_deviation | 15 | -7.560486 | 11.260000 |
| selection_rank | crossover_linear | mutation_random_deviation | 17 | -7.560296 | 11.230000 |
| selection_rank | crossover_blend | mutation_random_deviation | 18 | -7.559844 | 11.220000 |
| selection_proportional | crossover_uniform | mutation_fitness_driven_random_deviation | 6 | -7.558013 | 11.196488 |
| selection_tournament | crossover_linear | mutation_fitness_driven_random_deviation | 15 | -7.556336 | 11.181923 |
| selection_stochastic_universal_sampling | crossover_uniform | mutation_fitness_driven_random_deviation | 16 | -7.553396 | 11.162175 |
| selection_rank_with_elite | crossover_blend | mutation_fitness_driven_random_deviation | 17 | -7.552729 | 11.338911 |
| selection_rank_with_elite | crossover_uniform | mutation_fitness_driven_random_deviation | 8 | -7.528353 | 11.066866 |
| selection_rank | crossover_uniform | mutation_fitness_driven_random_deviation | 6 | -7.508747 | 11.018550 |
| selection_tournament | crossover_uniform | mutation_fitness_driven_random_deviation | 9 | -7.482516 | 10.966700 |
| selection_tournament | crossover_uniform | mutation_random_deviation | 6 | -7.438236 | 11.611488 |
| selection_proportional | crossover_linear | mutation_fitness_driven_random_deviation | 22 | -7.426778 | 11.628606 |
| selection_rank | crossover_uniform | mutation_random_deviation | 6 | -7.419598 | 11.638990 |
| selection_proportional | crossover_uniform | mutation_random_deviation | 6 | -6.934675 | 12.111320 |
| selection_stochastic_universal_sampling | crossover_linear | mutation_random_deviation | 7 | -6.703559 | 10.310253 |
| selection_stochastic_universal_sampling | crossover_uniform | mutation_random_deviation | 6 | -6.199424 | 15.000000 |
| selection_rank_with_elite | crossover_uniform | mutation_random_deviation | 6 | -5.932153 | 14.861109 |
| selection_rank | crossover_linear | mutation_fitness_driven_random_deviation | 13 | -5.437369 | 14.545644 |

## 통계 분석
- 평균 수렴 세대 수: 19.87
- 전체 최소값: -7.560616
- 최적의 연산자 조합:
  - 선택 연산자: selection_proportional
  - 교차 연산자: crossover_blend
  - 돌연변이 연산자: mutation_random_deviation
  - 최적 x: 11.250000