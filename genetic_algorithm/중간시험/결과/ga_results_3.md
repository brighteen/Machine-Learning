# 유전 알고리즘 실험 결과 #3

실험 일시: 2025-04-27 17:26:11

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
| selection_proportional | crossover_linear | mutation_random_deviation | 29 | -7.560616 | 11.250000 |
| selection_rank_with_elite | crossover_blend | mutation_random_deviation | 26 | -7.560616 | 11.250000 |
| selection_rank_with_elite | crossover_linear | mutation_random_deviation | 14 | -7.560616 | 11.250000 |
| selection_tournament | crossover_blend | mutation_random_deviation | 15 | -7.560616 | 11.250000 |
| selection_tournament | crossover_linear | mutation_random_deviation | 12 | -7.560616 | 11.250000 |
| selection_proportional | crossover_blend | mutation_random_deviation | 14 | -7.560553 | 11.240000 |
| selection_rank | crossover_blend | mutation_random_deviation | 22 | -7.560296 | 11.230000 |
| selection_rank | crossover_linear | mutation_random_deviation | 26 | -7.559198 | 11.210000 |
| selection_proportional | crossover_uniform | mutation_random_deviation | 7 | -7.553223 | 11.336016 |
| selection_stochastic_universal_sampling | crossover_blend | mutation_random_deviation | 6 | -7.532551 | 11.420000 |
| selection_stochastic_universal_sampling | crossover_linear | mutation_random_deviation | 6 | -6.703111 | 10.310000 |
| selection_rank_with_elite | crossover_uniform | mutation_random_deviation | 6 | -6.478362 | 12.435758 |
| selection_stochastic_universal_sampling | crossover_uniform | mutation_random_deviation | 6 | -6.173952 | 10.039846 |
| selection_tournament | crossover_uniform | mutation_random_deviation | 6 | -5.610822 | 14.669053 |
| selection_rank | crossover_uniform | mutation_random_deviation | 6 | -4.257832 | 4.561258 |

## 통계 분석
- 평균 수렴 세대 수: 13.40
- 전체 최소값: -7.560616
- 최적의 연산자 조합:
  - 선택 연산자: selection_proportional
  - 교차 연산자: crossover_linear
  - 돌연변이 연산자: mutation_random_deviation
  - 최적 x: 11.250000