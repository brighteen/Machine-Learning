# 유전 알고리즘 실험 결과 #6 (이진 인코딩)

실험 일시: 2025-04-27 17:33:55

## 실험 설정
- 인구 크기: 20
- 교차 확률: 0.8
- 돌연변이 확률: 0.2
- 최대 세대 수: 100
- 수렴 임계값: 1e-09
- 수렴 판단 세대 수: 5
- 이진 표현 비트 수: 16

## 실험 결과
| 선택 연산자 | 교차 연산자 | 돌연변이 연산자 | 세대 수 | 최소값 f(x) | 최적 x | 이진 표현 |
|------------|------------|----------------|---------|------------|--------|----------|
| selection_rank | crossover_one_point | mutation_fitness_driven_bit_flip | 17 | -7.560619 | 11.248054 | 0101111111111011 |
| selection_stochastic_universal_sampling | crossover_uniform | mutation_bit_flip | 30 | -7.560619 | 11.248054 | 0101111111111011 |
| selection_tournament | crossover_uniform | mutation_bit_flip | 31 | -7.560619 | 11.248054 | 0101111111111011 |
| selection_rank | crossover_uniform | mutation_bit_flip | 19 | -7.560619 | 11.248512 | 0101111111111100 |
| selection_rank_with_elite | crossover_uniform | mutation_fitness_driven_bit_flip | 27 | -7.560619 | 11.248512 | 0101111111111100 |
| selection_tournament | crossover_one_point | mutation_fitness_driven_bit_flip | 41 | -7.560619 | 11.248512 | 0101111111111100 |
| selection_stochastic_universal_sampling | crossover_uniform | mutation_fitness_driven_bit_flip | 28 | -7.560618 | 11.248970 | 0101111111111101 |
| selection_proportional | crossover_n_point | mutation_bit_flip | 79 | -7.560618 | 11.249428 | 0101111111111110 |
| selection_proportional | crossover_one_point | mutation_fitness_driven_bit_flip | 50 | -7.560615 | 11.246223 | 0101111111110111 |
| selection_rank | crossover_one_point | mutation_bit_flip | 11 | -7.560615 | 11.246223 | 0101111111110111 |
| selection_proportional | crossover_uniform | mutation_fitness_driven_bit_flip | 79 | -7.560615 | 11.250343 | 0110000000000000 |
| selection_rank | crossover_n_point | mutation_bit_flip | 21 | -7.560615 | 11.250343 | 0110000000000000 |
| selection_rank | crossover_n_point | mutation_fitness_driven_bit_flip | 28 | -7.560615 | 11.250343 | 0110000000000000 |
| selection_rank | crossover_uniform | mutation_fitness_driven_bit_flip | 23 | -7.560615 | 11.250343 | 0110000000000000 |
| selection_rank_with_elite | crossover_n_point | mutation_bit_flip | 22 | -7.560615 | 11.250343 | 0110000000000000 |
| selection_rank_with_elite | crossover_uniform | mutation_bit_flip | 31 | -7.560615 | 11.250343 | 0110000000000000 |
| selection_tournament | crossover_n_point | mutation_fitness_driven_bit_flip | 37 | -7.560615 | 11.250343 | 0110000000000000 |
| selection_stochastic_universal_sampling | crossover_one_point | mutation_fitness_driven_bit_flip | 21 | -7.560610 | 11.251259 | 0110000000000010 |
| selection_rank_with_elite | crossover_one_point | mutation_fitness_driven_bit_flip | 21 | -7.560607 | 11.251717 | 0110000000000011 |
| selection_tournament | crossover_one_point | mutation_bit_flip | 42 | -7.560600 | 11.252632 | 0110000000000101 |
| selection_proportional | crossover_one_point | mutation_bit_flip | 71 | -7.560557 | 11.240272 | 0101111111101010 |
| selection_proportional | crossover_n_point | mutation_fitness_driven_bit_flip | 64 | -7.560557 | 11.240272 | 0101111111101010 |
| selection_stochastic_universal_sampling | crossover_n_point | mutation_fitness_driven_bit_flip | 10 | -7.560541 | 11.257210 | 0110000000001111 |
| selection_tournament | crossover_uniform | mutation_fitness_driven_bit_flip | 20 | -7.560466 | 11.235694 | 0101111111100000 |
| selection_proportional | crossover_uniform | mutation_bit_flip | 100 | -7.560443 | 11.234779 | 0101111111011110 |
| selection_stochastic_universal_sampling | crossover_n_point | mutation_bit_flip | 9 | -7.559433 | 11.283303 | 0110000001001000 |
| selection_rank_with_elite | crossover_n_point | mutation_fitness_driven_bit_flip | 12 | -7.559402 | 11.283761 | 0110000001001001 |
| selection_tournament | crossover_n_point | mutation_bit_flip | 21 | -7.559402 | 11.283761 | 0110000001001001 |
| selection_rank_with_elite | crossover_one_point | mutation_bit_flip | 6 | -7.557150 | 11.188543 | 0101111101111001 |
| selection_stochastic_universal_sampling | crossover_one_point | mutation_bit_flip | 20 | -7.553896 | 11.165197 | 0101111101000110 |

## 통계 분석
- 평균 수렴 세대 수: 33.03
- 전체 최소값: -7.560619
- 최적의 연산자 조합:
  - 선택 연산자: selection_rank
  - 교차 연산자: crossover_one_point
  - 돌연변이 연산자: mutation_fitness_driven_bit_flip
  - 최적 x: 11.248054
  - 이진 표현: 0101111111111011