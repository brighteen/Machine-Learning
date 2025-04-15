import random
from individual import Individual
from random_individual_shifted_zeros import create_random_individual
from random_set_generator import random_set_generator
from toolbox import (
    selection_rank_with_elite,           # 선택 연산 함수 (엘리트 기반 선택)
    crossover_fitness_driven_one_point,    # 교차 연산 함수 (fitness-driven one-point)
    mutation_fitness_driven_bit_flip,      # 돌연변이 연산 함수 (fitness-driven bit flip)
    plot_stats,                            # 세대별 통계 플롯 함수
    stats,                                 # 세대별 통계 계산 함수
    crossover_operation,                   # 교차 연산 적용 함수 (전체 개체군에 적용)
    mutation_operation,                    # 돌연변이 연산 적용 함수 (전체 개체군에 적용)
)

# 교차 연산 함수 정의 (두 부모의 유전자를 교환)
def crossover(parent1, parent2):
    return crossover_fitness_driven_one_point(parent1, parent2)

# 돌연변이 연산 함수 정의
def mutate(ind):
    return mutation_fitness_driven_bit_flip(ind, max_tries=3)

# 선택 연산 함수 정의 (엘리트 선택 적용)
def select(population):
    return selection_rank_with_elite(population, elite_size=2)

random.seed(68)  # 시드 설정

# 아이템 집합 생성 및 설정
items = random_set_generator(1, 100, 0.1, 7, 200)
Individual.set_items(items)
Individual.set_max_weight(10)  # 배낭 최대 무게 제한 설정

# GA 파라미터
POPULATION_SIZE = 100
CROSSOVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.2
MAX_GENERATIONS = 50

# 초기 개체군 생성
first_population = [create_random_individual(len(items), zeros=30) for _ in range(POPULATION_SIZE)]
population = first_population.copy()

# 초기 개체군의 적합도 통계 계산
fitness_list = [ind.fitness for ind in population]
fit_avg = [sum(fitness_list) / len(population)]
fit_best = [max(fitness_list)]
fit_best_ever = [max(fitness_list + fit_best)]
best_ind = random.choice(first_population)
generation_number = 0

# GA 메인 루프: 지정된 세대만큼 반복
while generation_number < MAX_GENERATIONS:
    generation_number += 1
    # 선택 연산 적용
    offspring = select(population)
    # 교차 연산 적용 (전체 개체군에 대해)
    crossed_offspring = crossover_operation(offspring, crossover, CROSSOVER_PROBABILITY)
    # 돌연변이 연산 적용
    mutated_offspring = mutation_operation(crossed_offspring, mutate, MUTATION_PROBABILITY)
    population = mutated_offspring.copy()
    
    # 통계 업데이트: 현재 세대의 최고 적합도 및 평균 적합도 기록
    best_ind, fit_avg, fit_best, fit_best_ever = stats(population, best_ind, fit_avg, fit_best, fit_best_ever)

# 세대별 통계 플롯으로 결과 시각화
plot_stats(fit_avg, fit_best_ever, "General Knapsack Problem")
print(f'Best Fitness: {best_ind.fitness}')
print(f'Total Number of Individuals: {Individual.counter}')
