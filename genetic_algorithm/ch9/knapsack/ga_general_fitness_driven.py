# ga_general_fitness_driven.py
import random

from individual import Individual
from random_individual_shifted_zeros import create_random_individual
from random_set_generator import random_set_generator
from toolbox import (
    selection_rank_with_elite, crossover_fitness_driven_one_point,
    mutation_fitness_driven_bit_flip, plot_stats, stats,
    crossover_operation, mutation_operation,
)

# 교차 함수: 부모의 유전자 리스트를 기반으로 한 점 교차 수행
def crossover(parent1, parent2):
    return crossover_fitness_driven_one_point(parent1, parent2)

# 돌연변이 함수: fitness-driven 비트 플립 돌연변이 수행 (최대 3회 시도)
def mutate(ind):
    return mutation_fitness_driven_bit_flip(ind, max_tries=3)

# 선택 함수: 랭크 선택 및 엘리트 보존을 통한 다음 세대 선정
def select(population):
    return selection_rank_with_elite(population, elite_size=2)

random.seed(68)

# 아이템 집합 생성 및 개체 클래스 초기화
items = random_set_generator(1, 100, 0.1, 7, 200)
Individual.set_items(items)
Individual.set_max_weight(10)

# GA 파라미터 설정
POPULATION_SIZE = 100
CROSSOVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.2
MAX_GENERATIONS = 50

# 초기 개체군 생성
first_population = [create_random_individual(len(items), zeros=30) for _ in range(POPULATION_SIZE)]
population = first_population.copy()
fitness_list = [ind.fitness for ind in population]
fit_avg = [sum(fitness_list) / len(population)]
fit_best = [max(fitness_list)]
fit_best_ever = [max(fitness_list + fit_best)]
best_ind = random.choice(first_population)
generation_number = 0

# 세대마다 선택, 교차, 돌연변이 수행하면서 통계 업데이트
while generation_number < MAX_GENERATIONS:
    generation_number += 1
    offspring = select(population)
    # 도구 모듈의 crossover_operation을 통해 전체 개체군에 대해 교차 수행
    crossed_offspring = crossover_operation(offspring, crossover, CROSSOVER_PROBABILITY)
    # 도구 모듈의 mutation_operation을 통해 전체 개체군에 대해 돌연변이 수행
    mutated_offspring = mutation_operation(crossed_offspring, mutate, MUTATION_PROBABILITY)
    population = mutated_offspring.copy()
    
    # 각 세대의 최고 개체와 통계 정보를 업데이트
    best_ind, fit_avg, fit_best, fit_best_ever = stats(population, best_ind, fit_avg, fit_best, fit_best_ever)

# 세대별 통계 플롯 출력
plot_stats(fit_avg, fit_best_ever, "General Knapsack Problem")
print(f'Best Fitness: {best_ind.fitness}')
print(f'Total Number of Individuals: {Individual.counter}')
