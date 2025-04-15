import random
import matplotlib.pyplot as plt
from individual import Individual
from random_individual_shifted_zeros import create_random_individual
from random_set_generator import random_set_generator
from toolbox import (
    crossover_one_point,         # 기본 one-point 교차 연산
    mutation_bit_flip,           # 기본 bit flip 돌연변이 연산
    selection_rank_with_elite,   # 엘리트 선택 연산
)

# 교차 연산 함수: 두 부모의 유전자 배열을 one-point 방식으로 교환
def crossover(parent1, parent2):
    child1_genes, child2_genes = crossover_one_point(parent1.gene_list, parent2.gene_list)
    return Individual(child1_genes), Individual(child2_genes)

# 돌연변이 연산 함수: 유전자 배열에서 하나의 위치를 bit flip
def mutate(ind):
    mutated_gene = mutation_bit_flip(ind.gene_list)
    return Individual(mutated_gene)

# 선택 연산 함수: 엘리트 선택 적용
def select(population):
    return selection_rank_with_elite(population, elite_size=2)

random.seed(63)
# 아이템 집합 생성
items = random_set_generator(1, 100, 0.1, 7, 200)
Individual.set_items(items)
Individual.set_max_weight(10)

# GA 파라미터
POPULATION_SIZE = 100
CROSSOVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.2
MAX_GENERATIONS = 100

# 초기 개체군 생성
first_population = [create_random_individual(len(items), zeros=30) for _ in range(POPULATION_SIZE)]
best_individual = random.choice(first_population)
stats_fitness_average = []
stats_fitness_best = []
generation_number = 0
population = first_population.copy()

# GA 메인 루프
while generation_number < MAX_GENERATIONS:
    generation_number += 1
    # 선택
    offspring = select(population)
    # 교차: 전체 개체군에 대해 CROSSOVER_PROBABILITY 확률로 수행
    crossed_offspring = []
    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CROSSOVER_PROBABILITY:
            kid1, kid2 = crossover(ind1, ind2)
            crossed_offspring.extend([kid1, kid2])
        else:
            crossed_offspring.extend([ind1, ind2])
    # 돌연변이
    mutated_offspring = []
    for mutant in crossed_offspring:
        if random.random() < MUTATION_PROBABILITY:
            new_mutant = mutate(mutant)
            mutated_offspring.append(new_mutant)
        else:
            mutated_offspring.append(mutant)
    population = mutated_offspring.copy()
    # 세대 내 최적 개체 찾기 및 통계 업데이트
    best_of_generation = max(population, key=lambda ind: ind.fitness)
    if best_individual.fitness < best_of_generation.fitness:
        best_individual = best_of_generation
    stats_fitness_average.append(sum([ind.fitness for ind in population]) / len(population))
    stats_fitness_best.append(best_individual.fitness)

# 세대별 평균 및 최고 적합도 플롯
plt.plot(stats_fitness_average, label="Average Fitness of Generation")
plt.plot(stats_fitness_best, label="Best Fitness")
plt.title("General Knapsack Problem")
plt.legend(loc="lower right")
plt.show()

print(f'Best Fitness: {best_individual.fitness}')
print(f'Total Number of Individuals: {Individual.counter}')
