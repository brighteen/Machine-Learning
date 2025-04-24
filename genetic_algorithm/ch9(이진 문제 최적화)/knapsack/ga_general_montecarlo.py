# ga_general_montecarlo.py
import random
import matplotlib.pyplot as plt

from individual import Individual
from random_individual_shifted_zeros import create_random_individual
from random_set_generator import random_set_generator
from toolbox import (
    crossover_one_point, mutation_bit_flip, selection_rank_with_elite,
)

# 교차 함수: 부모의 gene_list에 한 점 교차 적용 후 두 자식 개체 생성
def crossover(parent1, parent2):
    child1_genes, child2_genes = crossover_one_point(parent1.gene_list, parent2.gene_list)
    return Individual(child1_genes), Individual(child2_genes)

# 돌연변이 함수: gene_list에 단순 비트 플립 돌연변이 적용
def mutate(ind):
    mutated_gene = mutation_bit_flip(ind.gene_list)
    return Individual(mutated_gene)

# 선택 함수: 엘리트 보존을 포함한 랭크 기반 선택 수행
def select(population):
    return selection_rank_with_elite(population, elite_size=2)

random.seed(63)

# 아이템 집합 생성 및 개체 클래스 초기화
items = random_set_generator(1, 100, 0.1, 7, 200)
Individual.set_items(items)
Individual.set_max_weight(10)

# GA 파라미터 설정
POPULATION_SIZE = 80
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.2
MAX_GENERATIONS = 70
RUNS = 100

# 결과를 기록할 리스트 초기화
best = []           # 각 실행의 최고 fitness 기록
total_numbers = []  # 생성된 전체 개체 수 기록

# RUNS 번 알고리즘 실행
for _ in range(RUNS):
    # 초기 개체군 생성 (랜덤 gene_list 사용)
    first_population = [create_random_individual(len(items), zeros=30) for _ in range(POPULATION_SIZE)]
    Individual.counter = 0   # 개체 생성 카운터 초기화
    best_individual = random.choice(first_population)  # 초기 최고 개체 선택
    generation_number = 0

    population = first_population.copy()

    while generation_number < MAX_GENERATIONS:
        generation_number += 1

        # [선택] 랭크 기반 선택으로 자식 개체군 생성
        offspring = select(population)

        # [교차] 자식들을 짝지어 교차 연산 수행
        crossed_offspring = []
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_PROBABILITY:
                kid1, kid2 = crossover(ind1, ind2)
                crossed_offspring.append(kid1)
                crossed_offspring.append(kid2)
            else:
                crossed_offspring.append(ind1)
                crossed_offspring.append(ind2)

        # [돌연변이] 각 개체에 대해 돌연변이 확률 적용
        mutated_offspring = []
        for mutant in crossed_offspring:
            if random.random() < MUTATION_PROBABILITY:
                new_mutant = mutate(mutant)
                mutated_offspring.append(new_mutant)
            else:
                mutated_offspring.append(mutant)

        population = mutated_offspring.copy()

        # 이번 세대에서 최고 개체 선택
        best_of_generation = max(population, key=lambda ind: ind.fitness)
        if best_individual.fitness < best_of_generation.fitness:
            best_individual = best_of_generation

    best.append(best_individual.fitness)
    total_numbers.append(Individual.counter)

# 전체 실행의 평균 fitness 계산 및 결과 플롯
avg_fitness = sum(best) / len(best)
plt.plot(best)
plt.title(f'Average fitness: {avg_fitness} \n'
          f'Average number of individuals: {sum(total_numbers) / len(total_numbers)}')
plt.axhline(y=avg_fitness, color='r', linestyle='-')
plt.show()
