import random
import matplotlib.pyplot as plt
from individual import Individual
from random_individual_shifted_zeros import create_random_individual
from random_set_generator import random_set_generator
from toolbox import (
    crossover_one_point,        # 기본 one-point 교차 연산
    mutation_bit_flip,          # 기본 bit flip 돌연변이 연산
    selection_rank_with_elite,  # 엘리트 선택 연산
)

# 교차 연산: 기본 one-point 교차를 수행하여 자식 두 개체 생성
def crossover(parent1, parent2):
    child1_genes, child2_genes = crossover_one_point(parent1.gene_list, parent2.gene_list)
    return Individual(child1_genes), Individual(child2_genes)

# 돌연변이 연산: 단순 bit flip 변환
def mutate(ind):
    mutated_gene = mutation_bit_flip(ind.gene_list)
    return Individual(mutated_gene)

# 선택 연산: 엘리트 선택 적용
def select(population):
    return selection_rank_with_elite(population, elite_size=2)

random.seed(63)
# 아이템 생성 및 설정
items = random_set_generator(1, 100, 0.1, 7, 200)
Individual.set_items(items)
Individual.set_max_weight(10)

# GA 파라미터 설정
POPULATION_SIZE = 80
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.2
MAX_GENERATIONS = 70
RUNS = 100

best = []
total_numbers = []

for _ in range(RUNS):
    # 초기 개체군 생성
    first_population = [create_random_individual(len(items), zeros=30) for _ in range(POPULATION_SIZE)]
    Individual.counter = 0
    best_individual = random.choice(first_population)
    generation_number = 0
    population = first_population.copy()

    # 세대 반복
    while generation_number < MAX_GENERATIONS:
        generation_number += 1
        # 선택 연산
        offspring = select(population)
        # 교차 연산: 짝을 이루어 교차 수행
        crossed_offspring = []
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_PROBABILITY:
                kid1, kid2 = crossover(ind1, ind2)
                crossed_offspring.extend([kid1, kid2])
            else:
                crossed_offspring.extend([ind1, ind2])
        # 돌연변이 연산
        mutated_offspring = []
        for mutant in crossed_offspring:
            if random.random() < MUTATION_PROBABILITY:
                new_mutant = mutate(mutant)
                mutated_offspring.append(new_mutant)
            else:
                mutated_offspring.append(mutant)
        population = mutated_offspring.copy()
        # 세대 내 최고 개체 업데이트
        best_of_generation = max(population, key=lambda ind: ind.fitness)
        if best_individual.fitness < best_of_generation.fitness:
            best_individual = best_of_generation

    best.append(best_individual.fitness)
    total_numbers.append(Individual.counter)

# 실행 결과 플롯: 최종 최고 적합도와 생성 개체의 평균 수
avg_fitness = sum(best) / len(best)
plt.plot(best)
plt.title(f'Average fitness: {avg_fitness} \n'
          f'Average number of individuals: {sum(total_numbers)/ len(total_numbers)}')
plt.axhline(y=avg_fitness, color='r', linestyle='-')
plt.show()
