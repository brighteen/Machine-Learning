# ga_intuitive_approach.py
import random

from individual import Individual
from toolbox import (
    crossover_one_point, mutation_bit_flip, selection_rank_with_elite,
    get_items_from_my_room, crossover_operation, mutation_operation,
    stats, plot_stats,
)

# 교차 함수: 한 점 교차 후 자식 개체 생성
def crossover(parent1, parent2):
    child1_genes, child2_genes = crossover_one_point(parent1.gene_list, parent2.gene_list)
    return Individual(child1_genes), Individual(child2_genes)

# 돌연변이 함수: 비트 플립 방식으로 돌연변이 수행
def mutate(ind):
    mutated_gene = mutation_bit_flip(ind.gene_list)
    return Individual(mutated_gene)

# '내 방'에서 가져온 아이템 목록을 사용
Individual.set_items(get_items_from_my_room())
Individual.set_max_weight(10)

random.seed(63)

# GA 파라미터 설정 (소규모 개체군과 적은 세대 수)
POPULATION_SIZE = 8
CROSSOVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.2
MAX_GENERATIONS = 20

# 초기 개체군 생성: Individual 클래스의 create_random() 메서드 사용
first_population = [Individual.create_random() for _ in range(POPULATION_SIZE)]
population = first_population.copy()
fitness_list = [ind.fitness for ind in population]
fit_avg = [sum(fitness_list) / len(population)]
fit_best = [max(fitness_list)]
fit_best_ever = [max(fitness_list + fit_best)]
best_ind = random.choice(first_population)

generation_number = 0

# 세대 루프: 선택, 교차, 돌연변이 및 통계 정보 업데이트
while generation_number < MAX_GENERATIONS:
    generation_number += 1
    offspring = selection_rank_with_elite(population, elite_size=2)
    crossed_offspring = crossover_operation(offspring, crossover, CROSSOVER_PROBABILITY)
    mutated_offspring = mutation_operation(crossed_offspring, mutate, MUTATION_PROBABILITY)
    population = mutated_offspring.copy()

    best_ind, fit_avg, fit_best, fit_best_ever = stats(population, best_ind, fit_avg, fit_best, fit_best_ever)

# 최종 통계 플롯 출력 및 최고 개체 정보 출력
plot_stats(fit_avg, fit_best_ever, "Knapsack Problem")
best_ind.plot_info()
