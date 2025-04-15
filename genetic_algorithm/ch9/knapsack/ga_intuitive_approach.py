import random
from individual import Individual
from toolbox import (
    crossover_one_point,              # 기본 one-point 교차 연산
    mutation_bit_flip,                # 기본 bit flip 돌연변이 연산
    selection_rank_with_elite,        # 엘리트 선택 연산
    get_items_from_my_room,           # 실제 환경 아이템 집합 함수
    crossover_operation,              # 전체 개체군에 교차 연산 적용
    mutation_operation,               # 전체 개체군에 돌연변이 연산 적용
    stats,                            # 세대별 통계 계산 함수
    plot_stats,                       # 세대별 결과 플롯 함수
)

# 교차 연산 함수
def crossover(parent1, parent2):
    child1_genes, child2_genes = crossover_one_point(parent1.gene_list, parent2.gene_list)
    return Individual(child1_genes), Individual(child2_genes)

# 돌연변이 연산 함수
def mutate(ind):
    mutated_gene = mutation_bit_flip(ind.gene_list)
    return Individual(mutated_gene)

# 실제 사용 아이템 집합 설정
Individual.set_items(get_items_from_my_room())
Individual.set_max_weight(10)

random.seed(63)
# GA 파라미터
POPULATION_SIZE = 8
CROSSOVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.2
MAX_GENERATIONS = 20

# 초기 개체군 생성 (직관적인 접근: 소규모 인원 사용)
first_population = [Individual.create_random() for _ in range(POPULATION_SIZE)]
population = first_population.copy()
# 초기 통계 계산
fitness_list = [ind.fitness for ind in population]
fit_avg = [sum(fitness_list) / len(population)]
fit_best = [max(fitness_list)]
fit_best_ever = [max(fitness_list + fit_best)]
best_ind = random.choice(first_population)
population = first_population.copy()
generation_number = 0

# GA 메인 루프
while generation_number < MAX_GENERATIONS:
    generation_number += 1
    # 선택 연산
    offspring = selection_rank_with_elite(population, elite_size=2)
    # 교차 연산 적용
    crossed_offspring = crossover_operation(offspring, crossover, CROSSOVER_PROBABILITY)
    # 돌연변이 연산 적용
    mutated_offspring = mutation_operation(crossed_offspring, mutate, MUTATION_PROBABILITY)
    population = mutated_offspring.copy()
    # 통계 업데이트
    best_ind, fit_avg, fit_best, fit_best_ever = stats(population, best_ind, fit_avg, fit_best, fit_best_ever)

# 결과 플롯
plot_stats(fit_avg, fit_best_ever, "Knapsack Problem")
best_ind.plot_info()
