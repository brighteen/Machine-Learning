import copy
import random

from individual import Individual
from landscape import SquareType, Square, generate_random_landscape, plot_coverage
from toolbox import (
    selection_rank_with_elite,
    crossover_operation,
    mutation_operation,
    plot_stats,
    stats,
    crossover_n_point,
    mutation_bit_flip_ones,
    mutation_shift_one,
)

# 교차 함수: 3-점 교차를 통해 부모 유전자 리스트에서 두 자식을 생성
def crossover(parent1, parent2):
    child1_genes, child2_genes = crossover_n_point(parent1.gene_list, parent2.gene_list, 3)
    return Individual(child1_genes), Individual(child2_genes)

# 돌연변이 함수: 50% 확률로 두 가지 돌연변이 연산 중 하나를 적용
def mutate(ind):
    if random.random() < .5:
        mut = mutation_bit_flip_ones(ind.gene_list)
    else:
        mut = mutation_shift_one(ind)
    return Individual(mut)

# 선택 함수: 엘리트 보존을 포함한 랭크 기반 선택
def select(population):
    return selection_rank_with_elite(population, elite_size = 2)

random.seed(15)

rows = 60
cols = 60
radar_radius = 7

# 지형 구성: 각 Square 타입별 빈도(가중치) 설정
square_grid = {
    Square(SquareType.water, needs_coverage = False, tower_cost = 500): 20,
    Square(SquareType.land, needs_coverage = False, tower_cost = 30):   100,
    Square(SquareType.hill, needs_coverage = False, tower_cost = 100):  8,
    Square(SquareType.city, needs_coverage = True, tower_cost = 200):   1
}

# 랜덤 지형 생성
landscape = generate_random_landscape(list(square_grid.keys()), square_grid, rows, cols)

# 동일한 fitness 함수 정의: 지형에 레이더를 배치하고 미커버 영역 및 레이더 비용을 평가
def fitness_function(coords):
    global landscape, radar_radius
    test_landscape = copy.deepcopy(landscape)
    test_landscape.add_radars(coords, radar_radius)
    return - test_landscape.uncovered_count() * 500 - test_landscape.radar_cost()

Individual.rows = rows
Individual.cols = cols
Individual.set_fitness_function(fitness_function)

POPULATION_SIZE = 60
CROSSOVER_PROBABILITY = .5
MUTATION_PROBABILITY = .5
MAX_GENERATIONS = 400

first_population = [Individual.generate_random(.005) for _ in range(POPULATION_SIZE)]
best_ind = random.choice(first_population)
fit_avg = []
fit_best = []
generation_num = 0
population = first_population.copy()
generation_number = 0

# 세대 반복: 선택, 교차, 돌연변이, 통계 업데이트, 그리고 현재 최고 개체의 커버리지 플롯
while generation_num < MAX_GENERATIONS:
    generation_num += 1
    offspring = selection_rank_with_elite(population, elite_size = 2)
    crossed_offspring = crossover_operation(offspring, crossover, CROSSOVER_PROBABILITY)
    mutated_offspring = mutation_operation(crossed_offspring, mutate, MUTATION_PROBABILITY)
    population = mutated_offspring.copy()
    best_ind, fit_avg, fit_best = stats(population, best_ind, fit_avg, fit_best)
    print(f'Generation {generation_num}. Avg fit: {fit_avg[-1]}. Best fit: {best_ind.fitness}')
    
    test_landscape = copy.deepcopy(landscape)
    test_landscape.add_radars(best_ind.get_coordinates(), radar_radius)
    plot_coverage(test_landscape, title = f"Best Individual for Generation: {generation_num}")

plot_stats(fit_avg, fit_best, "Radar Placement Problem")

plot_coverage(landscape)
landscape.add_radars(best_ind.get_coordinates(), radar_radius)
plot_coverage(landscape)

print(f'Radar count: {best_ind.count_radars()}')
print(f'Best Fitness: {best_ind.fitness}')
