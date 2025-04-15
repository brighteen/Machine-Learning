import copy
import random

# Individual 클래스: 레이더 배치 해(solution)를 표현 (다른 파일에서 정의됨)
from individual import Individual
# SquareType, Square, 랜덤 지형 생성, 지형 커버리지를 플롯하는 함수들을 가져옴
from landscape import SquareType, Square, generate_random_landscape, plot_coverage
# 선택, 교차, 돌연변이, 통계 및 플로팅 관련 유틸리티 함수들을 가져옴
from toolbox import (
    selection_rank_with_elite,
    crossover_operation,
    mutation_operation,
    plot_stats,
    stats,
    crossover_n_point,
    mutation_bit_flip,
)

# 교차 함수: 부모의 유전자 리스트에 대해 n-점 교차(여기서는 3점)를 수행하고 자식 개체 반환
def crossover(parent1, parent2):
    child1_genes, child2_genes = crossover_n_point(parent1.gene_list, parent2.gene_list, 3)
    return Individual(child1_genes), Individual(child2_genes)

# 돌연변이 함수: 부모의 유전자 리스트 중 무작위 위치 하나를 비트 플립하여 자식 개체 생성
def mutate(ind):
    mut = mutation_bit_flip(ind.gene_list)
    return Individual(mut)

# 선택 함수: 랭크 기반 선택 및 엘리트 보존 (엘리트 크기 2)
def select(population):
    return selection_rank_with_elite(population, elite_size = 2)

# 재현 가능한 결과를 위해 시드 설정
random.seed(15)

# 지형(grid)의 크기와 레이더 탐지 반경 설정
rows = 60
cols = 60
radar_radius = 7

# 지형의 각 영역별 타입(Square)과 가중치(해당 타입의 빈도)를 설정
square_grid = {
    Square(SquareType.water, needs_coverage = False, tower_cost = 500): 20,
    Square(SquareType.land, needs_coverage = False, tower_cost = 30):   100,
    Square(SquareType.hill, needs_coverage = False, tower_cost = 100):  8,
    Square(SquareType.city, needs_coverage = True, tower_cost = 200):   1
}

# 주어진 영역과 가중치를 바탕으로 랜덤 지형 생성
landscape = generate_random_landscape(list(square_grid.keys()), square_grid, rows, cols)

# fitness 함수 정의  
# 입력 좌표(레이다 배치 결과)를 바탕으로, 지형에 레이더를 배치한 후 미커버 영역의 수와 레이더 건설 비용을 평가  
def fitness_function(coords):
    global landscape, radar_radius
    test_landscape = copy.deepcopy(landscape)
    # 지정된 좌표에 레이더를 추가(탐지 반경 적용)
    test_landscape.add_radars(coords, radar_radius)
    # 미커버 영역 수와 건설 비용에 음수를 곱해 fitness 값을 생성(최소화 문제 → 음수 값 최대화)
    return - test_landscape.uncovered_count() * 500 - test_landscape.radar_cost()

# Individual 클래스의 클래스 변수로 지형 크기와 fitness 함수 설정
Individual.rows = rows
Individual.cols = cols
Individual.set_fitness_function(fitness_function)

# GA 파라미터 설정
POPULATION_SIZE = 60
CROSSOVER_PROBABILITY = .5
MUTATION_PROBABILITY = .5
MAX_GENERATIONS = 400

# 초기 개체군 생성: 각 개체는 .005의 확률로 해당 셀에 레이더 배치 여부 결정
first_population = [Individual.generate_random(.005) for _ in range(POPULATION_SIZE)]
best_ind = random.choice(first_population)
fit_avg = []   # 세대별 평균 fitness 기록 리스트
fit_best = []  # 세대별 최고 fitness 기록 리스트
generation_num = 0
population = first_population.copy()
generation_number = 0  # (동일 의미 변수로 보임)

# 세대 반복: 선택, 교차, 돌연변이, 통계 업데이트, 그리고 현재 최고 개체의 레이더 커버리지 플롯
while generation_num < MAX_GENERATIONS:
    generation_num += 1
    offspring = selection_rank_with_elite(population, elite_size = 2)
    crossed_offspring = crossover_operation(offspring, crossover, CROSSOVER_PROBABILITY)
    mutated_offspring = mutation_operation(crossed_offspring, mutate, MUTATION_PROBABILITY)
    population = mutated_offspring.copy()
    best_ind, fit_avg, fit_best = stats(population, best_ind, fit_avg, fit_best)
    print(f'Generation {generation_num}. Avg fit: {fit_avg[-1]}. Best fit: {best_ind.fitness}')
    
    # 현재 최고 개체의 좌표 정보를 가져와서 지형에 레이더 추가 후 커버리지 플롯
    test_landscape = copy.deepcopy(landscape)
    test_landscape.add_radars(best_ind.get_coordinates(), radar_radius)
    plot_coverage(test_landscape, title = f"Best Individual for Generation: {generation_num}")

# 전체 세대의 통계 플롯 출력
plot_stats(fit_avg, fit_best, "Radar Placement Problem")

# 최종 결과: 원래 지형(도시 등) 플롯 후 최고 해의 레이더 배치를 반영하여 커버리지 플롯 출력
plot_coverage(landscape, title = "Cities")
landscape.add_radars(best_ind.get_coordinates(), radar_radius)
plot_coverage(landscape, title = "Best Radar Placement")

# 최종 레이더 개수 및 최고 fitness 출력
print(f'Radar Count: {best_ind.count_radars()}')
print(f'Best Fitness: {best_ind.fitness}')
