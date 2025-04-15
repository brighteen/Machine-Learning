import copy
import random
from individual import Individual
# 지형 정보를 생성하고 시각화하는 모듈
from landscape import SquareType, Square, generate_random_landscape, plot_coverage
from toolbox import (
    selection_rank_with_elite,    # 선택 연산 함수
    crossover_operation,          # 교차 연산 적용 함수
    mutation_operation,           # 돌연변이 연산 적용 함수
    plot_stats,                   # 결과 통계 플롯 함수
    stats,                        # 통계 업데이트 함수
    crossover_n_point,            # n-점 교차 연산 함수
    mutation_bit_flip,            # bit flip 돌연변이 함수
)

# 교차 연산: 3-점 교차를 통해 두 부모로부터 자식 생성
def crossover(parent1, parent2):
    child1_genes, child2_genes = crossover_n_point(parent1.gene_list, parent2.gene_list, 3)
    return Individual(child1_genes), Individual(child2_genes)

# 돌연변이 연산: 기본 bit flip 적용
def mutate(ind):
    mut = mutation_bit_flip(ind.gene_list)
    return Individual(mut)

# 선택 함수: 엘리트 선택 적용 (elite_size=2)
def select(population):
    return selection_rank_with_elite(population, elite_size=2)

random.seed(15)

# 지형(landscape) 구성 파라미터 설정
rows = 60
cols = 60
radar_radius = 7

# 지도에 사용될 각 Square 유형과 가중치 설정
square_grid = {
    Square(SquareType.water, needs_coverage=False, tower_cost=500): 20,
    Square(SquareType.land, needs_coverage=False, tower_cost=30):   100,
    Square(SquareType.hill, needs_coverage=False, tower_cost=100):   8,
    Square(SquareType.city, needs_coverage=True, tower_cost=200):    1
}

# 지형 랜드스케이프 생성: 지정된 Square 유형과 가중치를 바탕으로 rows x cols 크기의 지도 생성
landscape = generate_random_landscape(list(square_grid.keys()), square_grid, rows, cols)

# 적합도 함수 정의: 주어진 좌표(레이다 배치)가 얼마나 효과적으로 영역을 커버하는지 평가 
def fitness_function(coords):
    global landscape, radar_radius
    # 원본 landscape 복사
    test_landscape = copy.deepcopy(landscape)
    # 레이다 배치 및 해당 반경 내 모든 칸 커버 처리
    test_landscape.add_radars(coords, radar_radius)
    # 커버되지 않은 칸에 대해 페널티 및 레이다 건설 비용 합산 (최소화 문제)
    return - test_landscape.uncovered_count() * 500 - test_landscape.radar_cost()

# Individual 클래스에 지도 크기 및 적합도 함수 설정
Individual.rows = rows
Individual.cols = cols
Individual.set_fitness_function(fitness_function)

# GA 파라미터 설정
POPULATION_SIZE = 60
CROSSOVER_PROBABILITY = 0.5
MUTATION_PROBABILITY = 0.5
MAX_GENERATIONS = 400

# 초기 개체군 생성: 각 개체는 특정 확률(.005)로 1을 가지도록 생성
first_population = [Individual.generate_random(0.005) for _ in range(POPULATION_SIZE)]
best_ind = random.choice(first_population)
fit_avg = []
fit_best = []
generation_num = 0
population = first_population.copy()
generation_number = 0

# GA 메인 루프
while generation_num < MAX_GENERATIONS:
    generation_num += 1
    # 선택 연산
    offspring = selection_rank_with_elite(population, elite_size=2)
    # 교차 연산 적용
    crossed_offspring = crossover_operation(offspring, crossover, CROSSOVER_PROBABILITY)
    # 돌연변이 연산 적용
    mutated_offspring = mutation_operation(crossed_offspring, mutate, MUTATION_PROBABILITY)
    population = mutated_offspring.copy()
    # 통계 업데이트: 최적 개체 및 세대별 평균 적합도 업데이트
    best_ind, fit_avg, fit_best = stats(population, best_ind, fit_avg, fit_best)
    print(f'Generation {generation_num}. Avg fit: {fit_avg[-1]}. Best fit: {best_ind.fitness}')
    # 현재 최적 개체를 지도에 적용한 후 커버리지를 시각화
    test_landscape = copy.deepcopy(landscape)
    test_landscape.add_radars(best_ind.get_coordinates(), radar_radius)
    plot_coverage(test_landscape, title=f"Best Individual for Generation: {generation_num}")

# 최종 결과 플롯
plot_stats(fit_avg, fit_best, "Radar Placement Problem")
plot_coverage(landscape, title="Cities")
landscape.add_radars(best_ind.get_coordinates(), radar_radius)
plot_coverage(landscape, title="Best Radar Placement")
print(f'Radar Count: {best_ind.count_radars()}')
print(f'Best Fitness: {best_ind.fitness}')
