import copy  # 깊은 복사를 위한 라이브러리
import random  # 난수 생성을 위한 라이브러리
from datetime import datetime  # 날짜 및 시간 관련 기능

from individual import Individual  # Individual 클래스 가져오기
from landscape import SquareType, Square, generate_random_landscape, plot_coverage  # 풍경 관련 클래스와 함수 가져오기
from toolbox import (  # 도구함수들 가져오기
    selection_rank_with_elite,
    crossover_operation,
    mutation_operation,
    plot_stats,
    stats,
    crossover_n_point,
    mutation_bit_flip_ones,
    mutation_shift_one,
)


def crossover(parent1, parent2):  # 두 부모 개체로부터 자식 개체들을 생성하는 교차 함수
    child1_genes, child2_genes = crossover_n_point(parent1.gene_list, parent2.gene_list, 3)  # 3점 교차 수행
    return Individual(child1_genes), Individual(child2_genes)  # 생성된 유전자로 두 자식 개체 반환


def mutate(ind):  # 개체에 돌연변이를 적용하는 함수
    if random.random() < .5:  # 50% 확률로 비트 플립 돌연변이 선택
        mut = mutation_bit_flip_ones(ind.gene_list)  # 1인 비트만 플립하는 돌연변이 적용
    else:  # 나머지 50% 확률로 위치 이동 돌연변이 선택
        mut = mutation_shift_one(ind)  # 1인 비트 위치를 이동시키는 돌연변이 적용
    return Individual(mut)  # 돌연변이가 적용된 새 개체 반환


def select(population):  # 개체 선택 함수
    return selection_rank_with_elite(population, elite_size = 2)  # 랭크 기반 선택 + 엘리트 2개 유지


random.seed(15)  # 난수 생성기 시드 설정 (재현성을 위해)

rows = 60  # 격자의 행 수
cols = 60  # 격자의 열 수
radar_radius = 7  # 레이더의 커버리지 반경

square_grid = {  # 격자에 배치할 다양한 타입의 지형 정의
    Square(SquareType.water, needs_coverage = False, tower_cost = 500): 20,  # 물: 레이더 불가, 비용 높음, 상대적 빈도 20
    Square(SquareType.land, needs_coverage = False, tower_cost = 30):   100,  # 땅: 레이더 가능, 비용 낮음, 상대적 빈도 100
    Square(SquareType.hill, needs_coverage = False, tower_cost = 100):  8,    # 언덕: 레이더 가능, 비용 중간, 상대적 빈도 8
    Square(SquareType.city, needs_coverage = True, tower_cost = 200):   1     # 도시: 레이더 커버리지 필요, 비용 높음, 상대적 빈도 1
}

landscape = generate_random_landscape(list(square_grid.keys()), square_grid, rows, cols)  # 무작위 지형 생성


def fitness_function(coords):  # 적합도 함수 정의
    global landscape, radar_radius  # 글로벌 변수 참조
    test_landscape = copy.deepcopy(landscape)  # 지형 깊은 복사 (원본 유지)
    test_landscape.add_radars(coords, radar_radius)  # 레이더 배치 적용
    return - test_landscape.uncovered_count() * 500 - test_landscape.radar_cost()  # 적합도 계산 (음수로 반환)


Individual.rows = rows  # Individual 클래스 변수 설정
Individual.cols = cols  # Individual 클래스 변수 설정
Individual.set_fitness_function(fitness_function)  # 적합도 함수 설정

POPULATION_SIZE = 60  # 인구 크기 설정
CROSSOVER_PROBABILITY = .5  # 교차 확률 설정
MUTATION_PROBABILITY = .5  # 돌연변이 확률 설정
MAX_GENERATIONS = 400  # 최대 세대 수 설정

first_population = [Individual.generate_random(.005) for _ in range(POPULATION_SIZE)]  # 초기 인구 생성 (레이더 배치 확률 0.005)
best_ind = random.choice(first_population)  # 임의로 최고 개체 초기화
fit_avg = []  # 평균 적합도 기록 리스트 초기화
fit_best = []  # 최고 적합도 기록 리스트 초기화
generation_num = 0  # 세대 카운터 초기화
population = first_population.copy()  # 현재 인구 설정

while generation_num < MAX_GENERATIONS:  # 최대 세대까지 반복
    generation_num += 1  # 세대 카운터 증가
    offspring = select(population)  # 선택 연산 수행
    crossed_offspring = crossover_operation(offspring, crossover, CROSSOVER_PROBABILITY)  # 병렬 교차 연산 수행
    mutated_offspring = mutation_operation(crossed_offspring, mutate, MUTATION_PROBABILITY)  # 병렬 돌연변이 연산 수행
    population = mutated_offspring.copy()  # 새로운 인구로 갱신
    best_ind, fit_avg, fit_best = stats(population, best_ind, fit_avg, fit_best)  # 통계 업데이트
    print(f'Generation {generation_num}. Avg fit: {fit_avg[-1]}. Best fit: {best_ind.fitness}')  # 세대 정보 출력

    test_landscape = copy.deepcopy(landscape)  # 지형 깊은 복사
    test_landscape.add_radars(best_ind.get_coordinates(), radar_radius)  # 최고 개체의 레이더 배치 적용
    plot_coverage(test_landscape, title = f"Best Individual for Generation: {generation_num}")  # 커버리지 시각화

plot_stats(fit_avg, fit_best, "Radar Placement Problem")  # 적합도 추이 그래프 출력

plot_coverage(landscape)  # 초기 지형 커버리지 시각화
landscape.add_radars(best_ind.get_coordinates(), radar_radius)  # 최종 최고 개체의 레이더 배치 적용
plot_coverage(landscape)  # 최종 커버리지 시각화

print(f'Radar count: {best_ind.count_radars()}')  # 사용된 레이더 수 출력
print(f'Best Fitness: {best_ind.fitness}')  # 최고 적합도 출력
