import random  # 난수 생성을 위한 라이브러리 가져오기

from individual import Individual  # Individual 클래스 가져오기
from schedule_analyzer import shift_deviations, shift_relax  # 스케줄 분석 함수들 가져오기
from toolbox import (  # 도구함수들 가져오기
    crossover_n_point, selection_rank_with_elite, mutation_bit_flip, crossover_operation,
    mutation_operation, stats, plot_stats,
)


def crossover(parent1, parent2):  # 두 부모 개체로부터 자식 개체들을 생성하는 교차 함수
    child1_genes, child2_genes = crossover_n_point(parent1.gene_list, parent2.gene_list, 3)  # 3점 교차 수행
    return Individual(child1_genes), Individual(child2_genes)  # 생성된 유전자로 두 자식 개체 반환


def mutate(ind):  # 개체에 돌연변이를 적용하는 함수
    mutated_gene = mutation_bit_flip(ind.gene_list)  # 비트 플립 돌연변이 적용
    return Individual(mutated_gene)  # 돌연변이가 적용된 새 개체 반환


def select(population):  # 개체 선택 함수
    return selection_rank_with_elite(population, elite_size = 2)  # 랭크 기반 선택 + 엘리트 2개 유지


if __name__ == '__main__':  # 스크립트가 직접 실행될 때만 실행되는 코드 블록

    random.seed(3)  # 난수 생성기 시드 설정 (재현성을 위해)

    Individual.set_employees(3)  # 직원 수를 3명으로 설정
    Individual.set_period(3)  # 스케줄링 기간을 3일로 설정


    def fitness_function(df):  # 적합도 함수 정의
        dev = shift_deviations(df,  # 교대 편차 계산
                               mor_min = 1, mor_max = 4,  # 아침 교대 최소/최대 직원 수
                               day_min = 2, day_max = 5,  # 점심 교대 최소/최대 직원 수
                               evn_min = 1, evn_max = 2   # 저녁 교대 최소/최대 직원 수
                               )
        relax = shift_relax(df, 1, 1, 3)  # 휴식 위반 계산 (교대별 필요 휴식 시간)
        return -(dev + relax * 5)  # 편차와 위반에 대해 음수 점수 반환 (최대화 문제로 변환)


    Individual.set_fitness_function(fitness_function)  # 적합도 함수 설정

    POPULATION_SIZE = 30  # 인구 크기 설정
    CROSSOVER_PROBABILITY = .8  # 교차 확률 설정
    MUTATION_PROBABILITY = .5  # 돌연변이 확률 설정
    MAX_GENERATIONS = 200  # 최대 세대 수 설정

    first_population = [Individual.generate_random() for _ in range(POPULATION_SIZE)]  # 초기 인구 생성
    best_ind = random.choice(first_population)  # 임의로 최고 개체 초기화
    fit_avg = []  # 평균 적합도 기록 리스트 초기화
    fit_best = []  # 최고 적합도 기록 리스트 초기화
    generation_num = 0  # 세대 카운터 초기화
    population = first_population.copy()  # 현재 인구 설정

    while generation_num < MAX_GENERATIONS and best_ind.fitness != 0:  # 최대 세대 또는 최적해 도달까지 반복
        generation_num += 1  # 세대 카운터 증가
        offspring = select(population)  # 선택 연산 수행
        crossed_offspring = crossover_operation(offspring, crossover, CROSSOVER_PROBABILITY)  # 교차 연산 수행
        mutated_offspring = mutation_operation(crossed_offspring, mutate, MUTATION_PROBABILITY)  # 돌연변이 연산 수행
        population = mutated_offspring.copy()  # 새로운 인구로 갱신
        best_ind, fit_avg, fit_best = stats(population, best_ind, fit_avg, fit_best)  # 통계 업데이트

    plot_stats(fit_avg, fit_best, "Schedule Problem")  # 적합도 그래프 출력

    print(f'Total Number of Individuals: {Individual.counter}')  # 총 생성된 개체 수 출력
    print(f'Cache Hits: {Individual.cache_hit}')  # 캐시 히트 횟수 출력

    best_ind.plot_schedule()  # 최종 최적 스케줄 시각화
