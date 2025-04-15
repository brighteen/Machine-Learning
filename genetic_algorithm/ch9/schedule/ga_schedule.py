import random
from individual import Individual
from schedule_analyzer import shift_deviations, shift_relax
from toolbox import (
    crossover_n_point,             # n-점 교차 함수 (스케줄에 맞게 적용)
    selection_rank_with_elite,      # 엘리트 선택 함수
    mutation_bit_flip,              # 기본 bit flip 돌연변이 함수
    crossover_operation,            # 전체 개체군에 교차 연산 적용 함수
    mutation_operation,             # 전체 개체군에 돌연변이 연산 적용 함수
    stats,                          # 세대별 통계 업데이트 함수
    plot_stats,                     # 세대별 결과 플롯 함수
)

# 교차 연산: 스케줄 문제에서 n-점 교차(3점)를 통해 교차 수행
def crossover(parent1, parent2):
    child1_genes, child2_genes = crossover_n_point(parent1.gene_list, parent2.gene_list, 3)
    return Individual(child1_genes), Individual(child2_genes)

# 돌연변이 연산: bit flip 방식 적용
def mutate(ind):
    mutated_gene = mutation_bit_flip(ind.gene_list)
    return Individual(mutated_gene)

# 선택 연산 함수: 엘리트 선택 적용
def select(population):
    return selection_rank_with_elite(population, elite_size=2)

if __name__ == '__main__':
    random.seed(3)
    # 직원 수와 근무 기간 설정
    Individual.set_employees(5)
    Individual.set_period(7)
    
    # 스케줄의 적합도 함수를 정의
    def fitness_function(df):
        # shift_deviations: 근무조별 인원 편차 계산 (아침, 낮, 저녁의 최소·최대 조건)
        dev = shift_deviations(df,
                               mor_min=1, mor_max=4,
                               day_min=2, day_max=5,
                               evn_min=1, evn_max=2)
        # shift_relax: 근무 연속 후 휴식 미비에 대한 penalty 계산
        relax = shift_relax(df, 1, 1, 3)
        # 총 벌점(penalty)을 음수로 반환 (벌점이 적을수록 적합도가 높음)
        return -(dev + relax * 5)
    
    Individual.set_fitness_function(fitness_function)

    # GA 파라미터
    POPULATION_SIZE = 30
    CROSSOVER_PROBABILITY = 0.8
    MUTATION_PROBABILITY = 0.5
    MAX_GENERATIONS = 200

    # 초기 개체군 생성: 각 개체는 스케줄을 이진 벡터로 표현
    first_population = [Individual.generate_random() for _ in range(POPULATION_SIZE)]
    best_ind = random.choice(first_population)
    fit_avg = []
    fit_best = []
    generation_num = 0
    population = first_population.copy()

    # GA 메인 루프: 최대 세대 또는 적합도가 0이 될 때까지 반복
    while generation_num < MAX_GENERATIONS and best_ind.fitness != 0:
        generation_num += 1
        offspring = select(population)
        crossed_offspring = crossover_operation(offspring, crossover, CROSSOVER_PROBABILITY)
        mutated_offspring = mutation_operation(crossed_offspring, mutate, MUTATION_PROBABILITY)
        population = mutated_offspring.copy()
        best_ind, fit_avg, fit_best = stats(population, best_ind, fit_avg, fit_best)
    plot_stats(fit_avg, fit_best, "Schedule Problem")
    print(f'Total Number of Individuals: {Individual.counter}')
    # 최적 스케줄 시각화
    best_ind.plot_schedule()
