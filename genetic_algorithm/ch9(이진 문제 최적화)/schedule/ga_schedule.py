import random

# Individual 클래스는 스케줄(근무 배정) 해(solution)를 표현 (개별 파일에서 정의됨)
from individual import Individual
# 스케줄 분석 함수: 근무 편차와 휴식 조건 평가 함수들을 임포트
from schedule_analyzer import shift_deviations, shift_relax
# 유전연산 관련 도구: n-점 교차, 랭크 선택, 단순 비트 플립 돌연변이, 교차/돌연변이 연산, 통계 및 플롯 함수들을 임포트
from toolbox import (
    crossover_n_point, selection_rank_with_elite, mutation_bit_flip, crossover_operation,
    mutation_operation, stats, plot_stats,
)

# 선택 함수: 개체군으로부터 엘리트 보존을 포함한 랭크 기반 선택을 수행  
def select(population):
    return selection_rank_with_elite(population, elite_size = 2)

# 교차 함수: 부모 두 개체의 유전자 리스트에 대해 3-점 교차를 수행하고, 생성된 자식 개체 두 개 반환  
def crossover(parent1, parent2):
    # 3개의 교차점을 선택하여 두 부모의 유전자 일부를 교환
    child1_genes, child2_genes = crossover_n_point(parent1.gene_list, parent2.gene_list, 3)
    return Individual(child1_genes), Individual(child2_genes)

# 돌연변이 함수: 부모의 유전자 리스트 중 한 위치를 단순 비트 플립하여 돌연변이 개체 생성  
def mutate(ind):
    mutated_gene = mutation_bit_flip(ind.gene_list)
    return Individual(mutated_gene)
  
if __name__ == '__main__':
    # 난수 시드 설정 (실행 결과 재현 가능)
    random.seed(1)

    # 스케줄 문제에서 사용할 파라미터 설정: 직원 수와 근무 기간 (여기서는 5명의 직원, 7일간의 스케줄)
    Individual.set_employees(3)
    Individual.set_period(3)
    # 개체군의 크기 설정: 직원 수 * 근무 기간 * 3 (3은 근무조 수)
    print(f'Gene List Length: {Individual.employees * Individual.period * 3}')

    # fitness 함수 정의: 스케줄 데이터프레임(df)을 입력받아, 근무 편차(shift_deviations)와 
    # 휴식 조건(shift_relax)을 평가하여 음수 값으로 반환 (값이 작을수록 좋은 스케줄)
    def fitness_function(df):
        dev = shift_deviations(df,
                               mor_min = 1, mor_max = 4,
                               day_min = 2, day_max = 5,
                               evn_min = 1, evn_max = 2
                               )
        relax = shift_relax(df, 1, 1, 3)
        # 휴식 조건 위반에 5배의 패널티를 추가하여 최종 평가 (음수 값 최대화)
        return -(dev + relax * 5)

    # Individual 클래스에 fitness 함수 등록
    Individual.set_fitness_function(fitness_function)

    # 유전 알고리즘 관련 파라미터 설정
    POPULATION_SIZE = 10
    CROSSOVER_PROBABILITY = .8
    MUTATION_PROBABILITY = .5
    # MUTATION_PROBABILITY = 1 / (Individual.employees * Individual.period * 3) # 돌연변이 확률 1/L (L은 유전자 길이)
    # print(f'[debug] Mutation Probability: {MUTATION_PROBABILITY:.4f}') # 돌연변이 확률 출력(디버깅)
    MAX_GENERATIONS = 40

    # 초기 개체군 생성: 각 개체는 무작위 근무 스케줄(비트 문자열)로 생성됨
    first_population = [Individual.generate_random() for _ in range(POPULATION_SIZE)]
    print(f'\n[debug] Initial Population Size: {len(first_population)}') # 초기 개체군 크기 출력(디버깅)
    # print(f'[debug]Initial Population: {first_population}') # 초기 개체군 출력(디버깅)

    # 초기 최고 개체를 무작위로 선택
    best_ind = random.choice(first_population)
    fit_avg = []   # 각 세대별 평균 fitness 저장 리스트
    fit_best = []  # 각 세대별 최고 fitness 저장 리스트
    generation_num = 0
    population = first_population.copy()

    # 세대 반복: 최고 fitness가 0이 될 때까지 혹은 최대 세대 수 도달할 때까지 진행
    while generation_num < MAX_GENERATIONS and best_ind.fitness != 0:
        generation_num += 1
        # 선택 단계: 랭크 기반 선택 (엘리트 보존)
        offspring = select(population)

        # 교차 단계: 선택된 개체들에 대해 지정된 교차 확률로 교차 연산 수행
        crossed_offspring = crossover_operation(offspring, crossover, CROSSOVER_PROBABILITY)
        # 돌연변이 단계: 교차된 개체들에 대해 지정된 돌연변이 확률로 돌연변이 연산 수행
        mutated_offspring = mutation_operation(crossed_offspring, mutate, MUTATION_PROBABILITY)
        population = mutated_offspring.copy()
        # 현재 개체군의 통계(평균 fitness, 최고 fitness)를 업데이트
        best_ind, fit_avg, fit_best = stats(population, best_ind, fit_avg, fit_best)

    print(f'\n[debug] Generation {generation_num}')
    print(f'[debug] Best Individual: {best_ind}') # 최고 개체 출력(디버깅)
    print(f'[debug] gene list: {best_ind.gene_list}') # 최고 개체 출력(디버깅)
    print(f'[debug] Best Fitness: {best_ind.fitness}') # 최고 fitness 출력(디버깅)
    print(f'[debug] Best Schedule:\n{best_ind.create_schedule().T}') # 최고 스케줄 출력(디버깅)

    # 세대별 통계 플롯 출력 (평균 및 최고 fitness)
    plot_stats(fit_avg, fit_best, "Schedule Problem")

    # 최종 생성된 전체 개체 수 출력
    print(f'\n[debug] Total Number of Individuals: {Individual.counter}')
    print(f'[debug] Gene List Length: {len(best_ind.gene_list)}')
    print(f'[debug] 최고 유전자:  {best_ind}')
    # 최고 개체의 스케줄을 시각화 (plot_schedule() 내부에서 matplotlib를 사용)
    best_ind.plot_schedule()

# 각 세대마다 통계 정보를 상세하게 출력 (예: 최대, 최소, 평균, 표준편차)
fitness_values = [ind.fitness for ind in population]
gen_max = max(fitness_values)
gen_min = min(fitness_values)
gen_avg = sum(fitness_values) / len(fitness_values)
gen_std = (sum((x - gen_avg) ** 2 for x in fitness_values) / len(fitness_values)) ** 0.5
print(f'\n[debug] Generation fit_values {generation_num} | Max: {gen_max}, Min: {gen_min}, Avg: {gen_avg:.2f}, Std: {gen_std:.2f}')