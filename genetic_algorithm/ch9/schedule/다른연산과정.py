from toolbox import (
    selection_rank_with_elite,

    crossover_n_point,
    crossover_fitness_driven_one_point,

    mutation_bit_flip,
    mutation_shuffle,
    mutation_fitness_driven_bit_flip,

    crossover_operation,
    mutation_operation,

    stats, plot_stats,
)
from schedule_analyzer import shift_deviations, shift_relax
from individual import Individual
import random

def select(population):
    return selection_rank_with_elite(population, elite_size = 2)

if __name__ == '__main__':
    # 난수 시드 설정 (실행 결과 재현 가능)
    random.seed(1)

    # 스케줄 문제에서 사용할 파라미터 설정: 직원 수와 근무 기간 (여기서는 5명의 직원, 7일간의 스케줄)
    Individual.set_employees(3)
    Individual.set_period(3)

    # 유전 알고리즘 관련 파라미터 설정
    POPULATION_SIZE = 10
    CROSSOVER_PROBABILITY = .8
    MUTATION_PROBABILITY = .5
    MAX_GENERATIONS = 40

    # 교차 및 돌연변이 조합 설정
    crossover_methods = [
        ("n_point", crossover_n_point),
        ("one_point", crossover_fitness_driven_one_point)
    ]

    mutation_methods = [
        ("bit_flip", mutation_bit_flip),
        ("shuffle", mutation_shuffle),
        ("fitness_driven_bit_flip", mutation_fitness_driven_bit_flip)
    ]

    results = []

    for crossover_name, crossover_func in crossover_methods:
        for mutation_name, mutation_func in mutation_methods:
            random.seed(1)
            Individual.set_employees(3)
            Individual.set_period(3)

            def crossover(parent1, parent2):
                if crossover_name == "n_point":
                    child1_genes, child2_genes = crossover_func(parent1.gene_list, parent2.gene_list, 3)
                    return Individual(child1_genes), Individual(child2_genes)
                else:
                    return crossover_func(parent1, parent2)

            def mutate(ind):
                if mutation_name == "fitness_driven_bit_flip":
                    if not isinstance(ind, Individual):
                        ind = Individual(ind)  # 리스트를 Individual 객체로 변환
                    mutated_gene = mutation_func(ind.gene_list)
                else:
                    mutated_gene = mutation_func(ind.gene_list)
                return Individual(mutated_gene)

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

            first_population = [Individual.generate_random() for _ in range(POPULATION_SIZE)]
            best_ind = random.choice(first_population)
            fit_avg = []
            fit_best = []
            generation_num = 0
            population = first_population.copy()

            while generation_num < MAX_GENERATIONS and best_ind.fitness != 0:
                generation_num += 1
                offspring = select(population)
                crossed_offspring = crossover_operation(offspring, crossover, CROSSOVER_PROBABILITY)
                mutated_offspring = mutation_operation(crossed_offspring, mutate, MUTATION_PROBABILITY)
                population = mutated_offspring.copy()
                best_ind, fit_avg, fit_best = stats(population, best_ind, fit_avg, fit_best)

            results.append((crossover_name, mutation_name, best_ind.fitness, fit_avg, fit_best))

    for result in results:
        crossover_name, mutation_name, best_fitness, fit_avg, fit_best = result
        print(f"Crossover: {crossover_name}, Mutation: {mutation_name}, Best Fitness: {best_fitness}")
        print(f"Average Fitness per Generation: {fit_avg}")
        print(f"Best Fitness per Generation: {fit_best}")