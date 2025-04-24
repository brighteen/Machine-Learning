# ga_general_fitness_driven_montecarlo.py
import random
import matplotlib.pyplot as plt

from individual import Individual
from random_individual_shifted_zeros import create_random_individual
from random_set_generator import random_set_generator
from toolbox import (
    selection_rank_with_elite, crossover_fitness_driven_one_point,
    mutation_fitness_driven_bit_flip,
)

# 교차(crossover) 함수: 부모 개체 두 개를 받아 한 점 교차 후 fitness 기반으로 자식을 선택
def crossover(parent1, parent2):
    return crossover_fitness_driven_one_point(parent1, parent2)

# 돌연변이(mutation) 함수: fitness 개선을 위해 비트 플립을 최대 3번 시도
def mutate(ind):
    return mutation_fitness_driven_bit_flip(ind, max_tries=3)

# 선택(selection) 함수: 엘리트 개체 2개를 보존하며 랭크 기반 선택 수행
def select(population):
    return selection_rank_with_elite(population, elite_size=2)

# 재현 가능한 결과를 위해 시드 설정
random.seed(68)

# 배낭 문제 해결에 사용할 아이템 집합 생성
items = random_set_generator(1, 100, 0.1, 7, 200)
Individual.set_items(items)
Individual.set_max_weight(10)  # 최대 허용 무게 설정

# GA 파라미터 설정
POPULATION_SIZE = 80
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.2
MAX_GENERATIONS = 50
RUNS = 100

# 결과를 기록할 리스트 초기화
best = []            # 각 실행별 최고 fitness 값
total_numbers = []   # 개체 생성 횟수 총합 기록

# RUNS 번 알고리즘을 실행하여 통계 정보를 수집
for _ in range(RUNS):
    # 초기 개체군 생성: 각 개체는 'zeros' 값이 반영된 랜덤 유전자 리스트로 생성
    first_population = [create_random_individual(len(items), zeros=30) for _ in range(POPULATION_SIZE)]
    Individual.counter = 0  # 개체 생성 카운터 초기화
    best_individual = random.choice(first_population)  # 초기 최고 개체를 랜덤 선택
    generation_number = 0

    population = first_population.copy()

    # 최대 세대 수 만큼 반복
    while generation_number < MAX_GENERATIONS:
        generation_number += 1

        # [선택 단계] 현재 개체군에서 다음 세대 후보 선택
        offspring = select(population)

        # [교차 단계] 선택된 후보들을 짝지어 교차 연산 적용
        crossed_offspring = []
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_PROBABILITY:
                # 교차가 일어나면 두 자식 생성 후 추가
                kid1, kid2 = crossover(ind1, ind2)
                crossed_offspring.append(kid1)
                crossed_offspring.append(kid2)
            else:
                # 교차가 일어나지 않으면 부모 그대로 유지
                crossed_offspring.append(ind1)
                crossed_offspring.append(ind2)

        # [돌연변이 단계] 교차 결과에 대해 돌연변이 연산 수행
        mutated_offspring = []
        for mutant in crossed_offspring:
            if random.random() < MUTATION_PROBABILITY:
                new_mutant = mutate(mutant)
                mutated_offspring.append(new_mutant)
            else:
                mutated_offspring.append(mutant)

        # 새로운 세대 개체군 업데이트
        population = mutated_offspring.copy()

        # 이번 세대에서 가장 좋은 개체 선택
        best_of_generation = max(population, key=lambda ind: ind.fitness)
        if best_individual.fitness < best_of_generation.fitness:
            best_individual = best_of_generation

    # 각 실행 후 최고 fitness 및 생성된 전체 개체 수 저장
    best.append(best_individual.fitness)
    total_numbers.append(Individual.counter)

# 전체 RUNS 동안의 평균 fitness 계산 후 시각화
avg_fitness = sum(best) / len(best)
plt.plot(best)
plt.title(f'Average fitness: {avg_fitness} \n'
          f'Average number of individuals: {sum(total_numbers) / len(total_numbers)}')
plt.axhline(y=avg_fitness, color='r', linestyle='-')
plt.show()
