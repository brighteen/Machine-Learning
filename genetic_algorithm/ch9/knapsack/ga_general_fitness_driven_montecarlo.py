import random
import matplotlib.pyplot as plt

# Individual 클래스: 배낭 문제에서 해(개체)를 표현
from individual import Individual
# 개체를 생성할 때, 지정된 수의 0을 포함시켜 초기화하는 함수
from random_individual_shifted_zeros import create_random_individual
# 아이템 집합 생성 함수 (가격, 무게 등 랜덤 생성)
from random_set_generator import random_set_generator
# GA의 선택, 교차, 돌연변이 등 기본 연산이 담긴 toolbox 모듈
from toolbox import (
    selection_rank_with_elite,           # 엘리트 기반 선택 연산
    crossover_fitness_driven_one_point,    # fitness-driven one-point 교차 연산
    mutation_fitness_driven_bit_flip,      # fitness-driven bit flip 돌연변이
)

# 교차 연산 함수: 두 부모를 입력받아 교차를 수행하여 두 자식을 반환
def crossover(parent1, parent2):
    return crossover_fitness_driven_one_point(parent1, parent2)

# 돌연변이 연산 함수: 한 개체의 유전자를 bit flip 방식으로 돌연변이 시킴
def mutate(ind):
    return mutation_fitness_driven_bit_flip(ind, max_tries=3)

# 선택 연산 함수: 엘리트 선택을 적용하여 새 개체군 구성
def select(population):
    return selection_rank_with_elite(population, elite_size=2)

# 랜덤 시드 설정 (실행 재현성을 위해)
random.seed(68)

# 아이템 집합 생성: 최소 가격 1, 최대 가격 100, 최소 무게 0.1, 아이템 개수 7, 전체 아이템 수 200
items = random_set_generator(1, 100, 0.1, 7, 200)
Individual.set_items(items)
# 배낭 최대 무게 설정 (제약 조건)
Individual.set_max_weight(10)

# GA 파라미터 설정
POPULATION_SIZE = 80
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.2
MAX_GENERATIONS = 50
RUNS = 100  # Monte Carlo run 횟수

# 결과 저장 리스트
best = []
total_numbers = []

# 여러 번 실행하여 통계적 결과를 얻는다.
for _ in range(RUNS):
    # 초기 개체군 생성: 각 개체는 items 개수에 맞춘 길이의 이진 벡터, 30개의 0을 포함하도록 생성
    first_population = [create_random_individual(len(items), zeros=30) for _ in range(POPULATION_SIZE)]
    # 개체 생성 카운터 초기화
    Individual.counter = 0
    # 임의의 개체를 최고 개체로 초기화
    best_individual = random.choice(first_population)
    generation_number = 0
    population = first_population.copy()

    # 최대 세대(MAX_GENERATIONS) 반복
    while generation_number < MAX_GENERATIONS:
        generation_number += 1

        # 선택 연산: 적합도 기반 엘리트 선택 적용
        offspring = select(population)

        # 교차 연산: 짝을 이루어 CROSSOVER_PROBABILITY 확률로 교차 수행
        crossed_offspring = []
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_PROBABILITY:
                kid1, kid2 = crossover(ind1, ind2)
                crossed_offspring.extend([kid1, kid2])
            else:
                crossed_offspring.extend([ind1, ind2])

        # 돌연변이 연산: 각 개체에 대해 MUTATION_PROBABILITY 확률로 돌연변이 적용
        mutated_offspring = []
        for mutant in crossed_offspring:
            if random.random() < MUTATION_PROBABILITY:
                new_mutant = mutate(mutant)
                mutated_offspring.append(new_mutant)
            else:
                mutated_offspring.append(mutant)

        # 새 개체군 업데이트
        population = mutated_offspring.copy()

        # 세대 내 최고 적합도를 가진 개체 업데이트
        best_of_generation = max(population, key=lambda ind: ind.fitness)
        if best_individual.fitness < best_of_generation.fitness:
            best_individual = best_of_generation

    # 각 run의 결과 기록
    best.append(best_individual.fitness)
    total_numbers.append(Individual.counter)

# 전체 run의 평균 최고 적합도 계산 및 플롯
avg_fitness = sum(best) / len(best)
plt.plot(best)
plt.title(f'Average fitness: {avg_fitness} \n'
          f'Average number of individuals: {sum(total_numbers)/ len(total_numbers)}')
plt.axhline(y=avg_fitness, color='r', linestyle='-')
plt.show()
