import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt

# 입력된 값 g가 지정된 최소(min)와 최대(max) 범위를 벗어나지 않도록 보정
def _utils_constraints(g, min, max):
    if max and g > max:
        g = max
    if min and g < min:
        g = min
    return g

# 두 부모의 유전자(g1, g2)를 블렌딩 방식으로 섞어 자식의 유전자를 만듦.  
# alpha 값에 따라 혼합 정도가 결정되며, 결과값은 경계 제약 함수(_utils_constraints)를 거침.
def crossover_blend(g1, g2, alpha, min = None, max = None):
    shift = (1. + 2. * alpha) * random.random() - alpha
    new_g1 = (1. - shift) * g1 + shift * g2
    new_g2 = shift * g1 + (1. - shift) * g2
    return _utils_constraints(new_g1, min, max), _utils_constraints(new_g2, min, max)

# 주어진 유전자 g에 평균(mu)과 표준편차(sigma)를 갖는 가우시안 분포의 무작위 값을 더해 돌연변이를 발생시키고, 결과값은 경계 내로 조정
def mutate_gaussian(g, mu, sigma, min = None, max = None):
    mutated_gene = g + random.gauss(mu, sigma)
    return _utils_constraints(mutated_gene, min, max)

# 인구(population) 내에서 무작위로 몇 개체를 선택하여(토너먼트 방식) 그 중 가장 높은 적합도를 가진 개체를 반환
# 이를 전체 인구에 대해 반복하여 새로운 선택된 개체군을 만듦
def select_tournament(population, tournament_size):
    new_offspring = []
    for _ in range(len(population)):
        candidates = [random.choice(population) for _ in
        range(tournament_size)]
        new_offspring.append(max(candidates, key = lambda ind:
        ind.fitness))
    return new_offspring

# 유전자 알고리즘에서 적합도(fitness)를 평가하는 함수로, 여기서는 sin(x) – 0.2×|x| 값을 계산
def func(x):
    return np.sin(x) - .2 * abs(x)

# 주어진 인구에서 가장 높은 적합도를 가진 개체를 찾아 반환
def get_best(population):
    best = population[0]
    for ind in population:
        if ind.fitness > best.fitness:
            best = ind
    return best

# 현재 인구의 유전자 값과 적합도를 시각화하여, 함수 그래프와 개체들의 분포, 그리고 최적 개체를 보여줌
def plot_population(population, number_of_population):
    best = get_best(population)
    x = np.linspace(-10, 10)
    plt.plot(x, func(x), '--', color = 'blue')
    plt.plot([ind.get_gene() for ind in population], [ind.fitness for ind in population], 'o', color = 'orange')
    plt.plot([best.get_gene()], [best.fitness], 's', color = 'green')
    plt.title(f"Generation number {number_of_population}")
    plt.show()
    plt.close()

class Individual: # 개체를 나타내는 클래스
    # 유전자 리스트를 받아 적합도 함수를 통해 fitness를 계산
    def __init__(self, gene_list: List[float]) -> None:
        self.gene_list = gene_list
        self.fitness = func(self.gene_list[0])
    # 개체의 유전자를 반환
    def get_gene(self):
        return self.gene_list[0]
	# 두 개체의 유전자를 crossover_blend 함수를 통해 교차시켜 두 자식을 생성
    @classmethod
    def crossover(cls, parent1, parent2): # 교차차
        child1_gene, child2_gene = crossover_blend(parent1.get_gene(),
        parent2.get_gene(), 1, -10, 10)
        return Individual([child1_gene]), Individual([child2_gene])
	# 주어진 개체의 유전자를 mutate_gaussian 함수를 통해 돌연변이시켜 새 개체를 반환
    @classmethod
    def mutate(cls, ind): # 돌연변이
        mutated_gene = mutate_gaussian(ind.get_gene(), 0, 1, -10, 10)
        return Individual([mutated_gene])
	# select_tournament 함수를 호출하여 선택 연산을 수행
    @classmethod
    def select(cls, population): # 자연 선택
        return select_tournament(population, tournament_size = 3)
	# 무작위로 유전자를 생성하여 개체를 만듦
    @classmethod
    def create_random(cls):
        return Individual([random.randrange(-1000, 1000) / 100])

# 초기 랜덤 시드 설정
random.seed(52)
# random.seed(16)  # local maximum

# GA 파라미터 설정
POPULATION_SIZE = 10
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.1
MAX_GENERATIONS = 10

# 초기 개체군 생성 및 첫 세대 시각화
first_population = [Individual.create_random() for _ in range(POPULATION_SIZE)]
plot_population(first_population, 0)

generation_number = 0
population = first_population.copy()

while generation_number < MAX_GENERATIONS:
    generation_number += 1
    
    # 1. 선택 (Selection)
    offspring = Individual.select(population)
    
    # 2. 교차 (Crossover)
    crossed_offspring = []
    # 선택된 개체군을 2개씩 짝지어 교차 진행
    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CROSSOVER_PROBABILITY:
            kid1, kid2 = Individual.crossover(ind1, ind2)
            crossed_offspring.append(kid1)
            crossed_offspring.append(kid2)
        else:
            crossed_offspring.append(ind1)
            crossed_offspring.append(ind2)
    
    # 3. 돌연변이 (Mutation)
    mutated_offspring = []
    for mutant in crossed_offspring:
        if random.random() < MUTATION_PROBABILITY:
            new_mutant = Individual.mutate(mutant)
            mutated_offspring.append(new_mutant)
        else:
            mutated_offspring.append(mutant)
    
    # 다음 세대로 업데이트 및 시각화
    population = mutated_offspring.copy()
    plot_population(population, generation_number)