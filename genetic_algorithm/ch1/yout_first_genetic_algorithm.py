import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    '''
    역할: 유전 알고리즘의 평가 기준(적합도)을 계산함
    설명: x에 대해 sin(x) - 0.2×|x| 값을 계산함
    결과: 계산된 함수 값(적합도)을 반환함
    '''
    return np.sin(x) - .2 * abs(x)

def _utils_constraints(g, min, max):
    '''
    역할: 입력된 값 g가 지정된 최소값(min)과 최대값(max) 범위를 넘지 않도록 조정함
    설명: 만약 g가 max보다 크면 max로, min보다 작으면 min으로 바꿔줌
    결과: g가 지정한 범위 내에 있도록 만든 값을 반환함
    '''
    if max and g > max:
        g = max
    if min and g < min:
        g = min
    return g

def select_tournament(population, tournament_size):
    '''
    역할: 인구 집단에서 무작위로 몇 개체(토너먼트 후보)를 뽑고, 그 중 적합도가 가장 높은 개체를 선택함
    설명:
    - 전체 인구 수만큼 반복하면서, 매번 토너먼트 크기만큼 무작위 후보를 선택하고, 그 중 최고 적합도(즉, 함수값이 가장 높은)를 고름
    결과: 선택된 개체들로 구성된 새로운 집단(리스트)을 반환함
    '''
    new_offspring = []
    for _ in range(len(population)):
        candidates = [random.choice(population) for _ in
        range(tournament_size)]
        new_offspring.append(max(candidates, key = lambda ind:
        ind.fitness))
    return new_offspring

def crossover_blend(g1, g2, alpha, min = None, max = None):
    '''
    역할: 두 부모의 유전자 값(g1, g2)을 섞어 새로운 자식 유전자를 만들어냄
    설명:
    - 먼저 무작위로 섞이는 정도(shift)를 결정함, random.random(0~1사이 무작위 값)
    - shift 값을 이용해 두 부모의 유전자를 일정 비율로 섞어 두 개의 자식 유전자를 계산함
    - 계산된 자식 유전자 값들이 지정된 범위(min, max)를 벗어나지 않도록 _utils_constraints 함수를 거침
    결과: 두 자식의 유전자 값을 튜플로 반환함
    '''
    shift = (1. + 2. * alpha) * random.random() - alpha
    new_g1 = (1. - shift) * g1 + shift * g2
    new_g2 = shift * g1 + (1. - shift) * g2
    return _utils_constraints(new_g1, min, max), _utils_constraints(new_g2, min, max)

def mutate_gaussian(g, mu, sigma, min = None, max = None):
    '''
    역할: 주어진 유전자 g에 가우시안(정규분포) 무작위 변화를 주어 돌연변이를 발생시킴
    설명:
    - 평균(mu)과 표준편차(sigma)를 가진 가우시안 난수를 g에 더함
    - 더한 후, 결과값이 지정된 범위 내에 있도록 _utils_constraints로 보정함
    결과: 돌연변이가 적용된 새로운 유전자 값을 반환함
    '''
    mutated_gene = g + random.gauss(mu, sigma)
    return _utils_constraints(mutated_gene, min, max)

def get_best(population):
    '''
    역할: 현재 인구(개체 집단) 중에서 가장 높은 적합도를 가진 개체를 찾음
    설명: 인구의 첫 번째 개체를 기준으로, 이후 개체들과 비교하면서 가장 좋은(적합도가 높은) 개체를 업데이트함
    결과: 가장 좋은 개체를 반환함
    '''
    best = population[0]
    for ind in population:
        if ind.fitness > best.fitness:
            best = ind
    return best

def plot_population(population, number_of_population):
    '''
    역할: 현재 세대의 개체 분포와 함수 곡선을 시각적으로 보여줌
    설명:
    - x축에 -10부터 10까지의 값을 생성하고, 함수 func(x) 그래프를 파란 점선으로 그림
    - 각 개체의 유전자(여기서는 x값)와 적합도를 주황색 점으로 표시함
    - 가장 좋은 개체는 녹색 사각형으로 표시함
    - 세대 번호를 제목으로 표시함
    결과: 그래프를 출력함
    '''
    best = get_best(population)
    x = np.linspace(-10, 10)
    plt.plot(x, func(x), '--', color = 'blue')
    plt.plot(
        [ind.get_gene() for ind in population],
        [ind.fitness for ind in population],
        'o', color = 'orange'
        )
    plt.plot([best.get_gene()], [best.fitness], 's', color = 'green')
    plt.title(f"Generation number {number_of_population}")
    plt.show()
    plt.close()

class Individual: # 클래스 역할: 한 개체(해답 후보)를 나타내며, 해당 개체의 유전자와 적합도(함수 값을 계산하여)를 저장함

    def __init__(self, gene_list: List[float]) -> None:
        '''
        설명: 유전자 리스트(gene_list)를 입력받아 저장하고, 이 중 첫 번째 유전자 값을 이용해 func 함수를 통해 적합도를 계산함
        결과: 개체가 생성됨
        '''
        self.gene_list = gene_list
        self.fitness = func(self.gene_list[0])

    def get_gene(self):
        '''
        역할: 개체의 유전자 중 첫 번째 값을 반환함
        설명: 개체의 대표 유전자로 사용함
        '''
        return self.gene_list[0]
    
    @classmethod
    def crossover(cls, parent1, parent2):
        '''
        역할: 두 부모 개체를 받아 교차를 수행함
        설명: 부모들의 유전자를 get_gene()로 가져와서 crossover_blend 함수를 이용해 섞은 후, 두 자식 개체(Individual)를 새로 생성함
        결과: 두 자식 개체를 반환함
        '''
        child1_gene, child2_gene = crossover_blend(parent1.get_gene(),
        parent2.get_gene(), 1, -10, 10)
        return Individual([child1_gene]), Individual([child2_gene])
    
    @classmethod
    def mutate(cls, ind):
        '''
        역할: 주어진 개체에 대해 돌연변이를 적용함
        설명: 개체의 유전자를 mutate_gaussian 함수를 사용해 돌연변이를 발생시킨 후, 새 개체를 생성함
        결과: 돌연변이가 적용된 새로운 개체를 반환함
        '''
        mutated_gene = mutate_gaussian(ind.get_gene(), 0, 1, -10, 10)
        return Individual([mutated_gene])
    
    @classmethod
    def select(cls, population):
        '''
        역할: 주어진 인구 집단에서 자연 선택을 수행함
        설명: select_tournament 함수를 호출하여, 토너먼트 방식으로 좋은 개체들을 선택함
        결과: 선택된 개체들로 구성된 새로운 리스트를 반환함
        '''
        return select_tournament(population, tournament_size = 3)
    
    @classmethod
    def create_random(cls):
        '''
        역할: 무작위로 유전자를 생성하여 새로운 개체를 만듦
        설명: -1000부터 1000 사이의 정수를 100으로 나눈 값(즉, -10부터 10 사이의 실수)을 무작위로 선택함
        결과: 임의의 개체를 생성함
        '''
        return Individual([random.randrange(-1000, 1000) / 100])

# 초기 랜덤 시드 설정
random.seed(52)
# random.seed(16)  # local maximum

# GA 파라미터 설정
POPULATION_SIZE = 10
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.1
MAX_GENERATIONS = 10

first_population = [Individual.create_random() for _ in range(POPULATION_SIZE)] # 10개의 무작위 개체 생성
plot_population(first_population, 0) # 생성된 개체들을 시각화해 첫 세대 확인

generation_number = 0
population = first_population.copy()

while generation_number < MAX_GENERATIONS:
    generation_number += 1
    
    # 1. Selection : 토너먼트 방식으로 좋은 개체 선택
    offspring = Individual.select(population)
    
    # 2. Crossover
    crossed_offspring = []
    # 선택된 개체군을 2개씩 짝지어 교차 진행
    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CROSSOVER_PROBABILITY: # 확률에 따라 교차 진행
            kid1, kid2 = Individual.crossover(ind1, ind2)
            crossed_offspring.append(kid1)
            crossed_offspring.append(kid2)
        else: # 확률에 걸리지 않으면 부모 그대로 다음 세대로 넘김
            crossed_offspring.append(ind1)
            crossed_offspring.append(ind2)
    
    # 3. Mutation
    mutated_offspring = []
    for mutant in crossed_offspring:
        if random.random() < MUTATION_PROBABILITY: # 확률에 따라 돌연변이 연산 진행
            new_mutant = Individual.mutate(mutant)
            mutated_offspring.append(new_mutant)
        else:
            mutated_offspring.append(mutant)
    
    # 다음 세대로 업데이트 및 시각화
    population = mutated_offspring.copy()
    plot_population(population, generation_number) 