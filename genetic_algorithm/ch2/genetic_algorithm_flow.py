'''
전체 유전 알고리즘의 흐름(Selection, Crossover, Mutation)을 순서대로 실행하는 메인 스크립트
'''

import random

from crossover import crossover
from individual import create_random_individual
from mutate import mutate
from population import plot_population
from selection import select_tournament

if __name__ == '__main__':
    '''
    코드 흐름:
    초기 설정:
        POPULATION_SIZE, CROSSOVER_PROBABILITY, MUTATION_PROBABILITY, MAX_GENERATIONS 등의 파라미터를 설정함
        random.seed()를 설정하여 무작위성의 재현성을 확보함
    초기 개체군 생성:
        create_random_individual()을 이용해 인구를 생성함
    세대 반복:
        매 세대마다 선택(select_tournament 함수 사용),
        선택된 개체들을 두 개씩 짝지어 CROSSOVER_PROBABILITY 확률로 교차 수행,
        그 결과에 대해 MUTATION_PROBABILITY 확률로 돌연변이 수행함
        새로운 개체군으로 업데이트 후, plot_population()을 통해 시각적으로 확인함
'''

    POPULATION_SIZE = 10
    CROSSOVER_PROBABILITY = .8
    MUTATION_PROBABILITY = .1
    MAX_GENERATIONS = 10

    random.seed(29)

    population = [create_random_individual() for _ in range(POPULATION_SIZE)]

    for generation_number in range(POPULATION_SIZE):
        # SELECTION
        selected = select_tournament(population, 3)
        # CROSSOVER
        crossed_offspring = []
        for ind1, ind2 in zip(selected[::2], selected[1::2]):
            if random.random() < CROSSOVER_PROBABILITY:
                children = crossover(ind1, ind2)
                crossed_offspring.append(children[0])
                crossed_offspring.append(children[1])
            else:
                crossed_offspring.append(ind1)
                crossed_offspring.append(ind2)
        # MUTATION
        mutated = []
        for ind in crossed_offspring:
            if random.random() < MUTATION_PROBABILITY:
                mutated.append(mutate(ind))
            else:
                mutated.append(ind)

        population = mutated

        plot_population(population)
