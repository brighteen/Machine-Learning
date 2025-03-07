'''
개체의 유전자에 돌연변이(mutation)를 적용하는 함수를 구현
'''
import random
import numpy as np
import matplotlib.pyplot as plt

from individual import create_random_individual, create_individual
from fitness import fitness
from settings import MIN_BORDER, MAX_BORDER


def gene_constraints(g, min_ = MIN_BORDER, max_ = MAX_BORDER):
    '''
    settings.py와 동일한 역할로, g값이 범위를 벗어나지 않도록 조정
    '''
    if max_ and g > max_:
        g = max_
    if min_ and g < min_:
        g = min_
    return g

def mutate_gaussian(g, mu, sigma):
    '''
    설명:
        주어진 g에 random.gauss(mu, sigma)로 생성한 가우시안 난수를 더해 돌연변이를 적용함
        결과값을 gene_constraints로 보정함
    결과: 돌연변이가 적용된 g 반환함
    '''
    mutated_gene = g + random.gauss(mu, sigma)
    return gene_constraints(mutated_gene)

def mutate(ind):
    '''
    설명:
        개체의 get_gene()를 사용해 유전자 값을 가져오고, mutate_gaussian을 적용하여 새로운 유전자 값을 만든 후,
        create_individual로 새 개체를 생성함
    결과: 돌연변이가 적용된 새로운 개체 반환함
    '''
    return create_individual(mutate_gaussian(ind.get_gene(), 0, 1))

if __name__ == '__main__':
    '''
    무작위 개체를 생성 후 돌연변이 적용한 결과를 그래프로 시각화
    '''
    random.seed(37)

    individual = create_random_individual()
    mutated = mutate(individual)

    x = np.linspace(MIN_BORDER, MAX_BORDER)
    plt.plot(x, fitness(x), '--', color = 'blue')
    plt.plot(
        [individual.get_gene()],
        [individual.fitness],
        'o', markersize = 20, color = 'orange'
    )
    plt.plot(
        [mutated.get_gene()],
        [mutated.fitness],
        's', markersize = 20, color = 'green'
    )
    plt.title("Circle : Before Mutation, Square: After Mutation")
    plt.show()
