'''
두 부모 개체의 유전자를 섞어 새로운 자식을 만드는 교차(crossover) 연산을 구현함
'''
import random
import numpy as np
import matplotlib.pyplot as plt

from individual import create_random_individual, create_individual
from fitness import fitness
from settings import MIN_BORDER, MAX_BORDER


def gene_constraints(g, min_ = MIN_BORDER, max_ = MAX_BORDER):
    '''
    설명: g가 지정된 최소/최대 범위를 벗어나지 않도록 보정함
    결과: 보정된 g 반환함
    '''
    if max_ and g > max_:
        g = max_
    if min_ and g < min_:
        g = min_
    return g

def crossover_blend(g1, g2, alpha = 0.3):
    '''
    설명:
        두 부모의 유전자 g1, g2를 섞는 블렌딩(crossover blend) 방식임
        alpha 값에 따라 혼합 정도(shift)가 결정됨
        계산 후 gene_constraints를 통해 범위 내로 보정함
    결과: 두 자식 유전자 값 반환함
    '''
    shift = (1. + 2. * alpha) * random.random() - alpha
    new_g1 = (1. - shift) * g1 + shift * g2
    new_g2 = shift * g1 + (1. - shift) * g2

    return gene_constraints(new_g1), gene_constraints(new_g2)


def crossover(ind1, ind2):
    '''
    설명:
        부모 개체들의 get_gene() 결과(즉, 유전자)를 가져와 crossover_blend로 섞고,
        create_individual을 호출해 두 자식 개체를 생성함
    결과: 자식 개체 리스트 반환함
    '''
    offspring_genes = crossover_blend(ind1.get_gene(), ind2.get_gene())
    return [create_individual(offspring_genes[0]),
            create_individual(offspring_genes[1])]


if __name__ == '__main__':
    '''
    두 개의 무작위 개체를 생성하고, 교차 후 부모와 자식의 유전자 및 적합도를 그래프로 시각화
    '''
    random.seed(30)

    p_1 = create_random_individual()
    p_2 = create_random_individual()

    offspring = crossover(p_1, p_2)

    c_1 = offspring[0]
    c_2 = offspring[1]

    x = np.linspace(MIN_BORDER, MAX_BORDER)
    plt.plot(x, fitness(x), '--', color = 'blue')
    plt.plot(
        [p_1.get_gene(), p_2.get_gene()],
        [p_1.fitness, p_2.fitness],
        'o', markersize = 15, color = 'orange'
    )
    plt.plot(
        [c_1.get_gene(), c_2.get_gene()],
        [c_1.fitness, c_2.fitness],
        's', markersize = 15, color = 'green'
    )
    plt.title("Circle : Parents, Square: Children")
    plt.show()
