'''
인구(개체 집단)의 통계(최고 적합도, 평균 적합도) 및 시각화를 위한 함수들을 제공
'''
import random
import numpy as np
import matplotlib.pyplot as plt

from individual import create_random_individual
from fitness import fitness
from settings import MIN_BORDER, MAX_BORDER


def get_best_individual(population):
    '''
    설명: 주어진 인구에서 가장 높은 적합도를 가진 개체를 찾음
    결과: 최고 개체 반환함
    '''
    return max(population, key = lambda ind: ind.fitness)


def get_average_fitness(population):
    '''
    설명: 인구의 모든 개체의 적합도 평균을 계산함
    결과: 평균 적합도 값 반환함
    '''
    return sum([i.fitness for i in population]) / len(population)


def plot_population(population):
    '''
    설명:
        x축: MIN_BORDER부터 MAX_BORDER까지의 범위에서 함수 func(x) (적합도 함수)를 파란 점선으로 그림
        각 개체의 유전자와 적합도를 주황색 점으로 표시함
        최고 개체는 녹색 사각형으로 표시하며, 인구의 평균 적합도도 회색 선으로 표시함
        제목에는 최고 개체와 평균 적합도 정보를 보여줌
    결과: 현재 인구의 분포를 시각적으로 확인할 수 있음
    '''
    best_ind = get_best_individual(population)
    best_fitness = best_ind.fitness
    average_fitness = get_average_fitness(population)

    x = np.linspace(MIN_BORDER, MAX_BORDER)
    plt.plot(x, fitness(x), '--', color = 'blue')
    plt.plot(
        [ind.get_gene() for ind in population],
        [ind.fitness for ind in population],
        'o', color = 'orange'
    )
    plt.plot(
        [best_ind.get_gene()], [best_ind.fitness],
        's', color = 'green')
    plt.plot(
        [MIN_BORDER, MAX_BORDER],
        [average_fitness, average_fitness],
        color = 'grey'
    )
    plt.title(f"Best Individual: {best_ind}, Best Fitness: {best_fitness:.2f} \n "
              f"Average Population Fitness: {average_fitness:.2f}"
              )
    plt.show()


if __name__ == '__main__':
    '''
    무작위 개체군을 생성하고 시각화
    '''
    POPULATION_SIZE = 10
    random.seed(22)

    population = [create_random_individual() for _ in range(POPULATION_SIZE)]
    plot_population(population)
