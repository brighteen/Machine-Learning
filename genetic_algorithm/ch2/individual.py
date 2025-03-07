'''
한 개체(해답 후보)를 정의하는 클래스와, 개체를 생성하는 함수들을 제공
'''

import random

from fitness import fitness
from settings import MIN_BORDER, MAX_BORDER


class Individual:

    def __init__(self, gene_list, fitness_function) -> None:
        '''
        gene_list와 fitness_function을 받아 개체를 생성함
        gene_list[0]을 이용해 fitness_function (즉, fitness 함수)을 호출하여 적합도(fitness)를 계산함
        '''
        self.gene_list = gene_list
        self.fitness = fitness_function(self.gene_list[0])

    def __str__(self) -> str:
        '''
        개체의 유전자 값과 적합도를 보기 좋게 문자열로 반환함
        '''
        return f"{self.gene_list[0]:.2f} -> {self.fitness:.2f}"

    def get_gene(self):
        '''
        개체의 대표 유전자(여기서는 gene_list의 첫 번째 값)를 반환함
        '''
        return self.gene_list[0]


def create_random_individual():
    '''
    MIN_BORDER와 MAX_BORDER 사이의 무작위 실수값을 생성해 개체를 만듦
    fitness 함수가 적합도 평가 함수로 사용됨
    '''
    return Individual([random.uniform(MIN_BORDER, MAX_BORDER)], fitness)


def create_individual(gene):
    '''
    주어진 gene 값을 이용해 개체를 생성함
    역시 fitness 함수를 이용해 적합도를 계산함
    '''
    return Individual([gene], fitness)
