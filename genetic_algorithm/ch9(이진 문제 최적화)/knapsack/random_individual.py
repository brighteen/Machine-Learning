# random_individual.py
import random

from individual import Individual
from random_set_generator import random_set_generator

if __name__ == '__main__':
    random.seed(1)
    items = random_set_generator(1, 100, 0.1, 7, 200)
    Individual.set_items(items)
    Individual.set_max_weight(10)

    # Individual 클래스의 create_random() 함수를 사용하여 개체군 생성
    population = [Individual.create_random() for _ in range(1000)]
    average_weight = sum([ind.total_weight() for ind in population]) / len(population)
    print(f'Average weight of population: {average_weight}')
