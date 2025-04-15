import random
from individual import Individual
from random_set_generator import random_set_generator

if __name__ == '__main__':
    random.seed(1)
    # 아이템 집합 생성
    items = random_set_generator(1, 100, 0.1, 7, 200)
    Individual.set_items(items)
    Individual.set_max_weight(10)
    # 기본적인 방식으로 무작위 개체군 생성
    population = [Individual.create_random() for _ in range(1000)]
    average_weight = sum([ind.total_weight() for ind in population]) / len(population)
    print(f'Average weight of population: {average_weight}')
