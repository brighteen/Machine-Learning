# random_individual_shifted_zeros.py
import random

from individual import Individual
from random_set_generator import random_set_generator

# 랜덤 개체 생성 함수
# gene_len: 유전자 길이, zeros: 선택될 0의 개수, ones: 선택될 1의 개수
def create_random_individual(gene_len, zeros=1, ones=1):
    # 0과 1의 리스트를 결합하여 후보군 생성
    s = ([0] * zeros) + ([1] * ones)
    # gene_len 만큼 후보군에서 랜덤 선택하여 gene list 생성
    return Individual([random.choice(s) for _ in range(gene_len)])

if __name__ == '__main__':
    random.seed(1)
    items = random_set_generator(1, 100, 0.1, 7, 200)
    Individual.set_items(items)
    Individual.set_max_weight(10)

    # 50개의 0과 1개의 1로 구성된 gene list를 가진 개체 1000개 생성
    population = [create_random_individual(len(items), 50, 1) for _ in range(1000)]
    average_weight = sum([ind.total_weight() for ind in population]) / len(population)
    print(f'Average weight of population: {average_weight}')
