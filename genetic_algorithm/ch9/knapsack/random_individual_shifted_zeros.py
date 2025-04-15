import random
from individual import Individual
from random_set_generator import random_set_generator

# 지정된 개수의 0과 1을 섞어 개체 생성
# gene_len: 전체 길이, zeros: 선택 가능한 0의 개수, ones: 선택 가능한 1의 개수
def create_random_individual(gene_len, zeros=1, ones=1):
    # 가능한 유전자 집합 구성 (예: 0이 여러 개, 1이 여러 개)
    s = ([0] * zeros) + ([1] * ones)
    # gene_len 길이만큼 s에서 무작위로 선택하여 개체 생성
    return Individual([random.choice(s) for _ in range(gene_len)])

if __name__ == '__main__':
    random.seed(1)
    items = random_set_generator(1, 100, 0.1, 7, 200)
    Individual.set_items(items)
    Individual.set_max_weight(10)
    # 50개의 0과 1 중 1은 1로 선택하는 방식으로 1000개 개체 생성
    population = [create_random_individual(len(items), 50, 1) for _ in range(1000)]
    average_weight = sum([ind.total_weight() for ind in population]) / len(population)
    print(f'Average weight of population: {average_weight}')
