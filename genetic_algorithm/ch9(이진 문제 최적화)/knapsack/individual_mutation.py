# individual_mutation.py
import random
import matplotlib.pyplot as plt

from individual import Individual
from random_set_generator import random_set_generator
from toolbox import mutation_bit_flip

# 돌연변이 함수: 기존 gene_list에 비트 플립 돌연변이 적용 후 새로운 Individual 생성
def mutate(ind):
    mutated_gene = mutation_bit_flip(ind.gene_list)
    return Individual(mutated_gene)

if __name__ == '__main__':
    # 시드 설정
    random.seed(1)
    random.seed(63)

    # 아이템 집합 생성 및 초기화
    items = random_set_generator(1, 100, 0.1, 7, 200)
    Individual.set_items(items)
    Individual.set_max_weight(10)

    # 특정 gene_set 생성: 미리 정의된 인덱스(inclusions)는 1로 설정
    gene_set = [0] * len(items)
    inclusions = [2, 30, 34, 42, 48, 64, 85, 104, 113, 119, 157, 174]
    for i in inclusions:
        gene_set[i] = 1
    ind = Individual(gene_set)

    alive = 0   # 돌연변이 후 fitness가 0이 아닌 경우
    killed = 0  # 돌연변이 후 fitness가 0인 경우(배낭 용량 초과 등으로 소멸)

    # 1000번 돌연변이 연산 실행
    for _ in range(1000):
        mutated = mutate(ind)
        if mutated.fitness == 0:
            killed += 1
        else:
            alive += 1

    print(f'Best individual: {ind.fitness}')
    # 결과를 파이 차트로 시각화
    labels = 'Killed', 'Alive'
    sizes = [killed, alive]
    plt.pie(sizes, labels=labels)
    plt.show()
