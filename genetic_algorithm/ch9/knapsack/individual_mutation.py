import random
import matplotlib.pyplot as plt
from individual import Individual
from random_set_generator import random_set_generator
from toolbox import mutation_bit_flip

# 돌연변이 함수: bit flip 연산 수행 후 새로운 Individual 반환
def mutate(ind):
    mutated_gene = mutation_bit_flip(ind.gene_list)
    return Individual(mutated_gene)

if __name__ == '__main__':
    random.seed(1)
    random.seed(63)
    # 아이템 집합 생성 및 설정
    items = random_set_generator(1, 100, 0.1, 7, 200)
    Individual.set_items(items)
    Individual.set_max_weight(10)
    
    # 특정 유전자 벡터 생성: 전체 길이 만큼 0으로 초기화, 일부 인덱스에는 1 설정
    gene_set = [0] * len(items)
    inclusions = [2, 30, 34, 42, 48, 64, 85, 104, 113, 119, 157, 174]
    for i in inclusions:
        gene_set[i] = 1
    ind = Individual(gene_set)
    
    alive = 0
    killed = 0

    # 돌연변이 1000회 수행하여, 돌연변이 후 적합도가 0이 되는 경우와 그렇지 않은 경우 카운트
    for _ in range(1000):
        mutated = mutate(ind)
        if mutated.fitness == 0:
            killed += 1
        else:
            alive += 1

    print(f'Best individual: {ind.fitness}')
    labels = 'Killed', 'Alive'
    sizes = [killed, alive]
    # 파이 차트로 돌연변이 결과 시각화
    plt.pie(sizes, labels=labels)
    plt.show()
