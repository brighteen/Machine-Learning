import copy
import random
from math import sin
from typing import List

# 피트니스 함수: f(x) = sin(x) - 0.2 * |x|
def func(x):
    return sin(x) - .2 * abs(x)

# Individual 클래스: 하나의 개체를 나타내며, 첫 번째 gene을 사용하여 피트니스 계산
class Individual:
    def __init__(self, gene_list: List[float]) -> None:
        self.gene_list = gene_list
        self.fitness = func(self.gene_list[0])
    def __str__(self):
        return f'x: {self.gene_list[0]}, fitness: {self.fitness}'

def mutation_fitness_driven_random_deviation(ind, mu, sigma, p, max_tries=3):
    # 최대 max_tries번 변이 시도
    for t in range(0, max_tries):
        # 원본 gene 리스트를 deepcopy하여 변이 시도
        mut_genes = copy.deepcopy(ind.gene_list)
        # 각 gene에 대해 확률 p로 가우시안 노이즈 추가
        for i in range(len(mut_genes)):
            if random.random() < p:
                # 수식: x' = x + δ, δ ~ N(mu, sigma^2)
                mut_genes[i] = mut_genes[i] + random.gauss(mu, sigma)
        # 변이된 gene 리스트로 새 개체 생성
        mut = Individual(mut_genes)
        # 만약 변이 후 피트니스가 개선되었다면(즉, f(new) > f(original)), 변이 개체 반환
        if ind.fitness < mut.fitness:
            return mut
    # max_tries 내에 개선되지 않으면 원본 개체 반환
    return ind

# 예시 실행 코드
if __name__ == '__main__':
    random.seed(14)
    # -10과 10 사이의 값으로 초기 개체 생성
    ind = Individual([random.uniform(-10, 10)])
    # 변이 확률 3, mu=0, sigma=1 적용, 최대 3회 시도
    mut = mutation_fitness_driven_random_deviation(ind, 0, 1, 0.3)
    print(f'Original: ({ind})')
    print(f'Mutated: ({mut})')
