import copy
import random

def mutation_random_deviation(ind, mu, sigma, p):
    # 원본 개체(ind)를 deepcopy하여 새로운 개체(mut)를 생성
    mut = copy.deepcopy(ind)
    # 각 gene에 대해 반복
    for i in range(len(mut)):
        # 0과 1 사이의 난수 생성, p보다 작으면 변이 발생
        if random.random() < p:
            # 수식: x(mutate_gene)' = x(gene) + δ(가우시안 노이즈), where δ ~ N(mu, sigma^2)
            mut[i] = mut[i] + random.gauss(mu, sigma)
    return mut

if __name__ == '__main__':
    random.seed(0)  # 재현성을 위한 시드 설정
    # 2개의 gene로 이루어진 개체 생성 (0~10 사이의 실수)
    ind = [random.uniform(0, 10) for _ in range(2)]
    # 평균 0, 표준편차 1, 변이 확률(p) 0.3
    mut = mutation_random_deviation(ind, 0, 1, 0.3)
    print(f'Original: {ind}')
    print(f'Mutated: {mut}')
