import copy
import random

def mutation_exchange(ind):
    # 원본 개체(ind)를 deepcopy하여 새로운 개체(mut)를 생성
    mut = copy.deepcopy(ind)
    # 리스트 내에서 서로 다른 두 인덱스 선택
    pos = random.sample(range(0, len(mut)), 2)
    # 선택된 두 위치의 gene을 교환
    g1 = mut[pos[0]]
    g2 = mut[pos[1]]
    mut[pos[1]] = g1
    mut[pos[0]] = g2
    return mut

# 예시 실행 코드
if __name__ == '__main__':
    random.seed(1)  # 재현성을 위한 시드 설정
    # 1부터 6까지의 숫자로 구성된 순열 개체 생성
    ind = list(range(1, 7))
    mut = mutation_exchange(ind)
    print(f'Original: {ind}')
    print(f'Mutated: {mut}')
