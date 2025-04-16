import copy
import random

def mutation_bit_flip(ind):
    # 원본 개체(ind)를 deepcopy하여 새로운 개체(mut)를 생성
    mut = copy.deepcopy(ind)
    # 리스트 내에서 무작위로 하나의 인덱스 선택
    pos = random.randint(0, len(ind) - 1)
    # 선택된 위치의 gene 값 가져오기
    g1 = mut[pos]
    # (g1 + 1) % 2를 통해 비트 반전: 0 -> 1, 1 -> 0
    mut[pos] = (g1 + 1) % 2
    return mut

# 예시 실행 코드
if __name__ == '__main__':
    random.seed(1)
    # 0 또는 1로 구성된 5길이 이진 개체 생성
    ind = [random.randint(0, 1) for _ in range(5)]
    mut = mutation_bit_flip(ind)
    print(f'Original: {ind}')
    print(f'Mutated: {mut}')
    # 원본 개체와 돌연변이된 개체 비교
    print(f'Original and Mutated are the same: {ind == mut}')
    # 원본 개체와 돌연변이된 개체의 차이점 출력
    for i in range(len(ind)):
        if ind[i] != mut[i]:
            print(f'Index {i}: Original {ind[i]} -> Mutated {mut[i]}')
