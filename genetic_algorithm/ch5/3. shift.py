import copy
import random
from math import copysign

def mutation_shift(ind):
    # 원본 개체(ind)를 deepcopy하여 새로운 개체(mut)를 생성
    mut = copy.deepcopy(ind)
    # 리스트 내에서 서로 다른 두 인덱스 선택
    pos = random.sample(range(0, len(mut)), 2)
    # 첫 번째 위치의 gene를 임시 저장
    g1 = mut[pos[0]]
    # 두 인덱스의 차이에 따른 이동 방향 결정: 양수면 +1, 음수면 -1
    dir = int(copysign(1, pos[1] - pos[0]))
    # pos[0]부터 pos[1]까지 d 방향으로 순환 이동
    for i in range(pos[0], pos[1], dir):
        mut[i] = mut[i + dir]
    # 이동한 마지막 자리에 원래 저장해둔 gene을 배치
    mut[pos[1]] = g1
    return mut

# 예시 실행 코드
if __name__ == '__main__':
    random.seed(21)
    # 1부터 5까지의 숫자로 구성된 순열 개체 생성
    ind = list(range(1, 6))
    mut = mutation_shift(ind)
    print(f'Original: {ind}')
    print(f'Mutated: {mut}')
