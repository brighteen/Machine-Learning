import copy
import random

def mutation_inversion(ind):
    # 원본 개체(ind)를 deepcopy하여 새로운 개체(mut)를 생성
    mut = copy.deepcopy(ind)
    # 변이 전 순서를 보존하기 위해 원본 복사본 생성
    temp = copy.deepcopy(ind)
    # 리스트 내에서 두 인덱스를 선택하고 오름차순 정렬 (i, j)
    pos = sorted(random.sample(range(0, len(mut)), 2))
    # pos[0]부터 pos[1]까지의 구간을 역순으로 재배치
    for i in range(0, (pos[1] - pos[0]) + 1):
        # 수식: new_gene[pos[0]+i] = original_gene[pos[1]-i]
        mut[pos[0] + i] = temp[pos[1] - i]
    return mut

# 예시 실행 코드
if __name__ == '__main__':
    random.seed(5)
    # 1부터 5까지의 숫자로 구성된 순열 개체 생성
    ind = list(range(1, 6))
    mut = mutation_inversion(ind)
    print(f'Original: {ind}')
    print(f'Mutated: {mut}')
