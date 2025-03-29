import copy
import random

def mutation_shuffle(ind):
    # 원본 개체(ind)를 deepcopy하여 새로운 개체(mut)를 생성
    mut = copy.deepcopy(ind)
    # 리스트 내에서 두 인덱스를 선택하고 오름차순 정렬 (i, j)
    pos = sorted(random.sample(range(0, len(mut)), 2))
    # 선택된 구간 [i, j]의 서브리스트 추출
    subrange = mut[pos[0]:pos[1] + 1]
    # 추출된 서브리스트를 무작위로 섞음 (내부 순서 변화)
    random.shuffle(subrange)
    # 섞인 서브리스트를 원래 위치에 대입
    mut[pos[0]:pos[1] + 1] = subrange
    return mut

# 예시 실행 코드
if __name__ == '__main__':
    random.seed(13)
    # 1부터 5까지의 숫자로 구성된 순열 개체 생성
    ind = list(range(1, 6))
    mut = mutation_shuffle(ind)
    print(f'Original: {ind}')
    print(f'Mutated: {mut}')
