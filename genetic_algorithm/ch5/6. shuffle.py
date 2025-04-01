import copy
import random

def mutation_shuffle(ind):
    # 원본 개체(ind)를 유지하면서 변이 진행
    mut = copy.deepcopy(ind)
    print(f'[debug] 초기 개체 : {ind}')
    # 리스트 내에서 두 인덱스를 선택하고 오름차순 정렬 (i, j)
    pos = sorted(random.sample(range(0, len(mut)), 2))
    print(f'[debug] 두 인덱스를 선택(개체x) : {pos}')
    # 선택된 구간 [i, j]의 서브리스트 추출
    subrange = mut[pos[0]:pos[1] + 1]
    print(f'[debug] subrange : {subrange}')
    # 추출된 서브리스트를 무작위로 섞음 (내부 순서 변화)
    random.shuffle(subrange)
    # 섞인 서브리스트를 원래 위치에 대입
    mut[pos[0]:pos[1] + 1] = subrange
    print(f'[debug] mute_range : {mut[pos[0]:pos[1] + 1]}')
    return mut

if __name__ == '__main__':
    random.seed(3)
    # 1부터 5까지의 숫자로 구성된 순열 개체 생성
    # ind = list(range(1, 6))
    # ind = ['A', 'B', 'C', 'D', 'E']
    ind = ['T1', 'T2', 'T3', 'T4', 'T5']
    mut = mutation_shuffle(ind)
    print(f'Original: {ind}')
    print(f'Mutated: {mut}')
