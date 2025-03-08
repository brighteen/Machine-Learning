'''
개체들을 선택하는 선택(Selection) 연산을 구현함 (여기서는 토너먼트 선택 방식을 사용함)
'''
import random

from individual import create_random_individual


def select_tournament(population, tournament_size):
    '''
    설명:
        전체 인구 수만큼 반복하면서, 매번 tournament_size(예, 3)만큼 무작위로 후보 개체를 뽑고,
        그 중 적합도가 가장 높은 개체를 선택하여 새로운 인구 리스트를 만듦
    결과: 선택된 개체들이 담긴 리스트 반환함
    '''
    new_offspring = []
    for _ in range(len(population)):
        candidates = [random.choice(population) for _ in range(tournament_size)]
        new_offspring.append(max(candidates, key = lambda ind: ind.fitness))
    return new_offspring


if __name__ == '__main__':
    '''
    5개의 개체로 구성된 인구를 생성 후, 토너먼트 선택을 적용한 결과를 출력
    '''
    random.seed(29)
    POPULATION_SIZE = 5

    generation_1 = [create_random_individual() for _ in range(POPULATION_SIZE)]
    # print('[debug] :', generation_1)
    generation_2 = select_tournament(generation_1, 3)
    # print('\n[debug] :', generation_2)

    print("\nGeneration 1")
    [print(ind) for ind in generation_1]

    print("Generation 2")
    [print(ind) for ind in generation_2]
