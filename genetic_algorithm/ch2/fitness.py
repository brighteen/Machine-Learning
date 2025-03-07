'''
유전 알고리즘에서 각 개체의 해답이 얼마나 좋은지를 평가하는 함수 정의함
'''
import numpy as np

def fitness(x):
    '''
    함수 fitness(x):
        설명: x에 대해 sin(x) - 0.2×|x|를 계산함
        결과: 해당 값(함수값)이 높을수록 좋은 해답이라고 판단함
    '''
    return np.sin(x) - .2 * abs(x)

if __name__ == '__main__':
    '''
    테스트 코드: individual.py의 Individual 클래스를 이용해 적합도 계산 결과를 출력함
    '''
    from individual import Individual

    ind = Individual([1], fitness)
    print(f"Individual fitness: {ind.fitness}")
    # prints : 0.6414709848078965
