import random  # 난수 생성을 위한 라이브러리 가져오기
from math import floor  # 소수점 아래 버림 함수 가져오기

import pandas as pd  # 데이터 프레임 처리를 위한 판다스 라이브러리 가져오기
import matplotlib.pyplot as plt  # 시각화를 위한 matplotlib 라이브러리 가져오기


class Individual:  # 개체 클래스 정의
    cache = {}  # 적합도 값을 저장하는 캐시 딕셔너리
    cache_hit = 0  # 캐시 히트 카운터
    counter = 0  # 생성된 개체 수 카운터
    period = 0  # 스케줄링 기간
    employees = 0  # 직원 수

    @classmethod
    def set_fitness_function(cls, fun):  # 적합도 함수 설정 클래스 메서드
        cls.fitness_function = fun  # 클래스 변수에 적합도 함수 할당

    @classmethod
    def set_period(cls, period):  # 기간 설정 클래스 메서드
        cls.period = period  # 클래스 변수에 기간 할당

    @classmethod
    def set_employees(cls, employees):  # 직원 수 설정 클래스 메서드
        cls.employees = employees  # 클래스 변수에 직원 수 할당

    @classmethod
    def generate_random(cls):  # 무작위 개체 생성 클래스 메서드
        return Individual([random.choice([0, 1]) for _ in range(cls.period * cls.employees * 3)])  # 무작위 0, 1 리스트로 개체 생성 및 반환

    def __init__(self, gene_list) -> None:  # 생성자 메서드
        self.gene_list = gene_list  # 유전자 리스트 저장
        gene_hash = ''.join([str(g) for g in gene_list])  # 유전자 리스트를 문자열로 변환하여 해시 키 생성
        cache = self.__class__.cache  # 클래스 캐시 참조
        if gene_hash not in cache.keys(): # 없으면 적합도를 계산하여 캐시에 저장
            cache[gene_hash] =\
                self.__class__.fitness_function(self.create_schedule())  # 스케줄 생성 후 적합도 계산 및 캐시 저장
        else: # 이미 계산된 적합도 값이 있으면 재사용하고 cache_hit를 증가
            self.__class__.cache_hit += 1  # 캐시 히트 카운터 증가

        self.fitness = cache[gene_hash]  # 개체의 적합도 값 저장
        self.__class__.counter += 1  # 개체 생성 카운터 증가

    def create_schedule(self):  # 스케줄 생성 메서드
        t = {}  # 임시 딕셔너리 생성
        for e in range(1, self.employees + 1):  # 각 직원별로 반복
            shift_len = 3 * self.period  # 한 직원의 총 교대 수 계산 (3교대 * 기간)
            t[e] = self.gene_list[shift_len * (e - 1): shift_len * e]  # 해당 직원의 교대 일정 추출
        schedule_df = pd.DataFrame(data = t)  # 딕셔너리를 데이터프레임으로 변환
        return schedule_df  # 스케줄 데이터프레임 반환

    def plot_schedule(self):  # 스케줄 시각화 메서드
        schedule_df = self.create_schedule()  # 스케줄 데이터프레임 생성
        x_labels = []  # x축 레이블 리스트 초기화
        shift_names = {0: 'mor', 1: 'mid', 2: 'evn'}  # 교대 이름 딕셔너리 (morning, midday, evening)
        for i in range(0, 3 * self.period):  # 각 교대에 대해 반복
            day = floor(i / 3) + 1  # 일자 계산 (1부터 시작)
            shift = shift_names[i % 3]  # 교대 이름 가져오기
            x_labels.append(f'Day {day} : {shift}')  # x축 레이블에 일자와 교대 추가
        plt.xticks(list(range(0, 3 * self.period)), x_labels, rotation = 90)  # x축 눈금 설정 (90도 회전)
        y_labels = []  # y축 레이블 리스트 초기화
        for i in range(0, self.employees):  # 각 직원에 대해 반복
            y_labels.append(f'Emp: {i+1}')  # y축 레이블에 직원 번호 추가
        plt.yticks(list(range(0, self.employees)), y_labels)  # y축 눈금 설정
        plt.imshow(schedule_df.T, cmap = 'binary')  # 스케줄 이미지로 시각화 (전치하여 직원을 행으로 표시)
        plt.title(f'Fitness: {self.fitness}')  # 제목에 적합도 값 표시
        plt.show()  # 그래프 출력
        plt.close()  # 그래프 창 닫기


if __name__ == '__main__':  # 스크립트가 직접 실행될 때만 실행되는 코드 블록

    random.seed(9)  # 난수 생성기 시드 설정 (재현성을 위해)

    Individual.set_employees(5)  # 직원 수를 5명으로 설정
    Individual.set_period(7)  # 스케줄링 기간을 7일로 설정


    def fitness_function(df):  # 간단한 적합도 함수 정의
        return 0  # 항상 0 반환 (실제로는 의미 있는 값을 반환해야 함)


    Individual.set_fitness_function(fitness_function)  # 적합도 함수 설정

    ind = Individual.generate_random()  # 무작위 개체 생성
    ind.plot_schedule()  # 생성된 스케줄 시각화
