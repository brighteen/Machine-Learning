import random
from math import floor

import pandas as pd
import matplotlib.pyplot as plt

# 스케줄 평가 함수들을 임포트 (근무 편차와 휴식 조건 계산)
from schedule_analyzer import shift_deviations, shift_relax

  
# Individual 클래스: 근무 스케줄 해(solution)를 표현
class Individual:
    counter = 0       # 생성된 개체 수 카운터
    period = 0        # 스케줄 기간 (예: 일수)
    employees = 0     # 직원 수

    # fitness 함수를 설정하는 클래스 메서드
    @classmethod
    def set_fitness_function(cls, fun):
        cls.fitness_function = fun

    # 스케줄 기간을 설정하는 클래스 메서드
    @classmethod
    def set_period(cls, period):
        cls.period = period

    # 직원 수를 설정하는 클래스 메서드
    @classmethod
    def set_employees(cls, employees):
        cls.employees = employees

    # 무작위 근무 스케줄을 생성하는 클래스 메서드  
    # 각 근무 슬롯(총 period * employees * 3)마다 0 또는 1을 임의 선택
    @classmethod
    def generate_random(cls):
        return Individual([random.choice([0, 1]) for _ in range(cls.period * cls.employees * 3)])

    # 생성자: 주어진 gene_list로 개체를 생성하고 fitness를 계산  
    def __init__(self, gene_list) -> None:
        self.gene_list = gene_list
        # 생성된 스케줄 DataFrame을 fitness 함수에 전달하여 fitness 계산
        self.fitness = self.__class__.fitness_function(self.create_schedule())

    # 근무 스케줄을 DataFrame으로 생성하는 함수  
    # 각 직원(e)별로 3 * period 길이의 연속적인 비트를 잘라서 DataFrame의 한 열로 배정
    def create_schedule(self):
        t = {}
        for e in range(1, self.employees + 1):
            shift_len = 3 * self.period
            t[e] = self.gene_list[shift_len * (e - 1): shift_len * e]
        schedule_df = pd.DataFrame(data = t)
        return schedule_df

    # 근무 스케줄을 시각화하는 함수  
    # 축 레이블(근무일 및 근무조: mor, mid, evn)을 설정한 후, imshow로 스케줄 상태를 흑백으로 표시
    def plot_schedule(self):
        schedule_df = self.create_schedule()
        x_labels = []
        shift_names = {0: 'mor', 1: 'mid', 2: 'evn'}
        # 근무 슬롯별로 day와 shift 정보를 문자열로 생성
        for i in range(0, 3 * self.period):
            day = floor(i / 3) + 1
            shift = shift_names[i % 3]
            x_labels.append(f'Day {day} : {shift}')
        plt.xticks(list(range(0, 3 * self.period)), x_labels, rotation = 90)
        y_labels = []
        for i in range(0, self.employees):
            y_labels.append(f'Emp: {i+1}')
        plt.yticks(list(range(0, self.employees)), y_labels)
        # schedule_df의 전치(transpose)를 imshow로 출력 (각 직원의 스케줄을 행으로 표시)
        plt.imshow(schedule_df.T, cmap = 'binary')
        plt.title(f'Fitness: {self.fitness}')
        plt.show()
  
if __name__ == '__main__':
    # 테스트용: 직원 5명, 7일의 스케줄 생성
    Individual.set_employees(5)
    Individual.set_period(7)

    # fitness 함수 정의: 근무 편차와 휴식 조건을 반영하여 음수 값 반환
    def fitness_function(df):
        dev = shift_deviations(df,
                               mor_min = 2, mor_max = 4,
                               day_min = 3, day_max = 5,
                               evn_min = 1, evn_max = 2
                               )
        relax = shift_relax(df, 1, 1, 3)
        return -(dev + relax)

    Individual.set_fitness_function(fitness_function)
    # 무작위 스케줄 개체 생성 후 스케줄 플롯 출력
    ind = Individual.generate_random()
    ind.plot_schedule()
