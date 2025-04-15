import random
from math import floor
import pandas as pd
import matplotlib.pyplot as plt
from schedule_analyzer import shift_deviations, shift_relax

# 스케줄 문제에서 사용될 Individual 클래스: 이진 벡터를 DataFrame 스케줄로 변환
class Individual:
    counter = 0
    period = 0         # 근무 기간 (일수)
    employees = 0      # 직원 수

    @classmethod
    def set_fitness_function(cls, fun):
        cls.fitness_function = fun

    @classmethod
    def set_period(cls, period):
        cls.period = period

    @classmethod
    def set_employees(cls, employees):
        cls.employees = employees

    # 무작위 스케줄(이진 벡터) 생성: period * employees * 3 (3회 근무조)
    @classmethod
    def generate_random(cls):
        return Individual([random.choice([0, 1]) for _ in range(cls.period * cls.employees * 3)])

    def __init__(self, gene_list) -> None:
        self.gene_list = gene_list
        # 이진 벡터를 DataFrame 스케줄로 변환한 후 적합도 함수 적용
        self.fitness = self.__class__.fitness_function(self.create_schedule())
        self.__class__.counter += 1

    # 이진 벡터를 스케줄 DataFrame으로 변환
    def create_schedule(self):
        t = {}
        for e in range(1, self.employees + 1):
            shift_len = 3 * self.period
            t[e] = self.gene_list[shift_len * (e - 1): shift_len * e]
        schedule_df = pd.DataFrame(data=t)
        return schedule_df

    # 스케줄 정보를 플롯으로 시각화
    def plot_schedule(self):
        schedule_df = self.create_schedule()
        x_labels = []
        shift_names = {0: 'mor', 1: 'mid', 2: 'evn'}
        for i in range(0, 3 * self.period):
            day = floor(i / 3) + 1
            shift = shift_names[i % 3]
            x_labels.append(f'Day {day} : {shift}')
        plt.xticks(list(range(0, 3 * self.period)), x_labels, rotation=90)
        y_labels = []
        for i in range(0, self.employees):
            y_labels.append(f'Emp: {i+1}')
        plt.yticks(list(range(0, self.employees)), y_labels)
        plt.imshow(schedule_df.T, cmap='binary')
        plt.title(f'Fitness: {self.fitness}')
        plt.show()

if __name__ == '__main__':
    Individual.set_employees(5)
    Individual.set_period(7)
    # 간단한 예시 fitness_function (shift_deviations, shift_relax 활용 가능)
    def fitness_function(df):
        dev = shift_deviations(df,
                               mor_min=2, mor_max=4,
                               day_min=3, day_max=5,
                               evn_min=1, evn_max=2)
        relax = shift_relax(df, 1, 1, 3)
        return -(dev + relax)
    Individual.set_fitness_function(fitness_function)
    ind = Individual.generate_random()
    ind.plot_schedule()
