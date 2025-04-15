import random
from math import floor

import pandas as pd
import matplotlib.pyplot as plt

  
# Individual 클래스: 근무 스케줄 해(solution)를 표현  
class Individual:
    counter = 0       # 생성된 개체의 총 개수를 기록
    period = 0        # 근무 스케줄 기간 (예: 일수)
    employees = 0     # 직원 수

    # fitness 함수를 설정하는 클래스 메서드
    @classmethod
    def set_fitness_function(cls, fun):
        cls.fitness_function = fun

    # 스케줄 기간 설정 메서드
    @classmethod
    def set_period(cls, period):
        cls.period = period

    # 직원 수 설정 메서드
    @classmethod
    def set_employees(cls, employees):
        cls.employees = employees

    # 무작위 근무 스케줄을 생성하는 메서드 (각 슬롯에 대해 0 또는 1의 선택)
    @classmethod
    def generate_random(cls):
        return Individual([random.choice([0, 1]) for _ in range(cls.period * cls.employees * 3)])

    # 생성자: 주어진 gene_list로 개체 생성 후, 스케줄을 생성하여 fitness 계산 및 개체 카운터 증가
    def __init__(self, gene_list) -> None:
        self.gene_list = gene_list
        self.fitness = self.__class__.fitness_function(self.create_schedule())
        self.__class__.counter += 1

    # gene_list를 직원별 근무 스케줄 DataFrame으로 변환  
    # 각 직원별로 3 * period 길이의 슬라이스를 추출하여 DataFrame 컬럼으로 할당
    def create_schedule(self):
        t = {}
        for e in range(1, self.employees + 1):
            shift_len = 3 * self.period
            t[e] = self.gene_list[shift_len * (e - 1): shift_len * e]
        schedule_df = pd.DataFrame(data = t)
        return schedule_df

    # 스케줄을 시각화하는 함수: 근무일 및 근무조 정보를 X축, 직원 정보를 Y축으로 설정하여 imshow 출력
    def plot_schedule(self):
        schedule_df = self.create_schedule()
        x_labels = []
        shift_names = {0: 'mor', 1: 'mid', 2: 'evn'}
        for i in range(0, 3 * self.period):
            day = floor(i / 3) + 1
            shift = shift_names[i % 3]
            x_labels.append(f'Day {day} : {shift}')
        plt.xticks(list(range(0, 3 * self.period)), x_labels, rotation = 90)
        y_labels = []
        for i in range(0, self.employees):
            y_labels.append(f'Emp: {i+1}')
        plt.yticks(list(range(0, self.employees)), y_labels)
        plt.imshow(schedule_df.T, cmap = 'binary')
        plt.title(f'Fitness: {self.fitness}')
        plt.show()
        plt.close()
  
if __name__ == '__main__':
    # 테스트용: 직원 5명, 7일의 스케줄 생성 및 플롯
    random.seed(9)
    Individual.set_employees(5)
    Individual.set_period(7)

    # 임시 fitness 함수 정의 (아직 미정의 상태 → 0 반환)
    def fitness_function(df):
        return 0

    Individual.set_fitness_function(fitness_function)
    ind = Individual.generate_random()
    ind.plot_schedule()
