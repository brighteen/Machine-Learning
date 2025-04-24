import copy
import random
from enum import Enum, auto

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import matplotlib.patches as mpatches

from individual import Individual

# SquareType 열거형: 지형의 종류 정의 (물, 육지, 언덕, 도시)
class SquareType(Enum):
    water = auto()
    land = auto()
    hill = auto()
    city = auto()

# Square 클래스: 지형의 한 셀을 표현
class Square:
    def __init__(self, type, needs_coverage, tower_cost, is_covered = False):
        self.type = type                # 지형 타입 (예: water, land 등)
        self.needs_coverage = needs_coverage  # 커버리지가 필요한지 여부 (예: 도시)
        self.is_covered = is_covered    # 현재 해당 셀이 커버되었는지 여부
        self.tower_cost = tower_cost    # 해당 셀에 레이더 타워를 설치할 때의 비용
        self.has_radar = False          # 이 셀에 실제로 레이더가 설치되었는지 여부

    # __repr__ 메서드: 출력 시 Square 타입의 이름을 반환
    def __repr__(self) -> str:
        return self.type.name()

# Landscape 클래스: 전체 지형을 행렬(2차원 리스트)로 구성
class Landscape:
    def __init__(self, matrix):
        self.matrix = matrix

    # 지형의 행(row) 수 반환
    def rows(self):
        return len(self.matrix)

    # 지형의 열(col) 수 반환
    def cols(self):
        return len(self.matrix[0])

    # 주어진 좌표(레이더 배치 결과)를 바탕으로, 각 레이더가 커버하는 셀을 업데이트
    def add_radars(self, coordinates, radius):
        for i in range(self.rows()):
            for j in range(self.cols()):
                if coordinates[i][j] == 1:
                    self.matrix[i][j].has_radar = True
                    # 레이더가 설치된 셀을 중심으로 반경(radius) 내의 모든 셀을 커버 처리
                    for i1 in range(self.rows()):
                        for j1 in range(self.cols()):
                            if (i1 - i)**2 + (j1 - j)**2 <= radius**2:
                                self.matrix[i1][j1].is_covered = True

    # 커버리지가 필요한 셀 중 아직 커버되지 않은 셀의 수를 계산하여 반환
    def uncovered_count(self):
        count = 0
        for i in range(self.rows()):
            for j in range(self.cols()):
                sqr = self.matrix[i][j]
                if sqr.needs_coverage and not sqr.is_covered:
                    count += 1
        return count

    # 지형 내에 설치된 레이더의 총 건설 비용을 계산하여 반환
    def radar_cost(self):
        cost = 0
        for i in range(self.rows()):
            for j in range(self.cols()):
                if self.matrix[i][j].has_radar:
                    cost += self.matrix[i][j].tower_cost
        return cost

# 지형을 플롯하여 시각화하는 함수: 각 Square 타입별 색상을 적용하여 출력
def plot_landscape(landscape):
    square_colors = {
        SquareType.water: 1,
        SquareType.land:  11,
        SquareType.hill:  21,
        SquareType.city:  31
    }
    m = np.empty([landscape.rows(), landscape.cols()])
    for i in range(landscape.rows()):
        for j in range(landscape.cols()):
            m[i, j] = square_colors[landscape.matrix[i][j].type]
    col_list = ['blue', 'green', 'brown', 'black']
    labels = [s.name for s in square_colors.keys()]
    cmap = colors.ListedColormap(col_list)
    bounds = [0, 10, 20, 30, 40]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(m, cmap = cmap, norm = norm)
    plt.grid(which = 'major', axis = 'both', linestyle = '--', color = 'k', linewidth = 1)
    patches = [mpatches.Patch(color = col_list[i], label = labels[i]) for i in range(len(col_list))]
    plt.legend(handles = patches, loc = 4, borderaxespad = 0.)
    plt.title('Landscape')
    plt.show()

# 커버리지를 플롯하는 함수: 각 셀이 커버되었는지 여부에 따라 다른 색으로 출력
def plot_coverage(landscape, title = "Coverage"):
    coverage_colors = {
        'neutral':         1,
        'is covered':      11,
        'needs coverage': 21
    }

    m = np.empty([landscape.rows(), landscape.cols()])
    for i in range(landscape.rows()):
        for j in range(landscape.cols()):
            if landscape.matrix[i][j].is_covered:
                m[i, j] = coverage_colors['is covered']
            elif not landscape.matrix[i][j].needs_coverage:
                m[i, j] = coverage_colors['neutral']
            elif landscape.matrix[i][j].needs_coverage:
                m[i, j] = coverage_colors['needs coverage']

    col_list = ['white', 'green', 'red']
    labels = list(coverage_colors.keys())
    cmap = colors.ListedColormap(col_list)
    bounds = [0, 10, 20, 30]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(m, cmap = cmap, norm = norm)
    plt.grid(which = 'major', axis = 'both', linestyle = '--', color = 'k', linewidth = 1)
    patches = [mpatches.Patch(color = col_list[i], label = labels[i]) for i in range(len(col_list))]
    plt.legend(handles = patches, loc = 4, borderaxespad = 0.)
    plt.title(title)
    plt.show()

# 지형 상의 각 셀에 대한 레이더 건설 비용을 플롯하는 함수 (열 지도)
def plot_costs(landscape):
    m = np.empty([landscape.rows(), landscape.cols()])
    for i in range(landscape.rows()):
        for j in range(landscape.cols()):
            m[i, j] = landscape.matrix[i][j].tower_cost
    plt.imshow(m, cmap = cm.Reds)
    plt.colorbar()
    plt.title('Radar Construction Costs')
    plt.show()

# 랜덤 지형을 생성하는 함수:
# points: 사용할 Square 객체 목록, weights: 각 Square의 빈도(가중치), rows, cols: 지형 크기
def generate_random_landscape(points, weights, rows, cols):
    matrix = [[None] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            p = random.choices(points, weights.values())
            square = copy.deepcopy(p[0])
            square.tower_cost = round(square.tower_cost * (1 + random.uniform(0, .1)))
            matrix[i][j] = square
    return Landscape(matrix)

if __name__ == '__main__':

    random.seed(15)

    rows = 60
    cols = 60

    square_grid = {
        Square(SquareType.water, needs_coverage = False, tower_cost = 500): 20,
        Square(SquareType.land, needs_coverage = False, tower_cost = 30):   100,
        Square(SquareType.hill, needs_coverage = False, tower_cost = 100):  8,
        Square(SquareType.city, needs_coverage = True, tower_cost = 200):   1
    }

    # 생성된 랜덤 지형을 플롯하여 확인
    landscape = generate_random_landscape(list(square_grid.keys()), square_grid, rows, cols)
    plot_landscape(landscape)
    plot_costs(landscape)
    plot_coverage(landscape)

    # 테스트용 임시 fitness 함수: 현재는 0 반환
    def fintess_function(coords):
        return 0

    Individual.set_fitness_function(fintess_function)
    Individual.rows = rows
    Individual.cols = cols
    ind = Individual.generate_random(.0005)
    test_landscape = copy.deepcopy(landscape)
    test_landscape.add_radars(ind.get_coordinates(), 7)
    plot_coverage(test_landscape)

    radars = ind.count_radars()
    uncovered = test_landscape.uncovered_count()

    print(f'Radars: {radars}')
    print(f'Uncovered Squares: {uncovered}')
