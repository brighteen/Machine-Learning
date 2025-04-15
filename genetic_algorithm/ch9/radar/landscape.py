import copy
import random
from enum import Enum, auto
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import matplotlib.patches as mpatches
from individual import Individual

# SquareType: 지형 종류를 열거형으로 정의 (물, 땅, 언덕, 도시)
class SquareType(Enum):
    water = auto()
    land = auto()
    hill = auto()
    city = auto()

# 각 칸(Square)을 나타내는 클래스: 타입, 커버 필요 여부, 레이다 설치 비용 등
class Square:
    def __init__(self, type, needs_coverage, tower_cost, is_covered=False):
        self.type = type
        self.needs_coverage = needs_coverage
        self.is_covered = is_covered
        self.tower_cost = tower_cost
        self.has_radar = False
    def __repr__(self) -> str:
        return self.type.name()

# 전체 지형(Landscape)을 나타내는 클래스
class Landscape:
    def __init__(self, matrix):
        self.matrix = matrix
    def rows(self):
        return len(self.matrix)
    def cols(self):
        return len(self.matrix[0])
    # 주어진 좌표에 레이다 배치 후, 해당 반경 내 모든 칸을 커버 처리
    def add_radars(self, coordinates, radius):
        for i in range(self.rows()):
            for j in range(self.cols()):
                if coordinates[i][j] == 1:
                    self.matrix[i][j].has_radar = True
                    for i1 in range(self.rows()):
                        for j1 in range(self.cols()):
                            if (i1 - i)**2 + (j1 - j)**2 <= radius**2:
                                self.matrix[i1][j1].is_covered = True
    # 커버되지 않은 칸 수 계산 (needs_coverage가 True인 칸 중 is_covered가 False)
    def uncovered_count(self):
        count = 0
        for i in range(self.rows()):
            for j in range(self.cols()):
                sqr = self.matrix[i][j]
                if sqr.needs_coverage and not sqr.is_covered:
                    count += 1
        return count
    # 설치된 레이다의 총 비용 계산 (각 Square의 tower_cost 합산)
    def radar_cost(self):
        cost = 0
        for i in range(self.rows()):
            for j in range(self.cols()):
                if self.matrix[i][j].has_radar:
                    cost += self.matrix[i][j].tower_cost
        return cost

# 지형 시각화 함수: 각 SquareType에 따라 색상 설정하여 출력
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
    plt.imshow(m, cmap=cmap, norm=norm)
    plt.grid(which='major', axis='both', linestyle='--', color='k', linewidth=1)
    patches = [mpatches.Patch(color=col_list[i], label=labels[i]) for i in range(len(col_list))]
    plt.legend(handles=patches, loc=4, borderaxespad=0.)
    plt.title('Landscape')
    plt.show()

# 레이다 설치 후 커버리지 상태를 시각화하는 함수
def plot_coverage(landscape, title="Coverage"):
    coverage_colors = {
        'neutral':         1,
        'is covered':      11,
        'needs coverage':  21
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
    plt.imshow(m, cmap=cmap, norm=norm)
    plt.grid(which='major', axis='both', linestyle='--', color='k', linewidth=1)
    patches = [mpatches.Patch(color=col_list[i], label=labels[i]) for i in range(len(col_list))]
    plt.legend(handles=patches, loc=4, borderaxespad=0.)
    plt.title(title)
    plt.show()

# 각 칸의 설치 비용을 시각화
def plot_costs(landscape):
    m = np.empty([landscape.rows(), landscape.cols()])
    for i in range(landscape.rows()):
        for j in range(landscape.cols()):
            m[i, j] = landscape.matrix[i][j].tower_cost
    plt.imshow(m, cmap=cm.Reds)
    plt.colorbar()
    plt.title('Radar Construction Costs')
    plt.show()

# 랜덤 지형 생성 함수: 지정된 Square 유형과 가중치를 사용하여 rows x cols 크기의 지도 생성
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
        Square(SquareType.water, needs_coverage=False, tower_cost=500): 20,
        Square(SquareType.land, needs_coverage=False, tower_cost=30):   100,
        Square(SquareType.hill, needs_coverage=False, tower_cost=100):   8,
        Square(SquareType.city, needs_coverage=True, tower_cost=200):    1
    }
    landscape = generate_random_landscape(list(square_grid.keys()), square_grid, rows, cols)
    plot_landscape(landscape)
    plot_costs(landscape)
    plot_coverage(landscape)
    # 테스트: 임의의 개체 생성 후 레이다 배치 및 커버리지 플롯
    def fintess_function(coords):
        return 0
    Individual.set_fitness_function(fintess_function)
    Individual.rows = rows
    Individual.cols = cols
    ind = Individual.generate_random(0.0005)
    test_landscape = copy.deepcopy(landscape)
    test_landscape.add_radars(ind.get_coordinates(), 7)
    plot_coverage(test_landscape)
    radars = ind.count_radars()
    uncovered = test_landscape.uncovered_count()
    print(f'Radars: {radars}')
    print(f'Uncovered Squares: {uncovered}')
