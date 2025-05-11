import copy  # 깊은 복사를 위한 라이브러리
import random  # 난수 생성을 위한 라이브러리
from enum import Enum, auto  # 열거형 정의를 위한 라이브러리

import numpy as np  # 수치 연산을 위한 라이브러리
import matplotlib.pyplot as plt  # 시각화를 위한 라이브러리
from matplotlib import colors, cm  # 색상 관련 기능
import matplotlib.patches as mpatches  # 범례 생성을 위한 패치

from individual import Individual  # Individual 클래스 가져오기


class SquareType(Enum):  # 격자 타입 정의를 위한 열거형
    water = auto()  # 물
    land = auto()   # 땅
    hill = auto()   # 언덕
    city = auto()   # 도시


class Square:  # 격자 칸 클래스

    def __init__(self, type, needs_coverage, tower_cost, is_covered = False):  # 생성자
        self.type = type  # 타입 (물, 땅, 언덕, 도시)
        self.needs_coverage = needs_coverage  # 커버리지 필요 여부
        self.is_covered = is_covered  # 커버됨 여부
        self.tower_cost = tower_cost  # 레이더 건설 비용
        self.has_radar = False  # 레이더 존재 여부

    def __repr__(self) -> str:  # 문자열 표현 메서드
        return self.type.name()  # 타입 이름으로 표현


class Landscape:  # 전체 풍경/지형 클래스

    def __init__(self, matrix):  # 생성자
        self.matrix = matrix  # 격자 행렬 저장

    def rows(self):  # 행 수 반환 메서드
        return len(self.matrix)  # 행렬의 행 개수 반환

    def cols(self):  # 열 수 반환 메서드
        return len(self.matrix[0])  # 행렬의 열 개수 반환

    def add_radars(self, coordinates, radius):  # 레이더 추가 및 커버리지 계산 메서드
        for i in range(self.rows()):  # 모든 행에 대해 반복
            for j in range(self.cols()):  # 모든 열에 대해 반복
                if coordinates[i][j] == 1:  # 좌표에 레이더가 있으면
                    self.matrix[i][j].has_radar = True  # 레이더 존재 표시
                    for i1 in range(self.rows()):  # 모든 행에 대해 다시 반복
                        for j1 in range(self.cols()):  # 모든 열에 대해 다시 반복
                            if (i1 - i)**2 + (j1 - j)**2 <= radius**2:  # 레이더 반경 내 거리면
                                self.matrix[i1][j1].is_covered = True  # 커버됨 표시

    def uncovered_count(self):  # 커버되지 않은 칸 수 계산 메서드
        count = 0  # 카운터 초기화
        for i in range(self.rows()):  # 모든 행에 대해 반복
            for j in range(self.cols()):  # 모든 열에 대해 반복
                sqr = self.matrix[i][j]  # 현재 칸 참조
                if sqr.needs_coverage and not sqr.is_covered:  # 커버리지가 필요하지만 커버되지 않은 경우
                    count += 1  # 카운터 증가
        return count  # 총 커버되지 않은 칸 수 반환

    def radar_cost(self):  # 레이더 비용 계산 메서드
        cost = 0  # 비용 초기화
        for i in range(self.rows()):  # 모든 행에 대해 반복
            for j in range(self.cols()):  # 모든 열에 대해 반복
                if self.matrix[i][j].has_radar:  # 레이더가 있는 칸이면
                    cost += self.matrix[i][j].tower_cost  # 해당 위치의 레이더 비용 추가
        return cost  # 총 레이더 비용 반환


def plot_landscape(landscape):  # 풍경 시각화 함수
    square_colors = {  # 각 타입별 색상 값 매핑
        SquareType.water: 1,
        SquareType.land:  11,
        SquareType.hill:  21,
        SquareType.city:  31
    }
    m = np.empty([landscape.rows(), landscape.cols()])  # 시각화용 빈 행렬 생성
    for i in range(landscape.rows()):  # 모든 행에 대해 반복
        for j in range(landscape.cols()):  # 모든 열에 대해 반복
            m[i, j] = square_colors[landscape.matrix[i][j].type]  # 타입에 따른 색상 값 할당
    col_list = ['blue', 'green', 'brown', 'black']  # 실제 색상 리스트
    labels = [s.name for s in square_colors.keys()]  # 범례 레이블
    cmap = colors.ListedColormap(col_list)  # 커스텀 색상맵 생성
    bounds = [0, 10, 20, 30, 40]  # 경계값 설정
    norm = colors.BoundaryNorm(bounds, cmap.N)  # 정규화 설정

    plt.imshow(m, cmap = cmap, norm = norm)  # 이미지 그리기
    plt.grid(which = 'major', axis = 'both', linestyle = '--', color = 'k', linewidth = 1)  # 그리드 추가
    patches = [mpatches.Patch(color = col_list[i], label = labels[i]) for i in range(len(col_list))]  # 범례 패치 생성
    plt.legend(handles = patches, loc = 4, borderaxespad = 0.)  # 범례 추가
    plt.title('Landscape')  # 제목 설정
    plt.show()  # 그래프 표시


def plot_coverage(landscape, title = "Coverage"):  # 커버리지 시각화 함수
    coverage_colors = {  # 커버리지 상태별 색상 값 매핑
        'neutral':        1,  # 커버리지 불필요
        'is covered':     11,  # 커버됨
        'needs coverage': 21   # 커버 필요하지만 커버되지 않음
    }

    m = np.empty([landscape.rows(), landscape.cols()])  # 시각화용 빈 행렬 생성
    for i in range(landscape.rows()):  # 모든 행에 대해 반복
        for j in range(landscape.cols()):  # 모든 열에 대해 반복
            if landscape.matrix[i][j].is_covered:  # 커버된 칸이면
                m[i, j] = coverage_colors['is covered']  # 커버됨 색상 할당
            elif not landscape.matrix[i][j].needs_coverage:  # 커버리지가 필요없는 칸이면
                m[i, j] = coverage_colors['neutral']  # 중립 색상 할당
            elif landscape.matrix[i][j].needs_coverage:  # 커버리지가 필요한 칸이면
                m[i, j] = coverage_colors['needs coverage']  # 커버 필요 색상 할당

    col_list = ['white', 'green', 'red']  # 실제 색상 리스트
    labels = list(coverage_colors.keys())  # 범례 레이블
    cmap = colors.ListedColormap(col_list)  # 커스텀 색상맵 생성
    bounds = [0, 10, 20, 30]  # 경계값 설정
    norm = colors.BoundaryNorm(bounds, cmap.N)  # 정규화 설정

    plt.imshow(m, cmap = cmap, norm = norm)  # 이미지 그리기
    plt.grid(which = 'major', axis = 'both', linestyle = '--', color = 'k', linewidth = 1)  # 그리드 추가
    patches = [mpatches.Patch(color = col_list[i], label = labels[i]) for i in range(len(col_list))]  # 범례 패치 생성
    plt.legend(handles = patches, loc = 4, borderaxespad = 0.)  # 범례 추가
    plt.title(title)  # 제목 설정
    plt.show()  # 그래프 표시


def plot_costs(landscape):  # 레이더 비용 시각화 함수
    m = np.empty([landscape.rows(), landscape.cols()])  # 시각화용 빈 행렬 생성
    for i in range(landscape.rows()):  # 모든 행에 대해 반복
        for j in range(landscape.cols()):  # 모든 열에 대해 반복
            m[i, j] = landscape.matrix[i][j].tower_cost  # 타워 비용 할당
    plt.imshow(m, cmap = cm.Reds)  # 적색 계열 색상맵으로 이미지 그리기
    plt.colorbar()  # 색상 바 추가
    plt.title('Radar Construction Costs')  # 제목 설정
    plt.show()  # 그래프 표시


def generate_random_landscape(points, weights, rows, cols):  # 무작위 풍경 생성 함수
    matrix = [[None] * cols for _ in range(rows)]  # 빈 행렬 초기화
    for i in range(rows):  # 모든 행에 대해 반복
        for j in range(cols):  # 모든 열에 대해 반복
            p = random.choices(points, weights.values())  # 가중치에 따라 지형 타입 선택
            square = copy.deepcopy(p[0])  # 선택된 지형 타입 깊은 복사
            square.tower_cost = round(square.tower_cost * (1 + random.uniform(0, .1)))  # 타워 비용에 임의의 변동 적용
            matrix[i][j] = square  # 행렬에 칸 할당
    return Landscape(matrix)  # 생성된 풍경 반환
