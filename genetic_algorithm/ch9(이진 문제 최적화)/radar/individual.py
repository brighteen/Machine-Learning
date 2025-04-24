import random

class Individual:
    counter = 0     # 생성된 개체 수를 카운트하는 클래스 변수
    rows = 0        # 지형 행(row) 수 (외부에서 설정)
    cols = 0        # 지형 열(col) 수 (외부에서 설정)

    # fitness 함수를 설정하는 클래스 메서드
    @classmethod
    def set_fitness_function(cls, fun):
        cls.fitness_function = fun

    # 랜덤 개체 생성 메서드:
    # radar_prob 확률에 따라 각 셀에 레이더(1)를 배치하는 gene_list를 생성
    @classmethod
    def generate_random(cls, radar_prob):
        gene_list = [0] * cls.rows * cls.cols
        for i in range(cls.rows * cls.cols):
            if random.random() < radar_prob:
                gene_list[i] = 1
        return Individual(gene_list)

    # 생성자: 주어진 gene_list로 개체 생성, fitness 계산 및 개체 카운트 증가
    def __init__(self, gene_list) -> None:
        self.gene_list = gene_list
        # gene_list를 좌표 형식으로 변환하여 fitness 함수에 전달
        self.fitness = self.__class__.fitness_function(self.get_coordinates())
        self.__class__.counter += 1

    # gene_list를 행렬(2차원 리스트)로 변환하여 좌표 정보를 반환
    def get_coordinates(self):
        r = self.__class__.rows
        c = self.__class__.cols
        matrix = [[None] * c for _ in range(r)]
        for i in range(r):
            for j in range(c):
                # gene_list는 1차원 리스트이며, 각 셀에 해당하는 인덱스는 i * r + j
                matrix[i][j] = self.gene_list[i * r + j]
        return matrix

    # 레이더(1) 개수를 세서 반환
    def count_radars(self):
        return sum(self.gene_list)

if __name__ == '__main__':
    # 테스트용 설정: 지형의 크기를 50x50으로 설정
    Individual.rows = 50
    Individual.cols = 50

    # 임시 fitness 함수 (아직 정의되지 않은 경우 0 반환)
    def fintess_function(coords):
        return 0

    Individual.set_fitness_function(fintess_function)
    # 낮은 확률로 레이더를 배치한 랜덤 개체 생성
    ind = Individual.generate_random(.01)
