import random

# 레이다 배치 문제용 개체 클래스
class Individual:
    counter = 0
    rows = 0
    cols = 0

    @classmethod
    def set_fitness_function(cls, fun):
        cls.fitness_function = fun

    # 랜드스케이프(지도) 행과 열 수 설정
    @classmethod
    def generate_random(cls, radar_prob):
        # 전체 grid 크기 만큼 0으로 초기화한 후, radar_prob 확률로 1(레이다 설치) 선택
        gene_list = [0] * cls.rows * cls.cols
        for i in range(cls.rows * cls.cols):
            if random.random() < radar_prob:
                gene_list[i] = 1
        return Individual(gene_list)

    def __init__(self, gene_list) -> None:
        self.gene_list = gene_list
        # get_coordinates()로 지도 형태로 변환한 후 적합도 계산
        self.fitness = self.__class__.fitness_function(self.get_coordinates())
        self.__class__.counter += 1

    # 1차원 벡터를 2차원 matrix(지도의 행렬)로 변환
    def get_coordinates(self):
        r = self.__class__.rows
        c = self.__class__.cols
        matrix = [[None] * c for _ in range(r)]
        for i in range(r):
            for j in range(c):
                matrix[i][j] = self.gene_list[i * r + j]
        return matrix

    # 설치한 레이다 총 개수 반환
    def count_radars(self):
        return sum(self.gene_list)

if __name__ == '__main__':
    Individual.rows = 50
    Individual.cols = 50
    # 예시용 fintess_function (아직 정의되지 않음)
    def fintess_function(coords):
        return 0
    Individual.set_fitness_function(fintess_function)
    # 랜덤 개체 생성 테스트
    ind = Individual.generate_random(0.01)
