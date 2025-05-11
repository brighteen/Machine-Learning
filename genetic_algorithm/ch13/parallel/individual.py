import random  # 난수 생성을 위한 라이브러리


class Individual:  # 레이더 배치 문제를 위한 개체 클래스
    rows = 0  # 격자 행 수
    cols = 0  # 격자 열 수

    @classmethod
    def set_fitness_function(cls, fun):  # 적합도 함수 설정 클래스 메서드
        cls.fitness_function = fun  # 클래스 변수에 적합도 함수 할당

    @classmethod
    def generate_random(cls, radar_prob):  # 무작위 개체 생성 클래스 메서드
        gene_list = [0] * cls.rows * cls.cols  # 모든 위치를 0으로 초기화
        for i in range(cls.rows * cls.cols):  # 모든 격자 위치에 대해
            if random.random() < radar_prob:  # 주어진 확률에 따라
                gene_list[i] = 1  # 레이더 배치 (1로 설정)
        return Individual(gene_list)  # 생성된 유전자로 개체 반환

    def __init__(self, gene_list) -> None:  # 생성자 메서드
        self.gene_list = gene_list  # 유전자 리스트 저장
        self.fitness = self.__class__.fitness_function(self.get_coordinates())  # 2D 좌표로 변환하여 적합도 계산

    def get_coordinates(self):  # 1D 유전자를 2D 좌표 행렬로 변환하는 메서드
        r = self.__class__.rows  # 행 수 가져오기
        c = self.__class__.cols  # 열 수 가져오기
        matrix = [[None] * c for _ in range(r)]  # 빈 2D 행렬 초기화
        for i in range(r):  # 각 행에 대해
            for j in range(c):  # 각 열에 대해
                matrix[i][j] = self.gene_list[i * r + j]  # 1D 인덱스를 2D 좌표로 변환하여 값 할당
        return matrix  # 2D 좌표 행렬 반환

    def count_radars(self):  # 배치된 레이더 수를 계산하는 메서드
        return sum(self.gene_list)  # 유전자 리스트의 합계 반환 (1의 개수 = 레이더 수)
