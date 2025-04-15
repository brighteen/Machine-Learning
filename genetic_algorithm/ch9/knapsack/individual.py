# individual.py
import random
from itertools import compress

# Item 클래스: 개별 아이템의 이름, 무게, 가격 정보를 저장
class Item:
    def __init__(self, name, weight, price) -> None:
        self.name = name
        self.weight = weight
        self.price = price

# Individual 클래스: 배낭 문제의 해(solution)를 표현하는 클래스
class Individual:
    counter = 0  # 생성된 개체 수를 기록하는 클래스 변수

    @classmethod
    def set_items(cls, items):
        cls.items = items

    @classmethod
    def set_max_weight(cls, max_weight):
        cls.max_weight = max_weight

    # 랜덤 개체 생성: 아이템 수 만큼 0과 1을 랜덤 선택하여 gene list 구성
    @classmethod
    def create_random(cls):
        return Individual([random.choice([0, 1]) for _ in range(len(cls.items))])

    def __init__(self, gene_list) -> None:
        self.gene_list = gene_list
        self.fitness = self.fitness_function()  # 초기 fitness 계산
        self.__class__.counter += 1  # 생성 시마다 카운터 증가

    # 선택된 아이템의 총 가격 계산 (gene_list가 1이면 해당 아이템 포함)
    def total_price(self):
        return sum([i.price for i in list(compress(self.__class__.items, self.gene_list))])

    # 선택된 아이템의 총 무게 계산
    def total_weight(self):
        return sum([i.weight for i in list(compress(self.__class__.items, self.gene_list))])

    # fitness 함수: 배낭의 최대 무게를 초과하면 0, 그렇지 않으면 총 가격 반환
    def fitness_function(self):
        if self.total_weight() > self.__class__.max_weight:
            return 0
        else:
            return self.total_price()

    # 문자열 표현: gene list, 총 가격, 총 무게 정보를 반환
    def __str__(self):
        return f'gene: {self.gene_list}, price: {self.total_price()}, weight: {self.total_weight()}'

    # 개체의 정보를 출력하는 함수 (포함된 아이템, fitness, 가격, 무게)
    def plot_info(self):
        print(f'Included: {[i.name for i in list(compress(self.__class__.items, self.gene_list))]}')
        print(f'Fitness: {self.fitness}')
        print(f'Price: {self.total_price()}')
        print(f'Weight: {self.total_weight()}')

if __name__ == '__main__':
    # 테스트용: 시드 설정 후 사전에 정의된 아이템 목록으로 랜덤 개체 생성
    random.seed(13)
    items = [
        Item('laptop', 3, 300),
        Item('book', 2, 15),
        Item('radio', 1, 30),
        Item('tv', 6, 230),
        Item('potato', 5, 7),
        Item('brick', 3, 1),
        Item('bottle', 1, 2),
        Item('camera', 0.5, 280),
        Item('smartphone', 0.1, 500),
        Item('picture', 1, 170),
        Item('flower', 2, 5),
        Item('chair', 3, 4),
        Item('watch', 0.05, 500),
        Item('boots', 1.5, 30),
        Item('radiator', 5, 25),
        Item('tablet', 0.5, 450),
        Item('printer', 4.5, 170)
    ]

    Individual.set_items(items)
    Individual.set_max_weight(10)

    ind = Individual.create_random()
    # 개체 정보를 출력
    ind.plot_info()
