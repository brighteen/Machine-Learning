import random
from itertools import compress

# 아이템을 표현하는 클래스
class Item:
    def __init__(self, name, weight, price) -> None:
        self.name = name        # 아이템 이름
        self.weight = weight    # 아이템 무게
        self.price = price      # 아이템 가격

# 개체(해)를 표현하는 클래스: 배낭 문제에서 각 개체는 아이템 선택 여부를 이진 벡터로 가짐
class Individual:
    counter = 0  # 생성된 개체 총 수
    
    @classmethod
    def set_items(cls, items):
        cls.items = items
    
    @classmethod
    def set_max_weight(cls, max_weight):
        cls.max_weight = max_weight
    
    # 난수 기반으로 개체를 생성 (각 아이템을 0 또는 1로 임의 선택)
    @classmethod
    def create_random(cls):
        return Individual([random.choice([0, 1]) for _ in range(len(cls.items))])
    
    def __init__(self, gene_list) -> None:
        self.gene_list = gene_list   # 개체의 유전자 배열 (0, 1의 리스트)
        self.fitness = self.fitness_function()  # 적합도 계산
        self.__class__.counter += 1  # 개체 생성 시 카운터 증가
    
    # 선택된 아이템들의 총 가격을 합산
    def total_price(self):
        return sum([i.price for i in list(compress(self.__class__.items, self.gene_list))])
    
    # 선택된 아이템들의 총 무게를 합산
    def total_weight(self):
        return sum([i.weight for i in list(compress(self.__class__.items, self.gene_list))])
    
    # 적합도 함수: 무게 초과하면 0, 그렇지 않으면 총 가격 반환
    def fitness_function(self):
        if self.total_weight() > self.__class__.max_weight:
            return 0
        else:
            return self.total_price()
    
    def __str__(self):
        return f'gene: {self.gene_list}, price: {self.total_price()}, weight: {self.total_weight()}'
    
    # 개체의 정보를 출력(포함된 아이템, 적합도, 총 가격, 총 무게)
    def plot_info(self):
        print(f'Included: {[i.name for i in list(compress(self.__class__.items, self.gene_list))]}')
        print(f'Fitness: {self.fitness}')
        print(f'Price: {self.total_price()}')
        print(f'Weight: {self.total_weight()}')

if __name__ == '__main__':
    random.seed(13)
    # 예시 아이템 목록
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
    ind.plot_info()
