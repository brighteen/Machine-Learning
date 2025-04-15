import random
import matplotlib.pyplot as plt
from individual import Item

# 주어진 범위 내에서 무작위 아이템 집합 생성
# min_price, max_price: 아이템 가격 범위, min_weight: 최소 무게, max_weight: 최대 무게,
# total_number: 생성할 아이템 총 개수
def random_set_generator(min_price, max_price, min_weight, max_weight, total_number):
    l = []
    for i in range(total_number):
        l.append(Item(f'Item#{i}', random.uniform(min_weight, max_weight), random.uniform(min_price, max_price)))
    return l

if __name__ == '__main__':
    random.seed(15)
    items = random_set_generator(1, 100, 0.1, 7, 200)
    # 산점도를 통해 아이템의 무게와 가격 분포 시각화
    plt.scatter([i.weight for i in items], [i.price for i in items])
    plt.xlabel('weight')
    plt.ylabel('price')
    plt.show()
