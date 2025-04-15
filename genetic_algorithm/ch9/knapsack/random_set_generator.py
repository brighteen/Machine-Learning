# random_set_generator.py
import random
import matplotlib.pyplot as plt

from individual import Item

# 랜덤 아이템 집합 생성 함수
# min_price, max_price: 가격 범위, min_weight, max_weight: 무게 범위, total_number: 생성할 아이템 수
def random_set_generator(min_price, max_price, min_weight, max_weight, total_number):
    l = []
    for i in range(total_number):
        # 각 아이템은 랜덤한 무게와 가격을 가지며 이름은 'Item#i' 형식
        l.append(Item(f'Item#{i}', random.uniform(min_weight, max_weight), random.uniform(min_price, max_price)))
    return l

if __name__ == '__main__':
    random.seed(15)
    items = random_set_generator(1, 100, 0.1, 7, 200)
    # 생성된 아이템들의 무게와 가격을 산점도로 시각화
    plt.scatter([i.weight for i in items], [i.price for i in items])
    plt.xlabel('weight')
    plt.ylabel('price')
    plt.show()
