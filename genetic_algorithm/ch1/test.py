import random

# a = random.random() # 0~1 사이 난수
a = random.gauss(0, 1) # 평균 0, 표준편차 1인 가우시안 난수(-1~1 사이 값)
print(a)