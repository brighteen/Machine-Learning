# 논문에는 없음

def cycle_crossover(p1, p2): # 함수생성
    length = len(p1) # p1의 요소의 개수
    c1, c2 = [None] * length, [None] * length # 자식 리스트 값 초기화 및 설정

    visited = [False] * length  # 각 인덱스의 방문기록 리스트(true 와 false 로 이루어짐.) 설정
    cycle = 0  # 사이클 순환 횟수를 저장하기 위해 지정

    while not all(visited):

        # 방문하지 않은 인덱스가 없을때까지 반복
        start = visited.index(False) # 방문하지 않은 첫번째 인덱스를 찾기
        current = start # 현재 탐색중인 인덱스

        if cycle % 2 == 0:
            # 사이클이 짝수일 때 방문한 p1의 값을 c1에, p2의 값을 c2에 복사
            while True: # 루프시작
                c1[current] = p1[current] # 현재 탐색중인 인덱스의 값을 p1 에서 c1로 복사
                c2[current] = p2[current] # 현재 탐색중인 인덱스의 값을 p2 에서 c2로 복사
                visited[current] = True # 현재 인덱스의 방문을 기록(중복탐색 방지)
                current = p1.index(p2[current]) # p2[current]의 값이 p1에서 어디 있는지 찾기
                if current == start: # 사이클이 한 바퀴를 돌면(시작지점으로 돌아오면)
                    break# 루프를 종료
        else:
            # 사이클이 홀수일 때 방문한 p1의 값을 c2에, p2의 값을 c1에 저장
            while True:# 루프시작
                c1[current] = p2[current] # 현재 탐색중인 인덱스의 값을 p2 에서 c1로 복사
                c2[current] = p1[current] # 현재 탐색중인 인덱스의 값을 p1 에서 c2로 복사
                visited[current] = True # 현재 인덱스의 방문을 기록(중복탐색 방지)
                current = p1.index(p2[current]) # p2[current]의 값이 p1에서 어디 있는지 찾기
                if current == start: # 사이클이 한 바퀴를 돌면(시작지점으로 돌아오면)
                    break # 루프를 종료

        cycle += 1 # 사이클 카운트

    return c1, c2

#부모 값 설정
p1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
p2 = [9, 3, 7, 8, 2, 6, 5, 1, 4]

offspring = cycle_crossover(p1, p2)

print(f'P1: {p1}')
print(f'P2: {p2}')
print(f'C1: {offspring[0]}')
print(f'C2: {offspring[1]}')