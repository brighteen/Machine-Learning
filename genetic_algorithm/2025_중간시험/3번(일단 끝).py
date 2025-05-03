import random
import copy
import matplotlib.pyplot as plt

# 1. 노드 및 간선 정보 정의
nodes = {'S', 'A', 'B', 'C', 'D', 'E', 'T'}
edges = {
    ('S', 'A'): 4, ('A', 'S'): 4,
    ('S', 'B'): 3, ('B', 'S'): 3,
    ('S', 'C'): 5, ('C', 'S'): 5,
    ('A', 'C'): 3, ('C', 'A'): 3,
    ('A', 'D'): 5, ('D', 'A'): 5,
    ('B', 'C'): 1, ('C', 'B'): 1,
    ('B', 'E'): 4, ('E', 'B'): 4,
    ('C', 'D'): 3, ('D', 'C'): 3,
    ('C', 'E'): 3, ('E', 'C'): 3,
    ('D', 'E'): 2, ('E', 'D'): 2,
    ('D', 'T'): 5, ('T', 'D'): 5,
    ('E', 'T'): 6, ('T', 'E'): 6
}

# 노드 좌표를 딕셔너리로 정의 (시각화를 위해)
node_coords = {
    'S': (0, 0),
    'A': (2, -2),
    'B': (2, 2),
    'C': (4, 0),
    'D': (6, -2),
    'E': (6, 2),
    'T': (8, 0)
}

# 2. 유전 알고리즘 파라미터 설정
POPULATION_SIZE = 100
MAX_GENERATIONS = 100
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.1
ELITE_SIZE = 10
INVALID_PATH_PENALTY = 1000000


# 3. Individual 클래스 및 관련 함수 정의
class Individual:
    def __init__(self, path):
        self.path = path
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        total_distance = 0
        full_path = ['S'] + self.path + ['S']

        # 매우 큰 페널티 값. 어떤 유효한 경로의 총 거리보다 커야 함.
        # INVALID_PATH_PENALTY = 1000000

        for i in range(len(full_path) - 1):
            u, v = full_path[i], full_path[i+1]
            # 해당 간선 (u, v)가 edges 딕셔너리에 존재하는지 확인
            if (u, v) not in edges:
                # 존재하지 않으면 이 경로는 유효하지 않으므로 큰 페널티를 부여
                # 최소화 문제(거리) -> 최대화 문제(적합도) 변환이므로,
                # 거리가 매우 커지도록 (음수 적합도가 매우 작아지도록) 설정
                return -INVALID_PATH_PENALTY

            total_distance += edges[(u, v)]

        # 유효한 경로인 경우 정상적으로 적합도 계산
        return -total_distance # 최소화 문제이므로 음수 반환

def create_initial_population(node_set, population_size):
    population = []
    other_nodes = list(node_set - {'S'})  # 'S'를 제외한 나머지 노드
    for _ in range(population_size):
        path = random.sample(other_nodes, len(other_nodes)) # 'S'를 제외한 노드의 순열
        population.append(Individual(list(path)))
    return population

# 4. 유전 연산 함수 정의
def selection(population, elite_size):
    # 적합도(fitness, 음수 거리) 내림차순 정렬 -> 거리는 오름차순
    sorted_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    # 엘리트 선택 + 나머지 랜덤 선택 (정렬된 리스트에서)
    return sorted_population[:elite_size] + random.choices(sorted_population, k=len(population) - elite_size)

def crossover(parent1, parent2):
    # Ordered Crossover (OX1) 변형
    p1_path = parent1.path
    p2_path = parent2.path
    length = len(p1_path)
    # 교차 지점 두 개 랜덤 선택
    start, end = sorted(random.sample(range(length), 2))

    # 자식 경로 생성 및 부모1의 중간 부분 복사
    child_path = [None] * length
    child_path[start:end+1] = p1_path[start:end+1]

    # 부모2의 노드들을 순서대로 순회하며 자식 경로의 빈 칸 채우기
    remaining_nodes_in_p2_order = [node for node in p2_path if node not in p1_path[start:end+1]]
    
    current_remaining_index = 0
    for i in range(length):
        if child_path[i] is None:
            child_path[i] = remaining_nodes_in_p2_order[current_remaining_index]
            current_remaining_index += 1
            
    return Individual(child_path) # 새로운 Individual 객체 반환

def mutate(individual):
    # Swap mutation (두 노드의 위치를 바꿈)
    mutated_path = individual.path[:] # 경로 복사
    if len(mutated_path) > 1: # 노드가 2개 이상일 때만 돌연변이 가능
        idx1, idx2 = random.sample(range(len(mutated_path)), 2)
        mutated_path[idx1], mutated_path[idx2] = mutated_path[idx2], mutated_path[idx1]
    return Individual(mutated_path) # 새로운 Individual 객체 반환


# 5. 경로 유효성 검사 및 거리 계산 함수 정의 (GA 로직에서는 사용되지 않음, 디버깅/확인용)
# calculate_total_distance 함수는 Individual 클래스의 calculate_fitness와 동일한 역할을 하므로 중복입니다.
# is_valid_path 함수는 모든 노드를 방문하고 모든 간선이 존재하는지 확인하지만, 
# 현재 GA는 permutation 기반이라 모든 노드 방문은 보장되나, 간선 존재 여부는 calculate_fitness에서 체크합니다.

# 6. 결과 시각화 함수 정의
def visualize_route(best_path):
    full_path = ['S'] + best_path + ['S']
    # 노드 좌표는 그대로 사용

    plt.figure(figsize=(8, 6))
    
    # 노드 위치에 점과 이름 표시
    for node, (nx, ny) in node_coords.items():
        plt.plot(nx, ny, 'o', markersize=10, color='red', zorder=5) # 노드 점 표시 (화살표보다 위에 오도록 zorder 설정)
        plt.text(nx, ny, node, fontsize=12, ha='right') # 노드 이름 표시

    # 최적 경로의 각 간선에 화살표 그리기
    for i in range(len(full_path) - 1):
        start_node = full_path[i]
        end_node = full_path[i+1]
        (x1, y1) = node_coords[start_node]
        (x2, y2) = node_coords[end_node]

        # annotate 함수를 사용하여 화살표 그리기
        plt.annotate(
            '', xy=(x2, y2), xycoords='data',
            xytext=(x1, y1), textcoords='data',
            arrowprops=dict(
                arrowstyle='->',    # 화살표 스타일 지정
                color='blue',       # 화살표 색상
                lw=2,               # 선 굵기
                shrinkA=12,         # 시작점에서 화살표를 약간 줄여 노드와 겹치지 않게 함
                shrinkB=12,         # 끝점에서 화살표를 약간 줄여 노드와 겹치지 않게 함
                connectionstyle='arc3,rad=0' # 직선 화살표
            )
        )

        # (선택 사항) 간선 위에 거리 표시 - 그래프가 복잡해질 수 있습니다.
        # mid_x = (x1 + x2) / 2
        # mid_y = (y1 + y2) / 2
        # distance = edges.get((start_node, end_node), '?') # 간선 거리를 가져옴
        # plt.text(mid_x, mid_y, str(distance), fontsize=8, color='gray', ha='center', va='bottom')


    plt.title('Best Route Found by GA (with direction)')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.axis('equal') # x, y 축 스케일을 같게 하여 실제 그래프 모양에 가깝게 표시
    # plt.legend() # 각 경로 세그먼트가 독립적이므로 범례는 불필요
    plt.show()

# 7. 메인 함수
def main():
    population = create_initial_population(nodes, POPULATION_SIZE)
    best_individual = None # 전체 세대 중 가장 좋은 개체
    best_fitness_history = [] # 각 세대별 최고 적합도 (거리) 기록
    avg_fitness_history = []  # 각 세대별 평균 적합도 (거리) 기록

    print("Starting Genetic Algorithm...")

    for generation in range(MAX_GENERATIONS):
        # 적합도 평가는 Individual 객체 생성 시 수행됨.
        # 여기서는 단순히 현재 population의 적합도를 사용.

        # 최고 개체 갱신 (현재 세대에서 가장 좋은 개체 찾기)
        best_of_generation = max(population, key=lambda ind: ind.fitness)

        # 전체 세대를 통틀어 가장 좋은 개체 업데이트
        # 페널티 값(-1000000)을 가진 개체는 best_individual이 되지 않도록 처리
        if best_of_generation.fitness > -INVALID_PATH_PENALTY: # 유효한 경로만 고려
             if best_individual is None or best_of_generation.fitness > best_individual.fitness:
                best_individual = copy.deepcopy(best_of_generation)

        # 통계 기록 (유효한 경로의 적합도만 평균에 포함시키는 것도 고려 가능)
        # 여기서는 모든 개체의 적합도를 평균냄 (페널티 값 포함)
        valid_fitnesses = [ind.fitness for ind in population if ind.fitness > -INVALID_PATH_PENALTY]
        if valid_fitnesses: # 유효한 경로가 하나라도 있는 경우
             avg_fitness = sum(valid_fitnesses) / len(valid_fitnesses)
        else: # 모든 경로가 유효하지 않은 경우
             avg_fitness = -INVALID_PATH_PENALTY # 또는 0이나 다른 의미있는 값

        # 기록 시에는 거리로 변환 (- 적합도)
        best_fitness_history.append(-best_individual.fitness if best_individual and best_individual.fitness > -INVALID_PATH_PENALTY else abs(INVALID_PATH_PENALTY)) # 유효하지 않으면 페널티 값 표시
        avg_fitness_history.append(-avg_fitness)

        print(f"Generation {generation + 1}/{MAX_GENERATIONS}, Best Distance: {best_fitness_history[-1]:.2f}, Avg Valid Distance: {avg_fitness_history[-1]:.2f}")

        # ----- 다음 세대 생성 -----

        # 선택: 다음 세대 생성을 위한 부모 선택
        parents = selection(population, ELITE_SIZE)

        # 교차: 선택된 부모로부터 자손 생성
        offsprings = []
        # parents 리스트의 길이가 홀수일 경우 마지막 한 명은 교차에 참여 못함 (현재 POPULATION_SIZE=100으로 짝수라 문제 없음)
        for i in range(0, len(parents) - 1, 2): 
            parent1 = parents[i]
            parent2 = parents[i+1]
            if random.random() < CROSSOVER_PROBABILITY:
                # 두 자식 생성
                offsprings.append(crossover(parent1, parent2))
                offsprings.append(crossover(parent2, parent1)) # 부모 순서 바꿔서 한 번 더
            else:
                # 교차하지 않으면 부모 그대로 자손으로
                offsprings.append(parent1)
                offsprings.append(parent2)

        # 돌연변이: 생성된 자손들에게 돌연변이 적용
        mutated_offsprings = []
        for offspring in offsprings:
            if random.random() < MUTATION_PROBABILITY:
                mutated_offsprings.append(mutate(offspring))
            else:
                mutated_offsprings.append(offspring)
                
        # 자손의 수가 POPULATION_SIZE와 다를 경우 보정 (엘리트 개체 수 고려 등)
        # 간단하게는 자손들로 다음 세대를 구성 (세대 교체 방식)
        # 엘리트 개체를 다음 세대에 보존하는 방식도 흔함
        # 현재 코드는 자손만으로 다음 세대 구성 (엘리트 보존은 selection 단계에서 간접적으로 이루어짐)
        # population = mutated_offsprings[:POPULATION_SIZE] # 개체 수가 넘칠 경우 자름

        # 여기서는 selection에서 POPULATION_SIZE와 동일한 수의 부모를 뽑고
        # 교차/돌연변이로 동일한 수의 자손을 만드는 방식으로 구현된 것으로 보임.
        # 따라서 그대로 다음 세대로 넘겨도 됩니다.
        population = mutated_offsprings


    print("\n--- Optimization Complete ---")
    if best_individual and best_individual.fitness > -INVALID_PATH_PENALTY:
        print("Best Path Found:", ['S'] + best_individual.path + ['S'])
        print("Best Distance:", -best_individual.fitness)
        # 최적 경로 시각화
        visualize_route(best_individual.path)
    else:
        print("Could not find a valid path within the given generations.")
        print("Best fitness found (may be invalid):", best_individual.fitness if best_individual else "None")


    # 적합도 변화 그래프 (거리 기준)
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_history, label='Best Valid Distance', color='blue')
    plt.plot(avg_fitness_history, label='Average Valid Distance', color='green', linestyle='--')
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    plt.title('Optimization Progress: Distance over Generations')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()