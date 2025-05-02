import random
import copy
import matplotlib.pyplot as plt

# 1. 문제 정의 (노드 및 간선 정보)
nodes = {'S', 'A', 'B', 'C', 'D', 'E', 'T'}
edges = {
    ('S', 'A'): 4, ('S', 'B'): 3, ('S', 'C'): 5,
    ('A', 'C'): 3, ('A', 'D'): 5,
    ('B', 'C'): 1, ('B', 'E'): 4,
    ('C', 'D'): 3, ('C', 'E'): 3,
    ('D', 'E'): 2, ('D', 'T'): 5,
    ('E', 'T'): 6,
    ('A', 'S'): 4, ('B', 'S'): 3, ('C', 'S'): 5,
    ('C', 'A'): 3, ('D', 'A'): 5,
    ('C', 'B'): 1, ('E', 'B'): 4,
    ('D', 'C'): 3, ('E', 'C'): 3, ('E', 'D'): 2,
    ('T', 'D'): 5, ('T', 'E'): 6
}

# 2. 유전 알고리즘 파라미터
POPULATION_SIZE = 100
MAX_GENERATIONS = 100
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.1

# 3. 개체 표현 및 관련 함수
def create_individual():
    """'S'를 제외한 나머지 노드 순열 생성"""
    individual = list(nodes - {'S'})
    random.shuffle(individual)
    return ['S'] + individual + ['S']  # 'S'를 시작과 끝에 추가

def calculate_fitness(individual):
    """경로의 총 거리 계산 (적합도 함수)"""
    fitness = 0
    for i in range(len(individual) - 1):
        try:
            fitness += edges[(individual[i], individual[i+1])]
        except KeyError:  # 유효하지 않은 경로 처리
            return float('inf')  # 매우 큰 값으로 설정
    return fitness

# 4. 유전 연산
def crossover(parent1, parent2):
    """순서 기반 교차 연산 (Ordered Crossover)"""

    p1 = parent1[1:-1]  # 'S' 제거
    p2 = parent2[1:-1]
    
    size = len(p1)
    start, end = sorted([random.randrange(size) for _ in range(2)])
    
    child = ['S'] + [None] * size + ['S']
    child[start+1:end+2] = p1[start:end+1]
    
    remaining_genes = [node for node in p2 if node not in child]
    
    idx = 0
    for i in range(1, len(child)-1):
        if child[i] is None:
            child[i] = remaining_genes[idx]
            idx += 1
    
    return child

def mutate(individual):
    """Swap Mutation"""
    mutated = individual[:]
    a, b = random.sample(range(1, len(individual) - 1), 2)  # 'S' 제외
    mutated[a], mutated[b] = mutated[b], mutated[a]
    return mutated

def selection(population, fitnesses, num_parents):
    """적합도 기반 선택 (룰렛 휠 선택)"""

    total_fitness = sum(fitnesses)
    if total_fitness == 0:  # 모든 적합도가 0인 경우 방지
        probabilities = [1 / len(fitnesses)] * len(fitnesses)
    else:
        probabilities = [fitness / total_fitness for fitness in fitnesses]

    parents = []
    for _ in range(num_parents):
        r = random.random()
        cumulative_probability = 0
        for i, probability in enumerate(probabilities):
            cumulative_probability += probability
            if r <= cumulative_probability:
                parents.append(population[i])
                break
    return parents

# 5. 유전 알고리즘 실행
def genetic_algorithm():
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    best_fitness_history = []
    avg_fitness_history = []

    for generation in range(MAX_GENERATIONS):
        fitnesses = [calculate_fitness(ind) for ind in population]
        best_fitness = min(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)

        best_individual = population[fitnesses.index(min(fitnesses))]

        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Avg Fitness = {avg_fitness}")

        parents = selection(population, fitnesses, POPULATION_SIZE)

        offspring = []
        for i in range(0, POPULATION_SIZE, 2):
            if random.random() < CROSSOVER_PROBABILITY:
                child1 = crossover(parents[i % len(parents)], parents[(i + 1) % len(parents)])
                offspring.append(child1)
            else:
                offspring.append(parents[i % len(parents)])
            if i + 1 < POPULATION_SIZE:
                offspring.append(parents[(i + 1) % len(parents)])

        mutated_offspring = []
        for mutant in offspring:
            if random.random() < MUTATION_PROBABILITY:
                mutated_offspring.append(mutate(mutant))
            else:
                mutated_offspring.append(mutant)

        population = mutated_offspring

    return best_individual, best_fitness_history, avg_fitness_history

# 6. 결과 출력 및 시각화
if __name__ == "__main__":
    best_route, best_fitness_history, avg_fitness_history = genetic_algorithm()

    print("\nBest Route:", best_route)
    print("Best Distance:", calculate_fitness(best_route))

    plt.plot(best_fitness_history, label="Best Fitness")
    plt.plot(avg_fitness_history, label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.legend()
    plt.show()