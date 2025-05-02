import random
import copy

# 문제 조건
MAX_GENERATIONS = 30
POPULATION_SIZE = 10  # 부모 수 (여기서는 편의상 전체 개체 수를 부모 수로 사용)
CHROMOSOME_LENGTH = 5  # 이진수 5자리
MUTATION_RATE = 0.1
MUTATION_INDIVIDUAL_RATE = 0.2

# 제약 조건 함수
def is_feasible(x1, x2):
    if x1 < 0 or x2 < 0:
        return False
    if -x1 + x2 > 5:
        return False
    if x1 + x2 > 10:
        return False
    if -2*x1 + x2 < -10:
        return False
    return True

# 적합도 함수
def fitness_function(x1, x2):
    if is_feasible(x1, x2):
        return x1 + 2*x2
    else:
        return 0

# 이진수 -> 십진수 변환 함수
def binary_to_decimal(binary):
    return int(binary, 2)

# 개체 생성 함수
def create_individual(length):
    return [random.randint(0, 1) for _ in range(length)]

# 초기 모집단 생성 함수
def create_population(size, length):
    return [create_individual(length) for _ in range(size)]

# 선택 함수 (토너먼트 선택)
def selection(population, fitnesses, num_parents):
    parents = []
    for _ in range(num_parents):
        tournament_indices = random.sample(range(len(population)), 3)  # 토너먼트 크기 = 3
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
        parents.append(population[winner_index])
    return parents

def crossover(p1, p2):
    point = random.randint(1, len(p1) - 1)
    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
    c1[point:], c2[point:] = p2[point:], p1[point:]
    return c1, c2

# 돌연변이 연산 (bit-flip)
def mutation_bit_flip(ind):
    mut = copy.deepcopy(ind)
    pos = random.randint(0, len(ind)-1)
    mut[pos] = (mut[pos] + 1) % 2
    return mut

# 유전 알고리즘 메인 함수
def genetic_algorithm(random_seed):
    random.seed(random_seed)  # 시드 값 설정
    population = create_population(POPULATION_SIZE, CHROMOSOME_LENGTH * 2)  # x1, x2 모두 표현하므로 *2
    best_fitness = 0
    best_individual = None
    final_fitnesses = []  # 최종 세대 적합도 분포 저장
    final_x1 = 0
    final_x2 = 0

    for generation in range(MAX_GENERATIONS):
        # 이진수 -> 십진수 변환
        x1_binaries = [ "".join(map(str,ind[:CHROMOSOME_LENGTH])) for ind in population]
        x2_binaries = [ "".join(map(str,ind[CHROMOSOME_LENGTH:])) for ind in population]

        x1_decimals = [binary_to_decimal(binary) for binary in x1_binaries]
        x2_decimals = [binary_to_decimal(binary) for binary in x2_binaries]

        # 적합도 계산
        fitnesses = [fitness_function(x1, x2) for x1, x2 in zip(x1_decimals, x2_decimals)]
        final_fitnesses = fitnesses  # 최종 세대 적합도 분포 저장

        # 현재 세대 최고 적합도 갱신
        if max(fitnesses) > best_fitness:
            best_fitness = max(fitnesses)
            best_individual = population[fitnesses.index(max(fitnesses))]
            final_x1 = x1_decimals[fitnesses.index(max(fitnesses))]
            final_x2 = x2_decimals[fitnesses.index(max(fitnesses))]

        # 선택
        parents = selection(population, fitnesses, POPULATION_SIZE)

        # 교차
        offsprings = []
        for i in range(0, POPULATION_SIZE, 2):
            c1, c2 = crossover(parents[i % len(parents)], parents[(i + 1) % len(parents)])
            offsprings.append(c1)
            offsprings.append(c2)

        # 돌연변이
        num_mutation_individuals = int(POPULATION_SIZE * MUTATION_INDIVIDUAL_RATE)
        mutation_targets = random.sample(range(len(offsprings)), num_mutation_individuals)

        for i in mutation_targets:
            if random.random() < MUTATION_RATE:
                offsprings[i] = mutation_bit_flip(offsprings[i])

        population = offsprings

    # 최종 결과 해석
    if best_individual is not None:
        x1_final = binary_to_decimal("".join(map(str,best_individual[:CHROMOSOME_LENGTH])))
        x2_final = binary_to_decimal("".join(map(str,best_individual[CHROMOSOME_LENGTH:])))
        return x1_final, x2_final, final_fitnesses  # 최종 적합도 분포 반환
    else:
        return final_x1, final_x2, final_fitnesses  # 최종 적합도 분포 반환

# 실행 (여러 초기값에 대해 테스트)
random_seed_list = [0, 1, 2, 3, 4, 5]  # 테스트할 시드 값 리스트

for seed in random_seed_list:
    x1, x2, fitnesses = genetic_algorithm(seed)  # 결과 받음
    print(f"\n[Seed: {seed}]")
    if isinstance(x1, int): # x1이 int 타입이면 최적해를 찾은 것으로 간주
        max_fitness = fitness_function(x1, x2)
        print(f"  최적해: x1 = {x1}, x2 = {x2}, 최대값 = {max_fitness}")
        print(f"  최종 모집단 적합도 분포: {fitnesses}")
    else:
        if fitnesses:
            print(f"  최적해를 찾지 못했습니다.")
            print(f"  최종 세대 최고 적합도 개체 x1: {x1}, x2: {x2}")
            print(f"  최종 모집단 적합도 분포: {fitnesses}")
        else:
            print(f"  최적해를 찾지 못했습니다.")
            print(f"  최종 세대 적합도 정보 없음")