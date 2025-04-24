# toolbox.py
import copy
import random
import matplotlib.pyplot as plt

from individual import Item, Individual

# 랭크 기반 선택 함수 (엘리트 보존 포함)
def selection_rank_with_elite(individuals, elite_size=0):
    # 개체군을 fitness 내림차순 정렬
    sorted_individuals = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)
    rank_distance = 1 / len(individuals)
    ranks = [(1 - i * rank_distance) for i in range(len(individuals))]
    ranks_sum = sum(ranks)
    # 엘리트 개체 보존: 최고 fitness를 가진 elite_size 개체 포함
    selected = sorted_individuals[0:elite_size]

    # 나머지 개체들은 확률적으로 선택
    for i in range(len(sorted_individuals) - elite_size):
        shave = random.random() * ranks_sum
        rank_sum = 0
        for i in range(len(sorted_individuals)):
            rank_sum += ranks[i]
            if rank_sum > shave:
                selected.append(sorted_individuals[i])
                break
    return selected

# 한 점 교차 함수 (리스트 복사 후 교차)
def crossover_one_point(p1, p2):
    point = random.randint(1, len(p1) - 1)
    # 깊은 복사로 두 부모 유전자 리스트 복사
    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
    c1[point:], c2[point:] = p2[point:], p1[point:]
    return [c1, c2]

# fitness-driven 한 점 교차 함수
def crossover_fitness_driven_one_point(p1, p2):
    point = random.randint(1, len(p1.gene_list) - 1)
    c1, c2 = copy.deepcopy(p1.gene_list), copy.deepcopy(p2.gene_list)
    c1[point:], c2[point:] = p2.gene_list[point:], p1.gene_list[point:]
    child1 = Individual(c1)
    child2 = Individual(c2)
    # 후보군: 자식 두 개와 부모 두 개
    candidates = [child1, child2, p1, p2]
    # fitness 기준 내림차순 정렬 후 상위 두 개 선택
    best = sorted(candidates, key=lambda ind: ind.fitness, reverse=True)
    return best[0:2]

# 단순 비트 플립 돌연변이 함수: 하나의 무작위 위치의 비트를 반전
def mutation_bit_flip(ind):
    mut = copy.deepcopy(ind)
    pos = random.randint(0, len(ind) - 1)
    g1 = mut[pos]
    mut[pos] = (g1 + 1) % 2  # 0이면 1, 1이면 0
    return mut

# fitness-driven 돌연변이 함수: 최대 max_tries 회 시도하여 fitness가 개선되면 채택
def mutation_fitness_driven_bit_flip(ind, max_tries=3):
    for t in range(0, max_tries):
        mut = copy.deepcopy(ind.gene_list)
        pos = random.randint(0, len(ind.gene_list) - 1)
        g1 = mut[pos]
        mut[pos] = (g1 + 1) % 2
        mutated = Individual(mut)
        if mutated.fitness > ind.fitness:
            return mutated
    # 개선되지 않으면 원본 반환
    return ind

# "내 방" 아이템 집합 반환 함수: 고정된 아이템 목록 제공
def get_items_from_my_room():
    return [
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

# 전체 개체군에 대해 교차 연산을 수행하는 함수
def crossover_operation(population, method, prob):
    crossed_offspring = []
    for ind1, ind2 in zip(population[::2], population[1::2]):
        if random.random() < prob:
            kid1, kid2 = method(ind1, ind2)
            crossed_offspring.append(kid1)
            crossed_offspring.append(kid2)
        else:
            crossed_offspring.append(ind1)
            crossed_offspring.append(ind2)
    return crossed_offspring

# 전체 개체군에 대해 돌연변이 연산을 수행하는 함수
def mutation_operation(population, method, prob):
    mutated_offspring = []
    for mutant in population:
        if random.random() < prob:
            new_mutant = method(mutant)
            mutated_offspring.append(new_mutant)
        else:
            mutated_offspring.append(mutant)
    return mutated_offspring

# 세대별 통계를 플롯하는 함수 (평균 fitness와 최고 fitness)
def plot_stats(fit_avg, fit_best_ever, title):
    plt.plot(fit_avg, label="Average Fitness of Gen")
    plt.plot(fit_best_ever, label="Best Fitness")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

# 현재 세대의 통계(평균, 최고, 최고 누적)를 업데이트하는 함수
def stats(population, best_ind, fit_avg, fit_best, fit_best_ever):
    best_of_generation = max(population, key=lambda ind: ind.fitness)
    if best_ind.fitness < best_of_generation.fitness:
        best_ind = best_of_generation
    fit_avg.append(sum([ind.fitness for ind in population]) / len(population))
    fit_best.append(best_of_generation.fitness)
    fit_best_ever.append(max(fit_best + fit_best_ever))
    return best_ind, fit_avg, fit_best, fit_best_ever
