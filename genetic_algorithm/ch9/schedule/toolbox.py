import copy
import random
import matplotlib.pyplot as plt

from individual import Individual

# 랭크 기반 선택 함수: 개체들을 fitness 기준 내림차순으로 정렬하고, 엘리트 개체를 보존한 후 확률적으로 선택
def selection_rank_with_elite(individuals, elite_size = 0):
    sorted_individuals = sorted(individuals, key = lambda ind: ind.fitness, reverse = True)
    rank_distance = 1 / len(individuals)
    ranks = [(1 - i * rank_distance) for i in range(len(individuals))]
    ranks_sum = sum(ranks)
    selected = sorted_individuals[0:elite_size]

    for i in range(len(sorted_individuals) - elite_size):
        shave = random.random() * ranks_sum
        rank_sum = 0
        for j in range(len(sorted_individuals)):
            rank_sum += ranks[j]
            if rank_sum > shave:
                selected.append(sorted_individuals[i])
                break

    return selected

  
# n-점 교차 함수: 두 부모의 유전자 리스트에서 n개의 교차점을 선택하여 교차 연산 수행  
def crossover_n_point(p1, p2, n):
    ps = random.sample(range(1, len(p1) - 1), n)
    ps.append(0)
    ps.append(len(p1))
    ps = sorted(ps)
    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
    for i in range(0, n + 1):
        if i % 2 == 0:
            continue
        c1[ps[i]:ps[i + 1]] = p2[ps[i]:ps[i + 1]]
        c2[ps[i]:ps[i + 1]] = p1[ps[i]:ps[i + 1]]
    return [c1, c2]

  
# fitness-driven 한 점 교차 함수: 한 점 교차 후, 생성된 자식과 부모 중 fitness가 높은 두 개체를 반환  
def crossover_fitness_driven_one_point(p1, p2):
    point = random.randint(1, len(p1.gene_list) - 1)
    c1, c2 = copy.deepcopy(p1.gene_list), copy.deepcopy(p2.gene_list)
    c1[point:], c2[point:] = p2.gene_list[point:], p1.gene_list[point:]
    child1 = Individual(c1)
    child2 = Individual(c2)
    candidates = [child1, child2, p1, p2]
    best = sorted(candidates, key = lambda ind: ind.fitness, reverse = True)
    return best[0:2]

  
# 단순 비트 플립 돌연변이 함수: 유전자 리스트의 임의 위치에서 0과 1을 반전  
def mutation_bit_flip(ind):
    mut = copy.deepcopy(ind)
    pos = random.randint(0, len(ind) - 1)
    g1 = mut[pos]
    mut[pos] = (g1 + 1) % 2
    return mut

  
# 유전자 리스트를 일정 구간으로 섞어 돌연변이 발생시키는 함수
def mutation_shuffle(ind):
    mut = copy.deepcopy(ind)
    pos = sorted(random.sample(range(0, len(mut)), 2))
    subrange = mut[pos[0]:pos[1] + 1]
    random.shuffle(subrange)
    mut[pos[0]:pos[1] + 1] = subrange
    return mut

  
# fitness-driven 돌연변이 함수: 최대 max_tries회 시도하여 fitness가 개선되면 돌연변이를 채택, 그렇지 않으면 원본 반환  
def mutation_fitness_driven_bit_flip(ind, max_tries = 3):
    for t in range(0, max_tries):
        mut = copy.deepcopy(ind.gene_list)
        pos = random.randint(0, len(ind.gene_list) - 1)
        g1 = mut[pos]
        mut[pos] = (g1 + 1) % 2
        mutated = Individual(mut)
        if mutated.fitness > ind.fitness:
            return mutated
    return ind

# 교차 연산: 전체 개체군을 두 개씩 짝지어, 지정된 확률에 따라 교차 연산 수행  
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

  
# 돌연변이 연산: 전체 개체군에 대해 지정된 확률로 돌연변이 함수 적용  
def mutation_operation(population, method, prob):
    mutated_offspring = []
    for mutant in population:
        if random.random() < prob:
            new_mutant = method(mutant)
            mutated_offspring.append(new_mutant)
        else:
            mutated_offspring.append(mutant)
    return mutated_offspring

  
# 세대별 통계 계산 함수: 현재 세대의 평균 및 최고 fitness를 계산하여 업데이트  
def stats(population, best_ind, fit_avg, fit_best):
    best_of_generation = max(population, key = lambda ind: ind.fitness)
    if best_ind.fitness < best_of_generation.fitness:
        best_ind = best_of_generation
    fit_avg.append(sum([ind.fitness for ind in population]) / len(population))
    fit_best.append(best_ind.fitness)
    return best_ind, fit_avg, fit_best

  
# 세대별 통계 플롯 함수: 평균 및 최고 fitness를 플롯으로 출력  
def plot_stats(fit_avg, fit_best, title):
    plt.plot(fit_avg, label = "Average Fitness of Generation")
    plt.plot(fit_best, label = "Best Fitness")
    plt.title(title)
    plt.legend(loc = "lower right")
    plt.show()
    plt.close()
