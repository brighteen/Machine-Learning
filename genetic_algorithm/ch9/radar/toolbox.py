import copy
import random
from math import floor
import matplotlib.pyplot as plt
from individual import Individual

# 엘리트 선택 함수: 개체군을 적합도 순으로 정렬한 후 확률적으로 선택
def selection_rank_with_elite(individuals, elite_size=0):
    sorted_individuals = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)
    rank_distance = 1 / len(individuals)
    ranks = [(1 - i * rank_distance) for i in range(len(individuals))]
    ranks_sum = sum(ranks)
    selected = sorted_individuals[0:elite_size]
    for _ in range(len(sorted_individuals) - elite_size):
        shave = random.random() * ranks_sum
        rank_sum = 0
        for i in range(len(sorted_individuals)):
            rank_sum += ranks[i]
            if rank_sum > shave:
                selected.append(sorted_individuals[i])
                break
    return selected

# n-점 교차 함수: 두 부모의 유전자 배열에서 n개의 교차점 선택 후 교차
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

# fitness-driven one-point 교차 (앞서 배낭 문제와 동일)
def crossover_fitness_driven_one_point(p1, p2):
    point = random.randint(1, len(p1.gene_list) - 1)
    c1, c2 = copy.deepcopy(p1.gene_list), copy.deepcopy(p2.gene_list)
    c1[point:], c2[point:] = p2.gene_list[point:], p1.gene_list[point:]
    child1 = Individual(c1)
    child2 = Individual(c2)
    candidates = [child1, child2, p1, p2]
    best = sorted(candidates, key=lambda ind: ind.fitness, reverse=True)
    return best[0:2]

# 기본 bit flip 돌연변이 함수
def mutation_bit_flip(ind):
    mut = copy.deepcopy(ind)
    pos = random.randint(0, len(ind) - 1)
    g1 = mut[pos]
    mut[pos] = (g1 + 1) % 2
    return mut

# 돌연변이 연산: 주어진 확률 하에 fitness-driven bit flip 반복 수행
def mutation_fitness_driven_bit_flip(ind, max_tries=3):
    for t in range(max_tries):
        mut = copy.deepcopy(ind.gene_list)
        pos = random.randint(0, len(ind.gene_list) - 1)
        g1 = mut[pos]
        mut[pos] = (g1 + 1) % 2
        mutated = Individual(mut)
        if mutated.fitness > ind.fitness:
            return mutated
    return ind

# 교차 연산을 전체 개체군에 적용하는 함수
def crossover_operation(population, method, prob):
    crossed_offspring = []
    for ind1, ind2 in zip(population[::2], population[1::2]):
        if random.random() < prob:
            kid1, kid2 = method(ind1, ind2)
            crossed_offspring.extend([kid1, kid2])
        else:
            crossed_offspring.extend([ind1, ind2])
    return crossed_offspring

# 돌연변이 연산을 전체 개체군에 적용하는 함수
def mutation_operation(population, method, prob):
    mutated_offspring = []
    for mutant in population:
        if random.random() < prob:
            new_mutant = method(mutant)
            mutated_offspring.append(new_mutant)
        else:
            mutated_offspring.append(mutant)
    return mutated_offspring

def stats(population, best_ind, fit_avg, fit_best):
    best_of_generation = max(population, key=lambda ind: ind.fitness)
    if best_ind.fitness < best_of_generation.fitness:
        best_ind = best_of_generation
    fit_avg.append(sum([ind.fitness for ind in population]) / len(population))
    fit_best.append(best_ind.fitness)
    return best_ind, fit_avg, fit_best

def plot_stats(fit_avg, fit_best, title):
    plt.plot(fit_avg, label="Average Fitness of Generation")
    plt.plot(fit_best, label="Best Fitness")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
