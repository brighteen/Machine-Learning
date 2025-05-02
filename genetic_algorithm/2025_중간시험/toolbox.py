import random
import copy
import math

# 1. Selection operators

def selection_proportional(individuals):
    sorted_inds = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)
    total = sum(ind.fitness for ind in sorted_inds)
    selected = []
    for _ in range(len(sorted_inds)):
        pick = random.random() * total
        acc = 0
        for ind in sorted_inds:
            acc += ind.fitness
            if acc > pick:
                selected.append(ind)
                break
    return selected


def selection_rank(individuals):
    sorted_inds = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)
    n = len(sorted_inds)
    ranks = [1 - i / n for i in range(n)]
    total = sum(ranks)
    selected = []
    for _ in range(n):
        pick = random.random() * total
        acc = 0
        for ind, rank in zip(sorted_inds, ranks):
            acc += rank
            if acc > pick:
                selected.append(ind)
                break
    return selected


def selection_rank_with_elite(individuals, elite_size=0):
    sorted_inds = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)
    n = len(sorted_inds)
    ranks = [1 - i / n for i in range(n)]
    total = sum(ranks)
    selected = sorted_inds[:elite_size]
    for _ in range(n - elite_size):
        pick = random.random() * total
        acc = 0
        for ind, rank in zip(sorted_inds, ranks):
            acc += rank
            if acc > pick:
                selected.append(ind)
                break
    return selected


def selection_stochastic_universal_sampling(individuals):
    sorted_inds = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)
    total = sum(ind.fitness for ind in sorted_inds)
    step = total / len(sorted_inds)
    start = random.uniform(0, step)
    borders = [start + i * step for i in range(len(sorted_inds))]
    selected = []
    acc = sorted_inds[0].fitness
    idx = 0
    for border in borders:
        while acc < border:
            idx += 1
            acc += sorted_inds[idx].fitness
        selected.append(sorted_inds[idx])
    return selected


def selection_tournament(individuals, group_size=2):
    selected = []
    for _ in range(len(individuals)):
        participants = random.sample(individuals, group_size)
        selected.append(max(participants, key=lambda ind: ind.fitness))
    return selected


# 2. Crossover and Mutation operations

def crossover_operation(population, method, prob):
    offspring = []
    for a, b in zip(population[::2], population[1::2]):
        if random.random() < prob:
            c1, c2 = method(a, b)
            offspring.extend([c1, c2])
        else:
            offspring.extend([a, b])
    return offspring


def mutation_operation(population, method, prob):
    mutated_offspring = []
    for mutant in population:
        if random.random() < prob:
            new_mutant = method(mutant)
            mutated_offspring.append(new_mutant)
        else:
            mutated_offspring.append(mutant)
    return mutated_offspring


# 3. Crossover operators

def crossover_blend(p1, p2, alpha):
    c1, c2 = copy.copy(p1), copy.copy(p2)
    for i in range(len(c1)):
        diff = abs(c2[i] - c1[i])
        low, high = min(c1[i], c2[i]) - alpha * diff, max(c1[i], c2[i]) + alpha * diff
        c1[i] = round(low + random.random() * (high - low), 2)
        c2[i] = round(low + random.random() * (high - low), 2)
    return [c1, c2]


def crossover_linear(p1, p2, alpha):
    c1, c2 = copy.copy(p1), copy.copy(p2)
    for i in range(len(c1)):
        c1[i] = round(p1[i] + alpha * (p2[i] - p1[i]), 2)
        c2[i] = round(p2[i] - alpha * (p2[i] - p1[i]), 2)
    return [c1, c2]


def cycle_crossover(p1, p2):
    n = len(p1)
    c1, c2 = [None]*n, [None]*n
    visited = [False]*n
    cycle = 0
    while not all(visited):
        start = visited.index(False)
        idx = start
        if cycle % 2 == 0:
            while True:
                c1[idx], c2[idx] = p1[idx], p2[idx]
                visited[idx] = True
                idx = p1.index(p2[idx])
                if idx == start:
                    break
        else:
            while True:
                c1[idx], c2[idx] = p2[idx], p1[idx]
                visited[idx] = True
                idx = p1.index(p2[idx])
                if idx == start:
                    break
        cycle += 1
    return c1, c2


def crossover_n_point(p1, p2, n):
    points = sorted(random.sample(range(1, len(p1)), n))
    points = [0] + points + [len(p1)]
    c1, c2 = copy.copy(p1), copy.copy(p2)
    for i in range(len(points)-1):
        if i % 2 == 1:
            start, end = points[i], points[i+1]
            c1[start:end], c2[start:end] = p2[start:end], p1[start:end]
    return [c1, c2]


def crossover_one_point(p1, p2):
    point = random.randint(1, len(p1)-1)
    c1, c2 = copy.copy(p1), copy.copy(p2)
    c1[point:], c2[point:] = p2[point:], p1[point:]
    return [c1, c2]

from math import nan

def crossover_order(p1, p2):
    zero_shift = min(p1)
    length = len(p1)
    start, end = sorted(random.sample(range(length), 2))
    c1, c2 = [nan]*length, [nan]*length
    t1 = [x-zero_shift for x in p1]
    t2 = [x-zero_shift for x in p2]
    spaces1 = [True]*length
    spaces2 = [True]*length
    for i in range(length):
        if not (start <= i <= end):
            spaces1[t2[i]] = False
            spaces2[t1[i]] = False
    j1 = j2 = end+1
    for i in range(length):
        idx1 = t1[(end + i + 1) % length]
        idx2 = t2[(end + i + 1) % length]
        if not spaces1[idx1]:
            c1[j1 % length] = idx1
            j1 += 1
        if not spaces2[idx2]:
            c2[j2 % length] = idx2
            j2 += 1
    for i in range(start, end+1):
        c1[i], c2[i] = t2[i], t1[i]
    child1 = [x+zero_shift for x in c1]
    child2 = [x+zero_shift for x in c2]
    return [child1, child2]

def crossover_uniform(p1, p2, prop):
    c1, c2 = copy.copy(p1), copy.copy(p2)
    for i in range(len(c1)):
        if random.random() < prop:
            c1[i], c2[i] = p2[i], p1[i]
    return [c1, c2]

# 4. Mutation operators

def mutation_random_deviation(ind, mu, sigma, p):
    m = copy.copy(ind)
    for i in range(len(m)):
        if random.random() < p:
            m[i] += random.gauss(mu, sigma)
    return m


def mutation_exchange(ind):
    m = copy.copy(ind)
    if len(m) < 2:
        m[0] += random.gauss(0, 0.1)
    else:
        i, j = random.sample(range(len(m)), 2)
        m[i], m[j] = m[j], m[i]
    return m


def mutation_shift(ind):
    m = copy.copy(ind)
    i, j = sorted(random.sample(range(len(m)), 2))
    segment = m.pop(i)
    m.insert(j, segment)
    return m


def mutation_bit_flip(ind):
    m = copy.copy(ind)
    i = random.randint(0, len(m)-1)
    m[i] = (m[i] + 1) % 2
    return m


def mutation_inversion(ind):
    m = copy.copy(ind)
    i, j = sorted(random.sample(range(len(m)), 2))
    m[i:j+1] = reversed(m[i:j+1])
    return m


def mutation_shuffle(ind):
    m = copy.copy(ind)
    i, j = sorted(random.sample(range(len(m)), 2))
    sub = m[i:j+1]
    random.shuffle(sub)
    m[i:j+1] = sub
    return m


# 5. Fitness-driven operators

def crossover_fitness_driven_blend(ind1, ind2, alpha):
    children = crossover_blend(ind1.gene_set, ind2.gene_set, alpha)
    inds = [ind1, ind2] + [type(ind1)(*c) for c in children]
    return sorted(inds, key=lambda x: x.fitness, reverse=True)[:2]


def crossover_fitness_driven_one_point(ind1, ind2):
    children = crossover_one_point(ind1.gene_list, ind2.gene_list)
    inds = [ind1, ind2] + [type(ind1)(c) for c in children]
    return sorted(inds, key=lambda x: x.fitness, reverse=True)[:2]


def crossover_fitness_driven_order(ind1, ind2):
    children = crossover_order(ind1.gene_list, ind2.gene_list)
    inds = [ind1, ind2] + [type(ind1)(c) for c in children]
    return sorted(inds, key=lambda x: x.fitness, reverse=True)[:2]


def mutation_fitness_driven_random_deviation(ind, mu, sigma, p, max_tries=3):
    for _ in range(max_tries):
        m = mutation_random_deviation(ind.gene_list if hasattr(ind, 'gene_list') else ind, mu, sigma, p)
        new = type(ind)(m) if hasattr(ind, 'gene_list') else m
        if getattr(new, 'fitness', ind.fitness) > ind.fitness:
            return new
    return ind


def mutation_fitness_driven_bit_flip(ind, max_tries=3):
    best = ind
    for _ in range(max_tries):
        m = mutation_bit_flip(ind.gene_list if hasattr(ind, 'gene_list') else ind)
        new = type(ind)(m) if hasattr(ind, 'gene_list') else m
        if getattr(new, 'fitness', ind.fitness) > getattr(best, 'fitness', ind.fitness):
            best = new
    return best


def mutation_fitness_driven_shift(ind, max_tries=3):
    for _ in range(max_tries):
        m = mutation_shift(ind.gene_list if hasattr(ind, 'gene_list') else ind)
        new = type(ind)(m) if hasattr(ind, 'gene_list') else m
        if new.fitness > ind.fitness:
            return new
    return ind


def mutation_fitness_driven_shuffle(ind, max_tries=3):
    for _ in range(max_tries):
        m = mutation_shuffle(ind.gene_list if hasattr(ind, 'gene_list') else ind)
        new = type(ind)(m) if hasattr(ind, 'gene_list') else m
        if new.fitness > ind.fitness:
            return new
    return ind
