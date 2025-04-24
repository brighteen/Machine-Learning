## 1. 선택 연산자 (Selection)


# selection_proportional.py
import random

def selection_proportional(individuals):
    sorted_individuals = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)
    fitness_sum = sum(ind.fitness for ind in individuals)
    selected = []

    for _ in range(len(sorted_individuals)):
        shave = random.random() * fitness_sum
        roulette_sum = 0
        for ind in sorted_individuals:
            roulette_sum += ind.fitness
            if roulette_sum > shave:
                selected.append(ind)
                break

    return selected



# selection_rank.py
import random

def selection_rank(individuals):
    sorted_individuals = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)
    rank_distance = 1 / len(individuals)
    ranks = [(1 - i * rank_distance) for i in range(len(individuals))]
    ranks_sum = sum(ranks)
    selected = []

    for _ in range(len(sorted_individuals)):
        shave = random.random() * ranks_sum
        rank_sum = 0
        for i in range(len(sorted_individuals)):
            rank_sum += ranks[i]
            if rank_sum > shave:
                selected.append(sorted_individuals[i])
                break

    return selected



# selection_rank_with_elite.py
import random

def selection_rank_with_elite(individuals, elite_size=0):
    sorted_individuals = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)
    rank_distance = 1 / len(individuals)
    ranks = [(1 - i * rank_distance) for i in range(len(individuals))]
    ranks_sum = sum(ranks)
    selected = sorted_individuals[:elite_size]

    for _ in range(len(sorted_individuals) - elite_size):
        shave = random.random() * ranks_sum
        rank_sum = 0
        for j in range(len(sorted_individuals)):
            rank_sum += ranks[j]
            if rank_sum > shave:
                selected.append(sorted_individuals[j])
                break

    return selected



# selection_stochastic_universal_sampling.py
import random

def selection_stochastic_universal_sampling(individuals):
    sorted_individuals = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)
    fitness_sum = sum(ind.fitness for ind in individuals)

    distance = fitness_sum / len(individuals)
    shift = random.uniform(0, distance)
    borders = [shift + i * distance for i in range(len(individuals))]

    selected = []
    for border in borders:
        i = 0
        roulette_sum = sorted_individuals[i].fitness
        while roulette_sum < border:
            i += 1
            roulette_sum += sorted_individuals[i].fitness
        selected.append(sorted_individuals[i])

    return selected



# selection_tournament.py
import random

def selection_tournament(individuals, group_size=2):
    selected = []
    for _ in range(len(individuals)):
        candidates = [random.choice(individuals) for _ in range(group_size)]
        selected.append(max(candidates, key=lambda ind: ind.fitness))
    return selected




## 2. 교차 연산자 (Crossover)


# blend.py (BLX-α)
import copy, random

def crossover_blend(p1, p2, alpha):
    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
    for i in range(len(p1)):
        l = min(c1[i], c2[i]) - alpha * abs(c2[i] - c1[i])
        u = max(c1[i], c2[i]) + alpha * abs(c2[i] - c1[i])
        c1[i] = round(l + random.random() * (u - l), 2)
        c2[i] = round(l + random.random() * (u - l), 2)
    return [c1, c2]



# cycle.py
def cycle_crossover(p1, p2):
    length = len(p1)
    c1, c2 = [None] * length, [None] * length
    visited = [False] * length
    cycle = 0

    while not all(visited):
        start = visited.index(False)
        current = start
        if cycle % 2 == 0:
            while True:
                c1[current] = p1[current]
                c2[current] = p2[current]
                visited[current] = True
                current = p1.index(p2[current])
                if current == start:
                    break
        else:
            while True:
                c1[current] = p2[current]
                c2[current] = p1[current]
                visited[current] = True
                current = p1.index(p2[current])
                if current == start:
                    break
        cycle += 1

    return c1, c2



# fitness_driven.py (BLX-α + fitness filtering)
import copy, random
from math import sin, cos

def fitness_function(x, y):
    return sin(x) * cos(y)

class Individual:
    def __init__(self, x, y):
        self.gene_set = [x, y]
        self.fitness = fitness_function(x, y)

def crossover_fitness_driven_blend(ind1, ind2, alpha):
    c1 = copy.deepcopy(ind1.gene_set)
    c2 = copy.deepcopy(ind2.gene_set)
    for i in range(len(c1)):
        l = min(c1[i], c2[i]) - alpha * abs(c2[i] - c1[i])
        u = max(c1[i], c2[i]) + alpha * abs(c2[i] - c1[i])
        c1[i] = round(l + random.random() * (u - l), 2)
        c2[i] = round(l + random.random() * (u - l), 2)
    child1 = Individual(c1[0], c1[1])
    child2 = Individual(c2[0], c2[1])
    candidates = [ind1, ind2, child1, child2]
    best = sorted(candidates, key=lambda ind: ind.fitness, reverse=True)
    return best[:2]



# linear.py
import copy, random

def crossover_linear(p1, p2, alpha):
    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
    for i in range(len(p1)):
        c1[i] = round(p1[i] + alpha * (p2[i] - p1[i]), 2)
        c2[i] = round(p2[i] - alpha * (p2[i] - p1[i]), 2)
    return [c1, c2]



# n_point.py
import copy, random

def crossover_n_point(p1, p2, n):
    ps = random.sample(range(1, len(p1) - 1), n) + [0, len(p1)]
    ps = sorted(ps)
    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
    for i in range(n + 1):
        if i % 2 == 1:
            c1[ps[i]:ps[i+1]] = p2[ps[i]:ps[i+1]]
            c2[ps[i]:ps[i+1]] = p1[ps[i]:ps[i+1]]
    return [c1, c2]



# one_point.py
import copy, random

def crossover_one_point(p1, p2):
    point = random.randint(1, len(p1) - 1)
    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
    c1[point:], c2[point:] = p2[point:], p1[point:]
    return [c1, c2]



# order.py
import random
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



# uniform.py
import copy, random

def crossover_uniform(p1, p2, prop):
    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
    for i in range(len(p1)):
        if random.random() < prop:
            c1[i], c2[i] = p2[i], p1[i]
    return [c1, c2]



# fitness-driven one-point crossover
import copy, random

def crossover_fitness_driven_one_point(p1, p2):
    point = random.randint(1, len(p1.gene_list) - 1)
    c1 = copy.deepcopy(p1.gene_list)
    c2 = copy.deepcopy(p2.gene_list)
    c1[point:], c2[point:] = p2.gene_list[point:], p1.gene_list[point:]
    child1 = Individual(c1)
    child2 = Individual(c2)
    candidates = [child1, child2, p1, p2]
    best = sorted(candidates, key=lambda ind: ind.fitness, reverse=True)
    return best[:2]



# fitness-driven order crossover
import copy, random
from math import nan

def crossover_fitness_driven_order(ind1, ind2):
    p1, p2 = ind1.gene_list, ind2.gene_list
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
            c1[j1 % length] = idx1; j1 += 1
        if not spaces2[idx2]:
            c2[j2 % length] = idx2; j2 += 1
    for i in range(start, end+1):
        c1[i], c2[i] = t2[i], t1[i]
    child1 = Individual([x+zero_shift for x in c1])
    child2 = Individual([x+zero_shift for x in c2])
    candidates = [child1, child2, ind1, ind2]
    best = sorted(candidates, key=lambda ind: ind.fitness, reverse=True)
    return best[:2]




## 3. 돌연변이 연산자 (Mutation)


# random_deviation.py
import copy, random

def mutation_random_deviation(ind, mu, sigma, p):
    mut = copy.deepcopy(ind)
    for i in range(len(mut)):
        if random.random() < p:
            mut[i] = mut[i] + random.gauss(mu, sigma)
    return mut



# exchange.py
import copy, random

def mutation_exchange(ind):
    mut = copy.deepcopy(ind)
    pos = random.sample(range(len(mut)), 2)
    mut[pos[0]], mut[pos[1]] = mut[pos[1]], mut[pos[0]]
    return mut



# shift.py
import copy, random
from math import copysign

def mutation_shift(ind):
    mut = copy.deepcopy(ind)
    pos = random.sample(range(len(mut)), 2)
    g1 = mut[pos[0]]
    dir = int(copysign(1, pos[1] - pos[0]))
    for i in range(pos[0], pos[1], dir):
        mut[i] = mut[i + dir]
    mut[pos[1]] = g1
    return mut



# bit_flip.py
import copy, random

def mutation_bit_flip(ind):
    mut = copy.deepcopy(ind)
    pos = random.randint(0, len(ind)-1)
    mut[pos] = (mut[pos] + 1) % 2
    return mut



# inversion.py
import copy, random

def mutation_inversion(ind):
    mut = copy.deepcopy(ind)
    temp = copy.deepcopy(ind)
    pos = sorted(random.sample(range(len(mut)), 2))
    for i in range(pos[1] - pos[0] + 1):
        mut[pos[0]+i] = temp[pos[1]-i]
    return mut



# shuffle.py
import copy, random

def mutation_shuffle(ind):
    mut = copy.deepcopy(ind)
    pos = sorted(random.sample(range(len(mut)), 2))
    subrange = mut[pos[0]:pos[1]+1]
    random.shuffle(subrange)
    mut[pos[0]:pos[1]+1] = subrange
    return mut



# fitness_driven random deviation
import copy, random
from math import sin
from typing import List

def func(x):
    return sin(x) - .2 * abs(x)

class Individual:
    def __init__(self, gene_list: List[float]):
        self.gene_list = gene_list
        self.fitness = func(self.gene_list[0])

def mutation_fitness_driven_random_deviation(ind, mu, sigma, p, max_tries=3):
    for _ in range(max_tries):
        mut_genes = copy.deepcopy(ind.gene_list)
        for i in range(len(mut_genes)):
            if random.random() < p:
                mut_genes[i] += random.gauss(mu, sigma)
        mut = Individual(mut_genes)
        if mut.fitness > ind.fitness:
            return mut
    return ind



# fitness_driven bit flip
import copy, random

def mutation_fitness_driven_bit_flip(ind, max_tries=3):
    for _ in range(max_tries):
        mut = copy.deepcopy(ind.gene_list)
        pos = random.randint(0, len(ind.gene_list)-1)
        mut[pos] = (mut[pos] + 1) % 2
        mutated = Individual(mut)
        if mutated.fitness > ind.fitness:
            return mutated
    return ind



# mutation_shift_one (1 비트 이동)
import copy, random
from math import floor

def mutation_shift_one(ind):
    mut = copy.deepcopy(ind.gene_list)
    one_poses = [i for i,v in enumerate(mut) if v==1]
    one_pos = random.choice(one_poses)
    x_coord = one_pos % ind.rows
    y_coord = floor(one_pos / ind.rows)
    x_shifted = max(min(x_coord + random.randint(-10,10), ind.cols-1),0)
    y_shifted = max(min(y_coord + random.randint(-10,10), ind.rows-1),0)
    mut[y_shifted*ind.rows + x_shifted] = 1
    mut[one_pos] = 0
    return mut

# mutation_bit_flip_ones (1 비트 중 무작위 비트 플립)
import copy, random

def mutation_bit_flip_ones(ind):
    mut = copy.deepcopy(ind)
    one_positions = [i for i,v in enumerate(mut) if v==1]
    flip_index = random.choice(one_positions)
    mut[flip_index] = (mut[flip_index] + 1) % 2
    return mut

# fitness_driven shift
import copy, random
from math import copysign

def mutation_fitness_driven_shift(ind, max_tries=3):
    for _ in range(max_tries):
        mut = copy.deepcopy(ind.gene_list)
        pos = random.sample(range(len(mut)), 2)
        g1 = mut[pos[0]]
        dir = int(copysign(1, pos[1] - pos[0]))
        for i in range(pos[0], pos[1], dir):
            mut[i] = mut[i + dir]
        mut[pos[1]] = g1
        mutated = Individual(mut)
        if mutated.fitness > ind.fitness:
            return mutated
    return ind

# fitness_driven shuffle
import copy, random

def mutation_fitness_driven_shuffle(ind, max_tries=3):
    for _ in range(max_tries):
        mut = copy.deepcopy(ind.gene_list)
        pos = sorted(random.sample(range(len(mut)), 2))
        sub = mut[pos[0]:pos[1]+1]
        random.shuffle(sub)
        mut[pos[0]:pos[1]+1] = sub
        mutated = Individual(mut)
        if mutated.fitness > ind.fitness:
            return mutated
    return ind
