# Jupyter Notebook Cell 1: Toolbox Functions
# 유전 알고리즘 연산자 함수 정의

import random
import copy
import math
import numpy as np

# 1. Selection operators (선택 연산자)

def selection_proportional(individuals):
    """적합도에 비례하여 개체 선택 (룰렛 휠)"""
    total_fitness = sum(ind.fitness for ind in individuals)
    n = len(individuals)
    selected = []

    if n == 0 or total_fitness <= 0:
        return random.sample(individuals, n) if n > 0 else []

    cumulative_fitness = np.cumsum([ind.fitness for ind in individuals])

    for _ in range(n):
        pick = random.random() * total_fitness
        idx = np.searchsorted(cumulative_fitness, pick)
        selected.append(individuals[idx])

    return selected


def selection_rank(individuals):
    """적합도 순위에 따라 개체 선택"""
    sorted_inds = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)
    n = len(sorted_inds)
    if n == 0: return []

    ranks = [n - i for i in range(n)]
    total_rank = sum(ranks)

    selected = []
    if total_rank == 0:
         return random.sample(sorted_inds, n)

    cumulative_ranks = np.cumsum(ranks)

    for _ in range(n):
        pick = random.random() * total_rank
        idx = np.searchsorted(cumulative_ranks, pick)
        selected.append(sorted_inds[idx])

    return selected


def selection_rank_with_elite(individuals, elite_size=0):
    """랭크 선택 및 엘리트 보존"""
    sorted_inds = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)
    n = len(sorted_inds)
    if n == 0: return []

    current_elite_size = min(elite_size, n)
    elites = sorted_inds[:current_elite_size]

    remaining_inds = sorted_inds[current_elite_size:]
    n_remaining = len(remaining_inds)

    selected_from_rest = []
    if n_remaining > 0:
        ranks = [n_remaining - i for i in range(n_remaining)]
        total_rank = sum(ranks)

        if total_rank == 0:
             selected_from_rest = random.sample(remaining_inds, n_remaining)
        else:
            cumulative_ranks = np.cumsum(ranks)
            for _ in range(n_remaining):
                pick = random.random() * total_rank
                idx = np.searchsorted(cumulative_ranks, pick)
                selected_from_rest.append(remaining_inds[idx])

    return elites + selected_from_rest


def selection_stochastic_universal_sampling(individuals):
    """확률적 만능 샘플링 (SUS)"""
    total_fitness = sum(ind.fitness for ind in individuals)
    n = len(individuals)
    selected = []

    if n == 0 or total_fitness <= 0:
        return random.sample(individuals, n) if n > 0 else []

    slot_size = total_fitness / n
    start_point = random.uniform(0, slot_size)
    pointers = [start_point + i * slot_size for i in range(n)]

    cumulative_fitness = np.cumsum([ind.fitness for ind in individuals])
    ind_idx = 0
    for pointer in pointers:
        while ind_idx < n - 1 and cumulative_fitness[ind_idx] < pointer:
            ind_idx += 1
        selected.append(individuals[ind_idx])

    return selected


def selection_tournament(individuals, group_size=2):
    """토너먼트 선택"""
    selected = []
    n = len(individuals)
    if n == 0: return []
    current_group_size = min(group_size, n)
    if current_group_size < 1: current_group_size = 1

    for _ in range(n):
        participants = random.sample(individuals, current_group_size)
        selected.append(max(participants, key=lambda ind: ind.fitness))
    return selected


# 3. Crossover operators (교차 연산자)

def crossover_blend(p1_genes, p2_genes, alpha):
    """실수값 유전자에 대한 블렌드 교차 (BLX-alpha)"""
    c1_genes, c2_genes = copy.copy(p1_genes), copy.copy(p2_genes)
    for i in range(len(c1_genes)):
        g1, g2 = float(p1_genes[i]), float(p2_genes[i])
        diff = abs(g2 - g1)
        low, high = min(g1, g2) - alpha * diff, max(g1, g2) + alpha * diff
        c1_genes[i] = low + random.random() * (high - low)
        c2_genes[i] = low + random.random() * (high - low)
    return [c1_genes, c2_genes]


def crossover_linear(p1_genes, p2_genes, alpha):
    """실수값 유전자에 대한 선형 교차"""
    c1_genes, c2_genes = copy.copy(p1_genes), copy.copy(p2_genes)
    for i in range(len(c1_genes)):
        g1, g2 = float(p1_genes[i]), float(p2_genes[i])
        c1_genes[i] = alpha * g1 + (1 - alpha) * g2
        c2_genes[i] = (1 - alpha) * g1 + alpha * g2
    return [c1_genes, c2_genes]


def cycle_crossover(p1_genes, p2_genes):
    """순열 기반 유전자에 대한 사이클 교차 (CX)"""
    n = len(p1_genes)
    if n != len(p2_genes):
        raise ValueError("Parents must have the same length for cycle crossover.")
    if n == 0: return [copy.copy(p1_genes), copy.copy(p2_genes)]

    c1_genes = [None] * n
    c2_genes = [None] * n
    visited = [False] * n

    for start_index in range(n):
        if not visited[start_index]:
            cycle = []
            current_index = start_index
            while not visited[current_index]:
                visited[current_index] = True
                cycle.append(current_index)
                value_in_p1 = p1_genes[current_index]
                try:
                    current_index = p2_genes.index(value_in_p1)
                except ValueError:
                     print(f"Warning: Value {value_in_p1} not found in parent 2 during cycle crossover.")
                     break

            for idx in cycle:
                c1_genes[idx] = p1_genes[idx]
                c2_genes[idx] = p2_genes[idx]

    for i in range(n):
        if c1_genes[i] is None:
             c1_genes[i] = p2_genes[i]
             c2_genes[i] = p1_genes[i]

    if None in c1_genes or None in c2_genes:
         print(f"Warning: None values remaining after order crossover. c1: {c1_genes}, c2: {c2_genes}")
         available_genes = sorted(list(set(p1_genes) | set(p2_genes)))
         fill_count = (c1_genes.count(None) + c2_genes.count(None))
         if len(available_genes) >= fill_count:
              temp_available_c1 = available_genes[:]
              temp_available_c2 = available_genes[:]
              c1_genes = [g if g is not None else temp_available_c1.pop(0) for g in c1_genes]
              c2_genes = [g if g is not None else temp_available_c2.pop(0) for g in c2_genes]
         else:
              print("Error: Not enough available genes to fill None values.")

    return [c1_genes, c2_genes]


def crossover_n_point(p1_genes, p2_genes, n):
    """N점 교차"""
    length = len(p1_genes)
    if length != len(p2_genes):
        raise ValueError("Parents must have the same length for N-point crossover.")
    if length < 2 or n <= 0: return [copy.copy(p1_genes), copy.copy(p2_genes)]

    max_points = length - 1
    num_points = min(n, max_points)

    if num_points == 0:
        return [copy.copy(p1_genes), copy.copy(p2_genes)]

    points = sorted(random.sample(range(1, length), num_points))

    c1_genes, c2_genes = copy.copy(p1_genes), copy.copy(p2_genes)
    swap = False

    current_point_idx = 0
    start = 0
    while start < length:
        end = length if current_point_idx >= len(points) else points[current_point_idx]

        if swap:
            c1_genes[start:end] = p2_genes[start:end]
            c2_genes[start:end] = p1_genes[start:end]
        else:
            c1_genes[start:end] = p1_genes[start:end]
            c2_genes[start:end] = p2_genes[start:end]

        start = end
        swap = not swap
        current_point_idx += 1

    return [c1_genes, c2_genes]


def crossover_one_point(p1_genes, p2_genes):
    """1점 교차"""
    return crossover_n_point(p1_genes, p2_genes, 1)


def crossover_order(p1_genes, p2_genes):
    """순열 기반 유전자에 대한 순서 교차 (OX)"""
    n = len(p1_genes)
    if n != len(p2_genes):
        raise ValueError("Parents must have the same length for order crossover.")
    if n < 2: return [copy.copy(p1_genes), copy.copy(p2_genes)]

    start, end = sorted(random.sample(range(n), 2))

    c1_genes = [None] * n
    c2_genes = [None] * n

    c1_genes[start:end+1] = p1_genes[start:end+1]
    c2_genes[start:end+1] = p2_genes[start:end+1]

    p1_remaining = [gene for gene in p1_genes if gene not in c2_genes[start:end+1]]
    p2_remaining = [gene for gene in p2_genes if gene not in c1_genes[start:end+1]]

    p1_rem_idx = 0
    p2_rem_idx = 0
    for i in range(n):
        child_idx = (end + 1 + i) % n

        if c1_genes[child_idx] is not None:
             continue

        while p2_rem_idx < len(p2_remaining) and p2_remaining[p2_rem_idx] in c1_genes:
             p2_rem_idx += 1
        if p2_rem_idx < len(p2_remaining):
             c1_genes[child_idx] = p2_remaining[p2_rem_idx]
             p2_rem_idx += 1

    for i in range(n):
        child_idx = (end + 1 + i) % n
        if c2_genes[child_idx] is not None:
             continue

        while p1_rem_idx < len(p1_remaining) and p1_remaining[p1_rem_idx] in c2_genes:
             p1_rem_idx += 1
        if p1_rem_idx < len(p1_remaining):
             c2_genes[child_idx] = p1_remaining[p1_rem_idx]
             p1_rem_idx += 1

    if None in c1_genes or None in c2_genes:
         print(f"Warning: None values remaining after order crossover. c1: {c1_genes}, c2: {c2_genes}")
         available_genes = sorted(list(set(p1_genes) | set(p2_genes)))
         fill_count = (c1_genes.count(None) + c2_genes.count(None))
         if len(available_genes) >= fill_count:
              temp_available_c1 = available_genes[:]
              temp_available_c2 = available_genes[:]
              c1_genes = [g if g is not None else temp_available_c1.pop(0) for g in c1_genes]
              c2_genes = [g if g is not None else temp_available_c2.pop(0) for g in c2_genes]
         else:
              print("Error: Not enough available genes to fill None values.")

    return [c1_genes, c2_genes]


def crossover_uniform(p1_genes, p2_genes, prop):
    """균일 교차"""
    n = len(p1_genes)
    if n != len(p2_genes):
        raise ValueError("Parents must have the same length for uniform crossover.")
    if n == 0: return [copy.copy(p1_genes), copy.copy(p2_genes)]

    c1_genes, c2_genes = copy.copy(p1_genes), copy.copy(p2_genes)
    for i in range(n):
        if random.random() < prop:
            c1_genes[i], c2_genes[i] = p2_genes[i], p1_genes[i]
    return [c1_genes, c2_genes]


# 4. Mutation operators (돌연변이 연산자)

def mutation_random_deviation(ind_genes, mu, sigma, p):
    """실수값 유전자에 대한 랜덤 편차 돌연변이"""
    m_genes = copy.copy(ind_genes)
    for i in range(len(m_genes)):
        if random.random() < p:
            m_genes[i] += random.gauss(mu, sigma)
    return m_genes


def mutation_exchange(ind_genes):
    """교환 돌연변이 (Swap Mutation)"""
    m_genes = copy.copy(ind_genes)
    n = len(m_genes)
    if n < 2:
        return m_genes
    else:
        i, j = random.sample(range(n), 2)
        m_genes[i], m_genes[j] = m_genes[j], m_genes[i]
    return m_genes


def mutation_shift(ind_genes):
    """이동 돌연변이 (Shift Mutation)"""
    m_genes = copy.copy(ind_genes)
    n = len(m_genes)
    if n < 2:
         return m_genes

    from_idx = random.randint(0, n - 1)
    to_idx = random.randint(0, n - 1)

    if from_idx == to_idx:
        return m_genes

    segment = m_genes.pop(from_idx)
    m_genes.insert(to_idx, segment)

    return m_genes


def mutation_bit_flip(ind_genes):
    """비트 플립 돌연변이 (이진 유전자)"""
    m_genes = copy.copy(ind_genes)
    n = len(m_genes)
    if n == 0: return m_genes
    i = random.randint(0, n - 1)
    m_genes[i] = 1 - m_genes[i]
    return m_genes


def mutation_inversion(ind_genes):
    """역전 돌연변이 (Inversion Mutation)"""
    m_genes = copy.copy(ind_genes)
    n = len(m_genes)
    if n < 2:
         return m_genes

    i, j = sorted(random.sample(range(n), 2))
    m_genes[i:j+1] = list(reversed(m_genes[i:j+1]))

    return m_genes


def mutation_shuffle(ind_genes):
    """셔플 돌연변이 (Shuffle Mutation)"""
    m_genes = copy.copy(ind_genes)
    n = len(m_genes)
    if n < 2:
         return m_genes

    i, j = sorted(random.sample(range(n), 2))
    sub_segment = m_genes[i:j+1]
    random.shuffle(sub_segment)
    m_genes[i:j+1] = sub_segment

    return m_genes


# 5. Fitness-driven operators (적합도 기반 연산자)

def mutation_fitness_driven_random_deviation(ind, mu, sigma, p, max_tries=3):
    """적합도 기반 랜덤 편차 돌연변이"""
    IndividualClass = type(ind)
    init_params = {}
    if hasattr(ind, 'bits'):
        init_params['bits'] = ind.bits
        init_params['min_value'] = ind.min_value
        init_params['max_value'] = ind.max_value
    else:
        init_params['min_value'] = ind.min_value
        init_params['max_value'] = ind.max_value

    current_best_mutant = ind

    for _ in range(max_tries):
        mutated_genes = mutation_random_deviation(ind.gene_list, mu, sigma, p)
        new_mutant = IndividualClass(mutated_genes, **init_params)
        if new_mutant.fitness > current_best_mutant.fitness:
            current_best_mutant = new_mutant

    return current_best_mutant


def mutation_fitness_driven_bit_flip(ind, max_tries=3):
    """적합도 기반 비트 플립 돌연변이"""
    IndividualClass = type(ind)
    init_params = {}
    if hasattr(ind, 'bits'):
        init_params['bits'] = ind.bits
        init_params['min_value'] = ind.min_value
        init_params['max_value'] = ind.max_value
    else:
         print("Error: mutation_fitness_driven_bit_flip applied to non-binary individual.")
         return ind

    current_best_mutant = ind

    for _ in range(max_tries):
        mutated_genes = mutation_bit_flip(ind.gene_list)
        new_mutant = IndividualClass(mutated_genes, **init_params)
        if new_mutant.fitness > current_best_mutant.fitness:
            current_best_mutant = new_mutant

    return current_best_mutant