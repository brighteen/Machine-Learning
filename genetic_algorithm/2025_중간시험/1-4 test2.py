import random, time, copy
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from toolbox import (
    selection_proportional, selection_rank, selection_rank_with_elite,
    selection_stochastic_universal_sampling, selection_tournament,
    crossover_blend, crossover_linear, crossover_uniform,
    crossover_n_point, crossover_one_point, cycle_crossover,
    crossover_order, 
    mutation_random_deviation, mutation_bit_flip,
    mutation_fitness_driven_random_deviation, mutation_fitness_driven_bit_flip,
    mutation_shift, mutation_inversion, mutation_shuffle, mutation_exchange, 
    crossover_operation, mutation_operation
)

# --------- 실험 파라미터 범위 ---------
encoding_types = ['real', 'binary']
bit_lengths = [16, 20]  # for binary
population_sizes = [50, 100]
generation_counts = [100, 200]
crossover_probs = [0.7, 0.9]
mutation_probs = [0.1, 0.3]
elite_sizes = [1, 3]

# 연산자 및 내부 파라미터 조합 정의
selection_methods = [
    ('selection_proportional', selection_proportional, {}),
    ('selection_rank', selection_rank, {}),
    ('selection_rank_with_elite', selection_rank_with_elite, {'elite_size': 2}),
    ('selection_tournament', selection_tournament, {'group_size': 3}),
]

crossover_methods = [
    ('crossover_blend', crossover_blend, {'alpha': a}) for a in [0.3, 0.5]
] + [
    ('crossover_linear', crossover_linear, {'alpha': 0.7}),
    ('crossover_uniform', crossover_uniform, {'prop': 0.5}),
    ('crossover_one_point', crossover_one_point, {}),
    ('crossover_n_point', crossover_n_point, {'n': 2}),
    ('crossover_order', crossover_order, {}),
    ('cycle_crossover', cycle_crossover, {}),
]

mutation_methods = [
    ('mutation_random_deviation', mutation_random_deviation, {'mu': 0, 'sigma': s, 'p': p})
    for s in [0.3, 0.5] for p in [0.1, 0.2]
] + [
    ('mutation_exchange', mutation_exchange, {}),
    ('mutation_shift', mutation_shift, {}),
    ('mutation_bit_flip', mutation_bit_flip, {}),
    ('mutation_inversion', mutation_inversion, {}),
    ('mutation_shuffle', mutation_shuffle, {}),
]

# --------- 대상 함수 ---------
def f(x):
    return 2 * np.sin(x) + 0.5 * x

# --------- 초기화 함수 ---------
def initialize_population(size, encoding, bits=None):
    if encoding == 'real':
        return [ [round(random.uniform(-5, 13), 2)] for _ in range(size) ]
    else:
        return [ [random.randint(0, 1) for _ in range(bits)] for _ in range(size) ]

def decode_binary(ind, bits):
    sign = -1 if ind[0] == 1 else 1
    value = sum(b * 2**i for i, b in enumerate(reversed(ind[1:])))
    scaled = sign * value / (2 ** (bits - 1) - 1) * 13
    return [max(-5, min(13, scaled))]

# --------- 적합도 함수 ---------
def evaluate(ind, encoding, bits=None):
    if encoding == 'real':
        return f(ind[0])
    else:
        real_x = decode_binary(ind, bits)[0]
        return f(real_x)

# --------- 실험 메인 루프 ---------
results = []

grid = product(
    encoding_types, population_sizes, generation_counts,
    crossover_probs, mutation_probs, elite_sizes,
    selection_methods, crossover_methods, mutation_methods
)

for (encoding, pop_size, gens, cp, mp, elite,
     (sel_name, sel_fn, sel_kwargs),
     (cross_name, cross_fn, cross_kwargs),
     (mut_name, mut_fn, mut_kwargs)) in tqdm(list(grid)):

    bits = random.choice(bit_lengths) if encoding == 'binary' else None
    pop = initialize_population(pop_size, encoding, bits)
    best_fit = -float('inf')
    best_x = None
    start = time.time()

    for gen in range(gens):
        # 평가 및 정렬
        for ind in pop:
            ind.fitness = evaluate(ind, encoding, bits)

        # 선택
        selected = sel_fn(pop, **sel_kwargs)

        # 교차
        offspring = crossover_operation(selected, lambda a, b: cross_fn(a, b, **cross_kwargs), cp)

        # 돌연변이
        mutated = mutation_operation(offspring, lambda a: mut_fn(a, **mut_kwargs), mp)

        pop = mutated

        # 최고 적합도 갱신
        for ind in pop:
            fit = evaluate(ind, encoding, bits)
            x = ind[0] if encoding == 'real' else decode_binary(ind, bits)[0]
            if fit > best_fit:
                best_fit = fit
                best_x = x

    end = time.time()

    results.append({
        'encoding': encoding,
        'bits': bits,
        'population': pop_size,
        'generations': gens,
        'crossover_prob': cp,
        'mutation_prob': mp,
        'elite_size': elite,
        'selection': sel_name,
        'crossover': cross_name,
        'mutation': mut_name,
        'selection_args': sel_kwargs,
        'crossover_args': cross_kwargs,
        'mutation_args': mut_kwargs,
        'best_fx': round(best_fit, 4),
        'best_x': round(best_x, 4),
        'time': round(end - start, 4)
    })

# --------- 결과 저장 ---------
df = pd.DataFrame(results)
df.to_csv('ga_all_combinations_results.csv', index=False)
print("✅ 모든 실험 결과 저장 완료: ga_all_combinations_results.csv")
df.sort_values('best_fx', ascending=False).head()
