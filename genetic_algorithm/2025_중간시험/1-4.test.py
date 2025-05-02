import random
import numpy as np
from itertools import product
from toolbox import (
    selection_proportional, selection_rank, selection_rank_with_elite,
    selection_stochastic_universal_sampling, selection_tournament,
    crossover_blend, crossover_linear, crossover_uniform,
    crossover_n_point, crossover_one_point,
    mutation_random_deviation, mutation_bit_flip,
    mutation_fitness_driven_random_deviation, mutation_fitness_driven_bit_flip
)

def func(x):
    return 2 * np.sin(x) + 0.5 * x

class RealIndividual:
    def __init__(self, gene_list) -> None:
        if isinstance(gene_list, RealIndividual):
            self.gene_list = gene_list.gene_list
        else:
            self.gene_list = gene_list if isinstance(gene_list, list) else [gene_list]
        self.gene_list[0] = max(min(self.gene_list[0], 13), -5)
        self.fitness = -func(self.gene_list[0])

    def __str__(self):
        return f'x: {self.gene_list[0]}, f(x): {func(self.gene_list[0])}'

class BinaryIndividual:
    def __init__(self, gene_list=None, bits=16):
        self.bits = bits
        self.min_value = -5
        self.max_value = 13

        if gene_list is None:
            self.gene_list = [random.randint(0, 1) for _ in range(bits)]
        else:
            self.gene_list = gene_list

        self.x = self._decode_binary()
        self.fitness = -func(self.x)

    def _decode_binary(self):
        sign = -1 if self.gene_list[0] == 1 else 1
        value = 0
        for i, bit in enumerate(self.gene_list[1:], 1):
            value = value * 2 + bit
        max_binary = 2 ** (self.bits - 1) - 1
        scaled = sign * (value / max_binary) * self.max_value
        return max(min(scaled, self.max_value), self.min_value)

    def __str__(self):
        binary = ''.join(map(str, self.gene_list))
        return f'Binary: {binary}, x: {self.x:.4f}, f(x): {func(self.x):.4f}'

class AdaptiveGA:
    def __init__(self, encoding_type='real', population_size=50, max_generations=200,
                 initial_crossover_prob=0.9, final_crossover_prob=0.7,
                 initial_mutation_prob=0.3, final_mutation_prob=0.1,
                 elite_size=3, bits=20):
        self.encoding_type = encoding_type
        self.population_size = population_size
        self.max_generations = max_generations
        self.initial_crossover_prob = initial_crossover_prob
        self.final_crossover_prob = final_crossover_prob
        self.initial_mutation_prob = initial_mutation_prob
        self.final_mutation_prob = final_mutation_prob
        self.elite_size = elite_size
        self.bits = bits
        self.convergence_threshold = 1e-12
        self.convergence_generations = 10

    def get_adaptive_rates(self, generation):
        """세대에 따라 교차/돌연변이 확률 조정"""
        progress = generation / self.max_generations
        crossover_prob = self.initial_crossover_prob - (self.initial_crossover_prob - self.final_crossover_prob) * progress
        mutation_prob = self.initial_mutation_prob - (self.initial_mutation_prob - self.final_mutation_prob) * progress
        return crossover_prob, mutation_prob

    def create_individual(self):
        """실수/이진 인코딩에 따른 개체 생성"""
        if self.encoding_type == 'real':
            return RealIndividual([random.uniform(-5, 13)])
        else:
            return BinaryIndividual(bits=self.bits)

    def hybrid_selection(self, population, generation):
        """하이브리드 선택 전략"""
        if generation % 2 == 0:
            return selection_rank_with_elite(population, elite_size=self.elite_size)
        else:
            return selection_tournament(population, group_size=3)

    def run(self, selection_method, crossover_tuple, mutation_tuple):
        """개선된 유전 알고리즘 실행"""
        population = [self.create_individual() for _ in range(self.population_size)]
        best_ever = min(population, key=lambda x: func(x.gene_list[0] if self.encoding_type == 'real' else x.x))
        
        prev_best_fx = float('inf')
        convergence_count = 0
        
        crossover_method, crossover_params = crossover_tuple
        mutation_method, mutation_params = mutation_tuple
        
        for generation in range(self.max_generations):
            # 적응적 확률 조정
            crossover_prob, mutation_prob = self.get_adaptive_rates(generation)
            
            # 하이브리드 선택 또는 일반 선택
            if selection_method == 'hybrid':
                selected = self.hybrid_selection(population, generation)
            else:
                selected = selection_method(population)

            # 교차
            offspring = []
            for p1, p2 in zip(selected[::2], selected[1::2]):
                if random.random() < crossover_prob:
                    if self.encoding_type == 'real':
                        # 각 교차 연산자에 맞는 파라미터 적용
                        current_params = crossover_params.copy()
                        if 'alpha' in current_params and generation >= self.max_generations // 2:
                            current_params['alpha'] *= 0.6
                        elif 'prop' in current_params and generation >= self.max_generations // 2:
                            current_params['prop'] = min(0.7, current_params['prop'] * 1.2)
                        c1, c2 = crossover_method(p1.gene_list, p2.gene_list, **current_params)
                    else:
                        c1, c2 = crossover_method(p1.gene_list, p2.gene_list, **crossover_params)
                    offspring.extend([
                        RealIndividual(c1) if self.encoding_type == 'real' else BinaryIndividual(c1, self.bits),
                        RealIndividual(c2) if self.encoding_type == 'real' else BinaryIndividual(c2, self.bits)
                    ])
                else:
                    offspring.extend([p1, p2])

            # 돌연변이
            mutated_offspring = []
            for ind in offspring:
                if random.random() < mutation_prob:
                    if self.encoding_type == 'real':
                        # 세대에 따라 sigma 값 조정
                        current_sigma = mutation_params['sigma'] if generation < self.max_generations // 2 else mutation_params['sigma'] * 0.5
                        current_params = {**mutation_params, 'sigma': current_sigma}
                        mutated = mutation_method(ind.gene_list, **current_params)
                    else:
                        mutated = mutation_method(ind.gene_list, **mutation_params)
                    mutated_offspring.append(
                        RealIndividual(mutated) if self.encoding_type == 'real' else BinaryIndividual(mutated, self.bits)
                    )
                else:
                    mutated_offspring.append(ind)

            # 엘리트 보존
            population = sorted(mutated_offspring, 
                             key=lambda x: func(x.gene_list[0] if self.encoding_type == 'real' else x.x))
            if self.elite_size > 0:
                elite = sorted(population, 
                             key=lambda x: func(x.gene_list[0] if self.encoding_type == 'real' else x.x))[:self.elite_size]
                population = elite + population[:-self.elite_size]

            # 현재 세대의 최소값 갱신
            current_best = min(population, 
                             key=lambda x: func(x.gene_list[0] if self.encoding_type == 'real' else x.x))
            current_fx = func(current_best.gene_list[0] if self.encoding_type == 'real' else current_best.x)

            if current_fx < func(best_ever.gene_list[0] if self.encoding_type == 'real' else best_ever.x):
                best_ever = current_best

            # 수렴 검사
            if abs(current_fx - prev_best_fx) < self.convergence_threshold:
                convergence_count += 1
            else:
                convergence_count = 0
            
            if convergence_count >= self.convergence_generations:
                break
                
            prev_best_fx = current_fx

        return {
            'generations': generation + 1,
            'best_fx': func(best_ever.gene_list[0] if self.encoding_type == 'real' else best_ever.x),
            'best_x': best_ever.gene_list[0] if self.encoding_type == 'real' else best_ever.x,
            'binary': ''.join(map(str, best_ever.gene_list)) if self.encoding_type == 'binary' else None
        }

# 실험 파라미터 세트
encoding_types = ['real', 'binary']
population_sizes = [50, 100, 150]
max_generations_list = [100, 200]
initial_crossover_probs = [0.7, 0.8, 0.9]
final_crossover_probs = [0.5, 0.6, 0.7]
initial_mutation_probs = [0.2, 0.3, 0.4]
final_mutation_probs = [0.05, 0.1, 0.2]
elite_sizes = [1, 3, 5]
bits_list = [16, 20, 24]

# 선택 연산자 목록
selection_methods = [
    selection_proportional,
    selection_rank,
    selection_rank_with_elite,
    selection_tournament,
    selection_stochastic_universal_sampling,
    'hybrid'
]

# 실수 인코딩 교차 연산자 목록
real_crossover_methods = [
    (crossover_blend, {'alpha': 0.5}),
    (crossover_linear, {'alpha': 0.7}),
    (crossover_uniform, {'prop': 0.5})
]

# 실수 인코딩 돌연변이 연산자 목록
real_mutation_methods = [
    (mutation_random_deviation, {'mu': 0, 'sigma': 1.0, 'p': 0.1}),
    (mutation_fitness_driven_random_deviation, {'mu': 0, 'sigma': 1.0, 'p': 0.1, 'max_tries': 3})
]

# 이진 인코딩 교차 연산자 목록
binary_crossover_methods = [
    (crossover_one_point, {}),
    (crossover_n_point, {'n': 2}),
    (crossover_n_point, {'n': 3}),
    (crossover_uniform, {'prop': 0.5})
]

# 이진 인코딩 돌연변이 연산자 목록
binary_mutation_methods = [
    (mutation_bit_flip, {}),
    (mutation_fitness_driven_bit_flip, {'max_tries': 3})
]

# 최적 결과 저장
best_overall = None

# 조합 반복 실행
total_experiments = 0
for encoding in encoding_types:
    for pop_size in population_sizes:
        for max_gen in max_generations_list:
            for init_cp in initial_crossover_probs:
                for final_cp in final_crossover_probs:
                    for init_mp in initial_mutation_probs:
                        for final_mp in final_mutation_probs:
                            for elite in elite_sizes:
                                for bits in bits_list if encoding == 'binary' else [None]:
                                    for selection in selection_methods:
                                        crossover_list = binary_crossover_methods if encoding == 'binary' else real_crossover_methods
                                        mutation_list = binary_mutation_methods if encoding == 'binary' else real_mutation_methods

                                        for crossover, crossover_param in crossover_list:
                                            for mutation, mutation_param in mutation_list:
                                                ga_params = {
                                                    'encoding_type': encoding,
                                                    'population_size': pop_size,
                                                    'max_generations': max_gen,
                                                    'initial_crossover_prob': init_cp,
                                                    'final_crossover_prob': final_cp,
                                                    'initial_mutation_prob': init_mp,
                                                    'final_mutation_prob': final_mp,
                                                    'elite_size': elite,
                                                    'bits': bits if bits else 16
                                                }
                                                ga = AdaptiveGA(**ga_params)

                                                result = ga.run(selection, (crossover, crossover_param), (mutation, mutation_param))

                                                result_summary = {
                                                    'encoding': encoding,
                                                    'selection': selection if isinstance(selection, str) else selection.__name__,
                                                    'crossover': crossover.__name__,
                                                    'mutation': mutation.__name__,
                                                    'crossover_params': crossover_param,
                                                    'mutation_params': mutation_param,
                                                    **result
                                                }

                                                total_experiments += 1

                                                print(f"실험 {total_experiments}: {result_summary['encoding']} / {result_summary['selection']} / {result_summary['crossover']} / {result_summary['mutation']}")
                                                print(f"- f(x): {result_summary['best_fx']:.6f}, x: {result_summary['best_x']:.6f}, 세대 수: {result_summary['generations']}")

                                                if best_overall is None or result_summary['best_fx'] < best_overall['best_fx']:
                                                    best_overall = result_summary

# 최적 결과 출력
print("\n===== 최적 결과 요약 =====")
print(f"- 인코딩: {best_overall['encoding']}")
print(f"- 선택: {best_overall['selection']}")
print(f"- 교차: {best_overall['crossover']} {best_overall['crossover_params']}")
print(f"- 돌연변이: {best_overall['mutation']} {best_overall['mutation_params']}")
print(f"- 최적 x: {best_overall['best_x']:.6f}")
print(f"- 최소 f(x): {best_overall['best_fx']:.6f}")
print(f"- 세대 수: {best_overall['generations']}")
if best_overall['encoding'] == 'binary':
    print(f"- 이진 표현: {best_overall['binary']}")
