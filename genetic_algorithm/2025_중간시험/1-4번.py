# Jupyter Notebook Cell 2: Main Execution Logic
# 개체 클래스, GARunner, 멀티프로세싱 실행, 파라미터 그리드 및 결과 저장

import random
import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime
import os
import multiprocessing
import time

from toolbox import (selection_proportional, selection_rank, selection_rank_with_elite, selection_stochastic_universal_sampling, selection_tournament, 
                     crossover_blend, crossover_linear, crossover_n_point, crossover_one_point, crossover_uniform,
                     mutation_random_deviation, mutation_exchange, mutation_shift, mutation_bit_flip, mutation_inversion, mutation_shuffle, 
                     mutation_fitness_driven_random_deviation, mutation_fitness_driven_bit_flip, 
    )
# Cell 1에서 정의된 함수들은 이 셀에서 바로 사용 가능합니다.
# 예: selection_proportional, crossover_blend, mutation_random_deviation 등

# 목적 함수 정의
def func(x):
    """최소화할 목적 함수: f(x) = 2sin(x) + 0.5x"""
    x = float(x)
    return 2 * np.sin(x) + 0.5 * x

# 개체 표현 클래스 정의
class RealIndividual:
    """실수 인코딩 개체"""
    def __init__(self, gene_list, min_value=-5, max_value=13):
        if isinstance(gene_list, RealIndividual):
            self.gene_list = [float(g) for g in gene_list.gene_list]
        elif isinstance(gene_list, (int, float)):
             self.gene_list = [float(gene_list)]
        elif isinstance(gene_list, list):
            self.gene_list = [float(g) for g in gene_list]
        else:
             raise TypeError(f"Unsupported type for gene_list: {type(gene_list)}")

        self.min_value = float(min_value)
        self.max_value = float(max_value)

        if self.gene_list:
             self.gene_list[0] = max(min(self.gene_list[0], self.max_value), self.min_value)
        else:
             self.gene_list = [random.uniform(self.min_value, self.max_value)]

        self.fitness = -func(self.gene_list[0]) # 최소화 문제 -> 적합도는 음수

    def get_x(self):
        """디코딩된 x 값 반환"""
        return self.gene_list[0]

    def __str__(self):
        return f'x: {self.get_x():.4f}, f(x): {func(self.get_x()):.4f}, Fitness: {self.fitness:.4f}'

    def __repr__(self):
         return self.__str__()


class BinaryIndividual:
    """이진 인코딩 개체"""
    def __init__(self, gene_list=None, bits=20, min_value=-5, max_value=13):
        self.bits = int(bits)
        self.min_value = float(min_value)
        self.max_value = float(max_value)

        if gene_list is None:
            self.gene_list = [random.randint(0, 1) for _ in range(self.bits)]
        else:
            if isinstance(gene_list, str):
                 self.gene_list = [int(bit) for bit in gene_list]
            elif isinstance(gene_list, list):
                 self.gene_list = [int(bit) for bit in gene_list]
            else:
                 raise TypeError(f"Unsupported type for gene_list: {type(gene_list)}")

            if len(self.gene_list) != self.bits:
                 raise ValueError(f"Gene list length ({len(self.gene_list)}) must match specified bits ({self.bits})")

        self.x = self._decode_binary()
        self.fitness = -func(self.x) # 최소화 문제 -> 적합도는 음수

    def _decode_binary(self):
        """이진 문자열을 실수 값으로 디코딩 (선형 매핑)"""
        if not self.gene_list:
             return self.min_value

        integer_value = 0
        for bit in self.gene_list:
            integer_value = integer_value * 2 + bit

        max_integer_value = (2 ** self.bits) - 1
        if max_integer_value == 0:
             return self.min_value

        range_size = self.max_value - self.min_value
        decoded_x = self.min_value + integer_value * (range_size / max_integer_value)

        return max(min(decoded_x, self.max_value), self.min_value)

    def get_x(self):
        """디코딩된 x 값 반환"""
        return self.x

    def __str__(self):
        binary = ''.join(map(str, self.gene_list))
        return f'Binary: {binary}, x: {self.get_x():.4f}, f(x): {func(self.get_x()):.4f}, Fitness: {self.fitness:.4f}'

    def __repr__(self):
         return self.__str__()


# GARunner 클래스 정의
class GARunner:
    """고정된 파라미터로 단일 GA 실행"""
    def __init__(self, params):
        self.params = params
        self.encoding_type = params['encoding_type']
        self.population_size = params['population_size']
        self.max_generations = params['max_generations']
        self.initial_crossover_prob = params['initial_crossover_prob']
        self.final_crossover_prob = params['final_crossover_prob']
        self.initial_mutation_prob = params['initial_mutation_prob']
        self.final_mutation_prob = params['final_mutation_prob']
        self.elite_size = params['elite_size']
        self.tournament_group_size = params['tournament_group_size']
        self.bits = params.get('bits', None)

        self.convergence_threshold = params.get('convergence_threshold', 1e-6)
        self.convergence_generations = params.get('convergence_generations', 10)

        self.selection_method = params['selection_method']
        self.crossover_method, self.crossover_params = params['crossover_tuple']
        self.mutation_method, self.mutation_params = params['mutation_tuple']

        self.IndividualClass = RealIndividual if self.encoding_type == 'real' else BinaryIndividual

        self.min_value = -5.0
        self.max_value = 13.0


    def get_adaptive_rates(self, generation):
        """세대 진행에 따른 적응적 확률 계산"""
        if self.max_generations <= 1:
             progress = 0
        else:
             progress = generation / (self.max_generations - 1)

        crossover_prob = self.initial_crossover_prob - (self.initial_crossover_prob - self.final_crossover_prob) * progress
        mutation_prob = self.initial_mutation_prob - (self.initial_mutation_prob - self.final_mutation_prob) * progress
        return crossover_prob, mutation_prob

    def run(self):
        """GA 실행"""
        # 개체군 초기화
        population = []
        for _ in range(self.population_size):
            if self.encoding_type == 'binary':
                population.append(self.IndividualClass(bits=self.bits, min_value=self.min_value, max_value=self.max_value))
            else:
                population.append(self.IndividualClass(random.uniform(self.min_value, self.max_value), min_value=self.min_value, max_value=self.max_value))

        # 전체 세대 중 최고의 개체 추적
        best_ever = max(population, key=lambda ind: ind.fitness)
        if self.encoding_type == 'binary':
            best_ever = self.IndividualClass(best_ever.gene_list, bits=self.bits, min_value=self.min_value, max_value=self.max_value)
        else:
            best_ever = self.IndividualClass(best_ever.gene_list, min_value=self.min_value, max_value=self.max_value)

        prev_best_fx = float('inf')
        convergence_count = 0
        generations_run = 0

        for generation in range(self.max_generations):
            generations_run = generation + 1

            crossover_prob, mutation_prob = self.get_adaptive_rates(generation)

            # 선택
            if self.selection_method == selection_rank_with_elite:
                 current_elite_size = min(self.elite_size, self.population_size)
                 selected = self.selection_method(population, elite_size=current_elite_size)
            elif self.selection_method == selection_tournament:
                 current_group_size = min(self.tournament_group_size, self.population_size)
                 selected = self.selection_method(population, group_size=current_group_size)
            else:
                 selected = self.selection_method(population)

            if len(selected) != self.population_size:
                 selected = random.sample(selected, self.population_size)

            # 교차
            offspring = []
            random.shuffle(selected)
            paired_selected = selected
            if len(paired_selected) % 2 != 0:
                 paired_selected = paired_selected[:-1]

            for i in range(0, len(paired_selected), 2):
                p1, p2 = paired_selected[i], paired_selected[i+1]
                if random.random() < crossover_prob:
                    c1_genes, c2_genes = self.crossover_method(p1.gene_list, p2.gene_list, **self.crossover_params)
                    if self.encoding_type == 'binary':
                        offspring.append(self.IndividualClass(c1_genes, bits=self.bits, min_value=self.min_value, max_value=self.max_value))
                        offspring.append(self.IndividualClass(c2_genes, bits=self.bits, min_value=self.min_value, max_value=self.max_value))
                    else:
                        offspring.append(self.IndividualClass(c1_genes, min_value=self.min_value, max_value=self.max_value))
                        offspring.append(self.IndividualClass(c2_genes, min_value=self.min_value, max_value=self.max_value))
                else:
                    if self.encoding_type == 'binary':
                        offspring.append(self.IndividualClass(p1.gene_list, bits=self.bits, min_value=self.min_value, max_value=self.max_value))
                        offspring.append(self.IndividualClass(p2.gene_list, bits=self.bits, min_value=self.min_value, max_value=self.max_value))
                    else:
                        offspring.append(self.IndividualClass(p1.gene_list, min_value=self.min_value, max_value=self.max_value))
                        offspring.append(self.IndividualClass(p2.gene_list, min_value=self.min_value, max_value=self.max_value))

            if len(selected) % 2 != 0:
                 if self.encoding_type == 'binary':
                     offspring.append(self.IndividualClass(selected[-1].gene_list, bits=self.bits, min_value=self.min_value, max_value=self.max_value))
                 else:
                     offspring.append(self.IndividualClass(selected[-1].gene_list, min_value=self.min_value, max_value=self.max_value))

            # 돌연변이
            mutated_offspring = []
            for ind in offspring:
                 if random.random() < mutation_prob:
                    # 적합도 기반 돌연변이는 개체 자체를 반환
                    if self.mutation_method in [mutation_fitness_driven_random_deviation, mutation_fitness_driven_bit_flip]:
                         mutated_ind = self.mutation_method(ind, **self.mutation_params)
                         mutated_offspring.append(mutated_ind)
                    # 일반 돌연변이는 유전자 리스트를 반환
                    else:
                         mutated_genes = self.mutation_method(ind.gene_list, **self.mutation_params)
                         if self.encoding_type == 'binary':
                             mutated_offspring.append(self.IndividualClass(mutated_genes, bits=self.bits, min_value=self.min_value, max_value=self.max_value))
                         else:
                             mutated_offspring.append(self.IndividualClass(mutated_genes, min_value=self.min_value, max_value=self.max_value))
                 else:
                    # 돌연변이 없음, 개체 복사
                    if self.encoding_type == 'binary':
                        mutated_offspring.append(self.IndividualClass(ind.gene_list, bits=self.bits, min_value=self.min_value, max_value=self.max_value))
                    else:
                        mutated_offspring.append(self.IndividualClass(ind.gene_list, min_value=self.min_value, max_value=self.max_value))

            # 세대 교체 및 엘리트 보존
            current_population_best = max(population, key=lambda ind: ind.fitness)
            if current_population_best.fitness > best_ever.fitness:
                 if self.encoding_type == 'binary':
                     best_ever = self.IndividualClass(current_population_best.gene_list, bits=self.bits, min_value=self.min_value, max_value=self.max_value)
                 else:
                     best_ever = self.IndividualClass(current_population_best.gene_list, min_value=self.min_value, max_value=self.max_value)

            mutated_offspring.sort(key=lambda ind: ind.fitness, reverse=True)
            next_population = mutated_offspring[:self.population_size]

            if self.elite_size > 0 and best_ever not in next_population:
                 if next_population:
                      worst_in_next_pop = next_population[-1]
                      if best_ever.fitness > worst_in_next_pop.fitness:
                          if self.encoding_type == 'binary':
                              next_population[-1] = self.IndividualClass(best_ever.gene_list, bits=self.bits, min_value=self.min_value, max_value=self.max_value)
                          else:
                              next_population[-1] = self.IndividualClass(best_ever.gene_list, min_value=self.min_value, max_value=self.max_value)

            while len(next_population) < self.population_size:
                 if self.encoding_type == 'binary':
                     next_population.append(self.IndividualClass(best_ever.gene_list, bits=self.bits, min_value=self.min_value, max_value=self.max_value))
                 else:
                     next_population.append(self.IndividualClass(best_ever.gene_list, min_value=self.min_value, max_value=self.max_value))

            population = next_population

            # 수렴 확인
            current_best_in_pop = population[0]
            current_fx = func(current_best_in_pop.get_x())

            if current_best_in_pop.fitness > best_ever.fitness:
                 if self.encoding_type == 'binary':
                     best_ever = self.IndividualClass(current_best_in_pop.gene_list, bits=self.bits, min_value=self.min_value, max_value=self.max_value)
                 else:
                     best_ever = self.IndividualClass(current_best_in_pop.gene_list, min_value=self.min_value, max_value=self.max_value)

            if abs(current_fx - prev_best_fx) < self.convergence_threshold:
                convergence_count += 1
            else:
                convergence_count = 0

            if convergence_count >= self.convergence_generations:
                break

            prev_best_fx = current_fx

        final_best_fx = func(best_ever.get_x())

        return {
            'generations_run': generations_run,
            'best_fx': final_best_fx,
            'best_x': best_ever.get_x(),
            'encoding_type': self.encoding_type,
            'population_size': self.population_size,
            'max_generations': self.max_generations,
            'initial_crossover_prob': self.initial_crossover_prob,
            'final_crossover_prob': self.final_crossover_prob,
            'initial_mutation_prob': self.initial_mutation_prob,
            'final_mutation_prob': self.final_mutation_prob,
            'elite_size': self.elite_size,
            'tournament_group_size': self.tournament_group_size,
            'bits': self.bits,
            'selection_method': self.selection_method.__name__ if callable(self.selection_method) else str(self.selection_method),
            'crossover_method': self.crossover_method.__name__,
            'crossover_params': str(self.crossover_params),
            'mutation_method': self.mutation_method.__name__,
            'mutation_params': str(self.mutation_params),
            'convergence_threshold': self.convergence_threshold,
            'convergence_generations': self.convergence_generations,
            'best_binary_repr': ''.join(map(str, best_ever.gene_list)) if self.encoding_type == 'binary' and best_ever.gene_list is not None else None,
        }


# 멀티프로세싱 워커 함수
def run_single_trial(params_with_seed):
    """단일 GA 실행을 위한 워커 함수"""
    run_num = params_with_seed['run_number']
    random_seed = params_with_seed['random_seed']

    random.seed(random_seed)
    np.random.seed(random_seed)

    try:
        # GARunner에 필요한 파라미터만 추출
        runner_params = {k: v for k, v in params_with_seed.items() if k not in ['run_number', 'random_seed']}
        runner = GARunner(runner_params)
        result = runner.run()

        result['run_number'] = run_num + 1
        result['random_seed'] = random_seed
        return result

    except Exception as e:
        error_result = params_with_seed.copy()
        error_result['run_number'] = run_num + 1
        error_result['error'] = str(e)
        error_result['generations_run'] = 0
        error_result['best_fx'] = float('nan')
        error_result['best_x'] = float('nan')
        error_result['best_binary_repr'] = None

        # 오류 로깅을 위해 함수/튜플을 문자열로 변환
        error_result['selection_method'] = error_result['selection_method'].__name__ if callable(error_result['selection_method']) else str(error_result['selection_method'])
        error_result['crossover_method'] = error_result['crossover_tuple'][0].__name__ if isinstance(error_result['crossover_tuple'], tuple) and callable(error_result['crossover_tuple'][0]) else str(error_result['crossover_tuple'])
        error_result['crossover_params'] = str(error_result['crossover_tuple'][1]) if isinstance(error_result['crossover_tuple'], tuple) else str(error_result['crossover_tuple'])
        error_result['mutation_method'] = error_result['mutation_tuple'][0].__name__ if isinstance(error_result['mutation_tuple'], tuple) and callable(error_result['mutation_tuple'][0]) else str(error_result['mutation_tuple'])
        error_result['mutation_params'] = str(error_result['mutation_tuple'][1]) if isinstance(error_result['mutation_tuple'], tuple) else str(error_result['mutation_tuple'])

        if 'selection_method' in error_result and callable(error_result['selection_method']): del error_result['selection_method']
        if 'crossover_tuple' in error_result: del error_result['crossover_tuple']
        if 'mutation_tuple' in error_result: del error_result['mutation_tuple']

        return error_result


# --- 파라미터 그리드 정의 ---

# 각 파라미터에 대해 테스트할 후보 값 정의

# 일반 GA 파라미터
population_sizes = [50, 100, 150]
max_generations_list = [100, 200, 300]
crossover_prob_ranges = [(0.9, 0.7), (0.9, 0.9), (0.6, 0.6)] # 적응적, 고정
mutation_prob_ranges = [(0.3, 0.1), (0.1, 0.1), (0.01, 0.01)] # 적응적, 고정
elite_sizes = [0, 1, 5]
tournament_group_sizes = [2, 3, 5]

# 이진 인코딩 특정 파라미터
binary_bits_list = [16, 24, 32]

# 연산자 옵션 (함수, {파라미터}) - 인코딩별 분리

# 선택 연산자 (함수 직접 전달)
selection_options = [
    selection_proportional,
    selection_rank,
    selection_rank_with_elite,
    selection_stochastic_universal_sampling,
    selection_tournament,
]

# 실수값 교차 옵션
real_crossover_options = [
    (crossover_blend, {'alpha': 0.0}),
    (crossover_blend, {'alpha': 0.5}),
    (crossover_blend, {'alpha': 1.0}),
    (crossover_blend, {'alpha': 1.5}),
    (crossover_linear, {'alpha': 0.5}),
]

# 실수값 돌연변이 옵션
real_mutation_options = [
    (mutation_random_deviation, {'mu': 0, 'sigma': 0.1, 'p': 1.0}),
    (mutation_random_deviation, {'mu': 0, 'sigma': 0.5, 'p': 1.0}),
    (mutation_random_deviation, {'mu': 0, 'sigma': 1.0, 'p': 1.0}),
    (mutation_random_deviation, {'mu': 0, 'sigma': 0.5, 'p': 0.5}),
    (mutation_fitness_driven_random_deviation, {'mu': 0, 'sigma': 1.0, 'p': 1.0, 'max_tries': 5}),
]

# 이진 교차 옵션
binary_crossover_options = [
    (crossover_one_point, {}),
    (crossover_n_point, {'n': 2}),
    (crossover_n_point, {'n': 5}),
    (crossover_uniform, {'prop': 0.5}),
    (crossover_uniform, {'prop': 0.8}),
]

# 이진 돌연변이 옵션
binary_mutation_options = [
    (mutation_bit_flip, {}),
    (mutation_exchange, {}),
    (mutation_shift, {}),
    (mutation_inversion, {}),
    (mutation_shuffle, {}),
    (mutation_fitness_driven_bit_flip, {'max_tries': 5}),
]


# --- 모든 파라미터 조합 생성 ---

all_combinations_params = []

general_params_product = product(
    population_sizes,
    max_generations_list,
    crossover_prob_ranges,
    mutation_prob_ranges,
    elite_sizes,
    tournament_group_sizes,
    selection_options,
)

for (pop_size, max_gen, (init_cx, final_cx), (init_mut, final_mut), elite_sz, tourn_sz, sel_method) in general_params_product:
    for encoding in ['real', 'binary']:
        if encoding == 'real':
            current_crossover_options = real_crossover_options
            current_mutation_options = real_mutation_options
            current_bits_list = [None]
        else:
            current_crossover_options = binary_crossover_options
            current_mutation_options = binary_mutation_options
            current_bits_list = binary_bits_list

        encoding_specific_product = product(
            current_bits_list,
            current_crossover_options,
            current_mutation_options,
        )

        for (bits, cx_tuple, mut_tuple) in encoding_specific_product:
            combo_params = {
                'encoding_type': encoding,
                'population_size': pop_size,
                'max_generations': max_gen,
                'initial_crossover_prob': init_cx,
                'final_crossover_prob': final_cx,
                'initial_mutation_prob': init_mut,
                'final_mutation_prob': final_mut,
                'elite_size': elite_sz,
                'tournament_group_size': tourn_sz,
                'bits': bits,
                'selection_method': sel_method, # 함수 객체 저장
                'crossover_tuple': cx_tuple, # (함수 객체, 파라미터 dict) 튜플 저장
                'mutation_tuple': mut_tuple, # (함수 객체, 파라미터 dict) 튜플 저장
            }
            all_combinations_params.append(combo_params)

print(f"생성된 고유 파라미터 조합 수: {len(all_combinations_params)}")


# --- 멀티프로세싱을 위한 작업 준비 ---

num_runs_per_combination = 10
tasks = []
base_seed_counter = int(datetime.now().timestamp())

for i, params in enumerate(all_combinations_params):
    combination_base_seed = base_seed_counter + i * num_runs_per_combination
    for run_num in range(num_runs_per_combination):
        trial_params = params.copy()
        trial_params['run_number'] = run_num
        trial_params['random_seed'] = combination_base_seed + run_num
        tasks.append(trial_params)

print(f"총 실행할 GA 시뮬레이션 수: {len(tasks)}")


# --- 멀티프로세싱을 사용하여 실험 실행 ---

num_processes = multiprocessing.cpu_count()
print(f"\n{num_processes}개의 CPU 코어를 사용하여 실험 실행...")

all_results = []
start_time = time.time()

with multiprocessing.Pool(processes=num_processes) as pool:
    # imap_unordered로 비동기 실행 및 결과 수집
    results_iterator = pool.imap_unordered(run_single_trial, tasks, chunksize=10)

    # 결과 도착 시 처리 및 진행 상황 출력
    for i, result in enumerate(results_iterator):
        all_results.append(result)
        if (i + 1) % 100 == 0 or (i + 1) == len(tasks):
            elapsed_time = time.time() - start_time
            avg_time_per_run = elapsed_time / (i + 1) if (i + 1) > 0 else 0
            estimated_total_time = avg_time_per_run * len(tasks)
            remaining_time = estimated_total_time - elapsed_time
            print(f"  처리 완료: {i + 1}/{len(tasks)} | 경과 시간: {elapsed_time:.2f}s | 남은 시간 추정: {remaining_time:.2f}s")

print("\n실험 완료.")

# --- 결과 저장 ---

result_dir = "ga_optimization_results"
os.makedirs(result_dir, exist_ok=True)

df = pd.DataFrame(all_results)
filename = os.path.join(result_dir, f"ga_results_multiprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
df.to_csv(filename, index=False, encoding='utf-8-sig')

print(f"\n결과가 {filename} 파일에 저장되었습니다.")
