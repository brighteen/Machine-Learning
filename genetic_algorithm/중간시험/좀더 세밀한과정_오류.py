import random
import numpy as np
from itertools import product
from datetime import datetime
import os
from toolbox import (
    # 선택 연산자
    selection_proportional, selection_rank, selection_rank_with_elite,
    selection_stochastic_universal_sampling, selection_tournament,
    # 교차 연산자
    crossover_blend, crossover_linear, crossover_uniform,
    crossover_n_point, crossover_one_point, crossover_order, 
    cycle_crossover, crossover_fitness_driven_blend,
    # 돌연변이 연산자
    mutation_random_deviation, mutation_bit_flip,
    mutation_exchange, mutation_shift, mutation_inversion,
    mutation_shuffle, mutation_fitness_driven_random_deviation,
    mutation_fitness_driven_bit_flip
)

def func(x):
    """최소화할 목적 함수: f(x) = 2sin(x) - 0.5x"""
    return 2 * np.sin(x) - 0.5 * x

class RealIndividual:
    """실수 인코딩을 위한 클래스"""
    def __init__(self, gene_list) -> None:
        if isinstance(gene_list, RealIndividual):
            self.gene_list = gene_list.gene_list
        else:
            self.gene_list = gene_list if isinstance(gene_list, list) else [gene_list]
        # 제약조건 적용: -15 ≤ x ≤ 15
        self.gene_list[0] = max(min(self.gene_list[0], 15), -15)
        # 최소값을 찾는 문제이므로 적합도는 함수값의 음수
        self.fitness = -func(self.gene_list[0])

    def __str__(self):
        return f'x: {self.gene_list[0]}, f(x): {func(self.gene_list[0])}'

class BinaryIndividual:
    """이진 인코딩을 위한 클래스"""
    def __init__(self, gene_list=None, bits=16):
        self.bits = bits
        self.min_value = -15
        self.max_value = 15
        
        if gene_list is None:
            # 랜덤 이진 문자열 생성
            self.gene_list = [random.randint(0, 1) for _ in range(bits)]
        else:
            self.gene_list = gene_list
            
        # 이진수를 실수로 디코딩
        self.x = self._decode_binary()
        # 적합도 계산
        self.fitness = -func(self.x)

    def _decode_binary(self):
        """이진 문자열을 실수값으로 변환"""
        # 첫 비트는 부호 비트
        sign = -1 if self.gene_list[0] == 1 else 1
        
        # 나머지 비트로 값 계산
        value = 0
        for i, bit in enumerate(self.gene_list[1:], 1):
            value = value * 2 + bit
            
        # 값을 범위에 맞게 스케일링
        max_binary = 2 ** (self.bits - 1) - 1
        scaled = sign * (value / max_binary) * self.max_value
        
        # 범위 제한
        return max(min(scaled, self.max_value), self.min_value)

    def __str__(self):
        binary = ''.join(map(str, self.gene_list))
        return f'Binary: {binary}, x: {self.x:.4f}, f(x): {func(self.x):.4f}'

# 하이퍼파라미터 설정
population_sizes = [20, 50, 100]
crossover_probabilities = [0.7, 0.8, 0.9]
mutation_probabilities = [0.1, 0.2, 0.3]
max_generations_list = [100, 200]
tournament_sizes = [2, 3, 4]  # tournament 선택용
elite_sizes = [1, 2, 3]  # rank_with_elite 선택용
n_points = [1, 2, 3]  # n-point 교차용
alphas = [0.3, 0.5, 0.7]  # blend, linear 교차용
uniform_props = [0.3, 0.5, 0.7]  # uniform 교차용
mutation_sigmas = [0.5, 1.0, 2.0]  # random deviation 돌연변이용
binary_bits = [16, 20, 24]  # 이진 인코딩 비트 수

# 연산자 리스트
selection_methods = [
    (selection_proportional, {}),
    (selection_rank, {}),
    (selection_rank_with_elite, {'elite_size': 2}),
    (selection_stochastic_universal_sampling, {}),
    (selection_tournament, {'group_size': 3})
]

# 실수 인코딩용 연산자
real_crossover_methods = [
    (crossover_blend, {'alpha': 0.5}),
    (crossover_linear, {'alpha': 0.7}),
    (crossover_uniform, {'prop': 0.5}),
    (crossover_n_point, {'n': 2}),
    (crossover_one_point, {}),
    (cycle_crossover, {}),
    (crossover_order, {}),
    (crossover_fitness_driven_blend, {'alpha': 0.5})
]

real_mutation_methods = [
    (mutation_random_deviation, {'mu': 0, 'sigma': 1.0, 'p': 0.1}),
    (mutation_exchange, {}),
    (mutation_shift, {}),
    (mutation_inversion, {}),
    (mutation_shuffle, {}),
    (mutation_fitness_driven_random_deviation, {'mu': 0, 'sigma': 1.0, 'p': 0.1, 'max_tries': 3})
]

# 이진 인코딩용 연산자
binary_crossover_methods = [
    (crossover_one_point, {}),
    (crossover_n_point, {'n': 2}),
    (crossover_uniform, {'prop': 0.5}),
    (cycle_crossover, {}),
    (crossover_order, {}),
    (crossover_fitness_driven_blend, {'alpha': 0.5})
]

binary_mutation_methods = [
    (mutation_bit_flip, {}),
    (mutation_exchange, {}),
    (mutation_shift, {}),
    (mutation_inversion, {}),
    (mutation_shuffle, {}),
    (mutation_fitness_driven_bit_flip, {'max_tries': 3})
]

# 실험 설정
experiment_settings = []

# 실수 인코딩 설정
for pop_size in population_sizes:
    for cross_prob in crossover_probabilities:
        for mut_prob in mutation_probabilities:
            for max_gen in max_generations_list:
                experiment_settings.append({
                    'encoding_type': 'real',
                    'population_size': pop_size,
                    'max_generations': max_gen,
                    'initial_crossover_prob': cross_prob,
                    'final_crossover_prob': cross_prob * 0.8,
                    'initial_mutation_prob': mut_prob,
                    'final_mutation_prob': mut_prob * 0.5,
                    'elite_size': 3,
                })

# 이진 인코딩 설정
for pop_size in population_sizes:
    for cross_prob in crossover_probabilities:
        for mut_prob in mutation_probabilities:
            for max_gen in max_generations_list:
                for bits in binary_bits:
                    experiment_settings.append({
                        'encoding_type': 'binary',
                        'population_size': pop_size,
                        'max_generations': max_gen,
                        'initial_crossover_prob': cross_prob,
                        'final_crossover_prob': cross_prob * 0.8,
                        'initial_mutation_prob': mut_prob,
                        'final_mutation_prob': mut_prob * 0.5,
                        'elite_size': 3,
                        'bits': bits,
                    })

def run_experiment(encoding_type='real', params=None):
    """실험 실행 함수"""
    if params is None:
        params = {
            'population_size': 20,
            'crossover_prob': 0.8,
            'mutation_prob': 0.2,
            'max_generations': 100,
            'convergence_threshold': 1e-9,
            'convergence_generations': 5,
            'bits': 16,  # 이진 인코딩용
            'selection': selection_methods[0],
            'crossover': real_crossover_methods[0] if encoding_type == 'real' else binary_crossover_methods[0],
            'mutation': real_mutation_methods[0] if encoding_type == 'real' else binary_mutation_methods[0]
        }

    # 초기 개체군 생성
    if encoding_type == 'real':
        population = [
            RealIndividual([random.uniform(-15, 15)])
            for _ in range(params['population_size'])
        ]
    else:
        population = [
            BinaryIndividual(bits=params['bits'])
            for _ in range(params['population_size'])
        ]

    best_ever = min(population, key=lambda x: func(x.gene_list[0] if encoding_type == 'real' else x.x))
    
    # 조기 종료를 위한 변수들
    prev_best_fx = float('inf')
    convergence_count = 0
    
    for generation in range(params['max_generations']):
        # 선택
        selection_method, selection_params = params['selection']
        selected = selection_method(population, **selection_params)

        # 교차
        offspring = []
        crossover_method, crossover_params = params['crossover']
        
        for p1, p2 in zip(selected[::2], selected[1::2]):
            if random.random() < params['crossover_prob']:
                c1, c2 = crossover_method(p1.gene_list, p2.gene_list, **crossover_params)
                offspring.extend([
                    RealIndividual(c1) if encoding_type == 'real' else BinaryIndividual(c1, params['bits']),
                    RealIndividual(c2) if encoding_type == 'real' else BinaryIndividual(c2, params['bits'])
                ])
            else:
                offspring.extend([p1, p2])

        # 돌연변이
        mutation_method, mutation_params = params['mutation']
        mutated_offspring = []
        
        for ind in offspring:
            if random.random() < params['mutation_prob']:
                mutated = mutation_method(ind.gene_list, **mutation_params)
                mutated_offspring.append(
                    RealIndividual(mutated) if encoding_type == 'real' else BinaryIndividual(mutated, params['bits'])
                )
            else:
                mutated_offspring.append(ind)

        # 새로운 세대 설정
        population = mutated_offspring
        
        # 현재 세대의 최소값 갱신
        current_best = min(population, key=lambda x: func(x.gene_list[0] if encoding_type == 'real' else x.x))
        current_fx = func(current_best.gene_list[0] if encoding_type == 'real' else current_best.x)
        
        if current_fx < func(best_ever.gene_list[0] if encoding_type == 'real' else best_ever.x):
            best_ever = current_best

        # 수렴 검사
        if abs(current_fx - prev_best_fx) < params['convergence_threshold']:
            convergence_count += 1
        else:
            convergence_count = 0
        
        if convergence_count >= params['convergence_generations']:
            break
            
        prev_best_fx = current_fx

    return {
        'encoding': encoding_type,
        'generations': generation + 1,
        'selection': params['selection'][0].__name__,
        'crossover': params['crossover'][0].__name__,
        'mutation': params['mutation'][0].__name__,
        'population_size': params['population_size'],
        'crossover_prob': params['crossover_prob'],
        'mutation_prob': params['mutation_prob'],
        'bits': params['bits'] if encoding_type == 'binary' else None,
        'best_fx': func(best_ever.gene_list[0] if encoding_type == 'real' else best_ever.x),
        'best_x': best_ever.gene_list[0] if encoding_type == 'real' else best_ever.x,
        'binary': ''.join(map(str, best_ever.gene_list)) if encoding_type == 'binary' else None
    }

def save_results(results, encoding_type):
    """결과를 markdown 파일로 저장"""
    result_dir = r"C:\Users\brigh\Documents\GitHub\Machine-Learning\genetic_algorithm\중간시험\결과"
    
    # 다음 결과 파일 번호 찾기
    files = [f for f in os.listdir(result_dir) if f.startswith('ga_results_') and f.endswith('.md')]
    result_number = 1 if not files else max(int(f.split('_')[2].split('.')[0]) for f in files) + 1
    
    filename = os.path.join(result_dir, f'ga_results_{result_number}.md')
    
    # 최종 결과를 markdown 파일로 저장
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# 유전 알고리즘 하이퍼파라미터 실험 결과 #{result_number} ({encoding_type} 인코딩)\n\n")
        f.write(f"실험 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 실험 설정\n")
        f.write(f"- 인구 크기 범위: {population_sizes}\n")
        f.write(f"- 교차 확률 범위: {crossover_probabilities}\n")
        f.write(f"- 돌연변이 확률 범위: {mutation_probabilities}\n")
        f.write(f"- 최대 세대 수: {max_generations_list}\n")
        f.write(f"- 수렴 임계값: 1e-9\n")
        f.write(f"- 수렴 판단 세대 수: 5\n")
        if encoding_type == 'binary':
            f.write(f"- 이진 표현 비트 수 범위: {binary_bits}\n")
        f.write("\n## 실험 결과\n\n")
        
        # 결과를 최소값 기준으로 정렬
        sorted_results = sorted(results, key=lambda x: x['best_fx'])
        
        for i, result in enumerate(sorted_results, 1):
            f.write(f"### 실험 {i}\n")
            f.write(f"- 연산자 조합:\n")
            f.write(f"  - 선택: {result['selection']}\n")
            f.write(f"  - 교차: {result['crossover']}\n")
            f.write(f"  - 돌연변이: {result['mutation']}\n")
            f.write(f"- 파라미터:\n")
            f.write(f"  - 인구 크기: {result['population_size']}\n")
            f.write(f"  - 교차 확률: {result['crossover_prob']:.1f}\n")
            f.write(f"  - 돌연변이 확률: {result['mutation_prob']:.1f}\n")
            if encoding_type == 'binary':
                f.write(f"  - 비트 수: {result['bits']}\n")
            f.write(f"- 결과:\n")
            f.write(f"  - 수렴 세대 수: {result['generations']}\n")
            f.write(f"  - 최소값 f(x): {result['best_fx']:.6f}\n")
            f.write(f"  - 최적 x: {result['best_x']:.6f}\n")
            if encoding_type == 'binary':
                f.write(f"  - 이진 표현: {result['binary']}\n")
            f.write("\n")
        
        # 통계 분석
        avg_generations = sum(r['generations'] for r in results) / len(results)
        best_result = min(results, key=lambda x: x['best_fx'])
        
        f.write("## 통계 분석\n")
        f.write(f"- 평균 수렴 세대 수: {avg_generations:.2f}\n")
        f.write(f"- 전체 최소값: {best_result['best_fx']:.6f}\n")
        f.write("- 최적의 파라미터 조합:\n")
        f.write(f"  - 선택 연산자: {best_result['selection']}\n")
        f.write(f"  - 교차 연산자: {best_result['crossover']}\n")
        f.write(f"  - 돌연변이 연산자: {best_result['mutation']}\n")
        f.write(f"  - 인구 크기: {best_result['population_size']}\n")
        f.write(f"  - 교차 확률: {best_result['crossover_prob']}\n")
        f.write(f"  - 돌연변이 확률: {best_result['mutation_prob']}\n")
        if encoding_type == 'binary':
            f.write(f"  - 비트 수: {best_result['bits']}\n")
        f.write(f"  - 최적 x: {best_result['best_x']:.6f}\n")
        if encoding_type == 'binary':
            f.write(f"  - 이진 표현: {best_result['binary']}")

    print(f"\n결과가 {filename} 파일에 저장되었습니다.")

# 실수 인코딩 실험 실행
print("실수 인코딩 실험 시작...")
real_results = []

# 실수 인코딩 실험을 위한 파라미터 조합
for pop_size, cross_prob, mut_prob in product(population_sizes[:2], crossover_probabilities[:2], mutation_probabilities[:2]):
    for selection in selection_methods[:2]:
        for crossover in real_crossover_methods:
            for mutation in real_mutation_methods:
                params = {
                    'population_size': pop_size,
                    'crossover_prob': cross_prob,
                    'mutation_prob': mut_prob,
                    'max_generations': 100,
                    'convergence_threshold': 1e-9,
                    'convergence_generations': 5,
                    'selection': selection,
                    'crossover': crossover,
                    'mutation': mutation
                }
                
                print(f"\n실행 중:")
                print(f"- 연산자: {selection[0].__name__} / {crossover[0].__name__} / {mutation[0].__name__}")
                print(f"- 파라미터: 인구={pop_size}, 교차={cross_prob}, 돌연변이={mut_prob}")
                
                result = run_experiment('real', params)
                real_results.append(result)
                
                print(f"- 결과: f(x)={result['best_fx']:.6f}, x={result['best_x']:.6f}")

save_results(real_results, 'real')

# 이진 인코딩 실험 실행
print("\n이진 인코딩 실험 시작...")
binary_results = []

# 이진 인코딩 실험을 위한 파라미터 조합
for pop_size, cross_prob, mut_prob, bits in product(
    population_sizes[:2], crossover_probabilities[:2],
    mutation_probabilities[:2], binary_bits[:2]
):
    for selection in selection_methods[:2]:
        for crossover in binary_crossover_methods:
            for mutation in binary_mutation_methods:
                params = {
                    'population_size': pop_size,
                    'crossover_prob': cross_prob,
                    'mutation_prob': mut_prob,
                    'max_generations': 100,
                    'convergence_threshold': 1e-9,
                    'convergence_generations': 5,
                    'bits': bits,
                    'selection': selection,
                    'crossover': crossover,
                    'mutation': mutation
                }
                
                print(f"\n실행 중:")
                print(f"- 연산자: {selection[0].__name__} / {crossover[0].__name__} / {mutation[0].__name__}")
                print(f"- 파라미터: 인구={pop_size}, 교차={cross_prob}, 돌연변이={mut_prob}, 비트={bits}")
                
                result = run_experiment('binary', params)
                binary_results.append(result)
                
                print(f"- 결과: f(x)={result['best_fx']:.6f}, x={result['best_x']:.6f}")

save_results(binary_results, 'binary')