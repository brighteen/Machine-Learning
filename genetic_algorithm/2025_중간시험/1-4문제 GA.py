import random
import numpy as np
from itertools import product
from datetime import datetime
import os
from toolbox import (
    selection_proportional, selection_rank, selection_rank_with_elite,
    selection_stochastic_universal_sampling, selection_tournament,
    crossover_blend, crossover_linear, crossover_uniform,
    crossover_n_point, crossover_one_point,
    mutation_random_deviation, mutation_bit_flip,
    mutation_fitness_driven_random_deviation, mutation_fitness_driven_bit_flip
)

def func(x):
    """최소화할 목적 함수: f(x) = 2sin(x) + 0.5x"""
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
            return RealIndividual([random.uniform(-15, 15)])
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

# 실험 설정
experiment_settings = [
    {
        'encoding_type': 'real',
        'population_size': 100,
        'max_generations': 200,
        'initial_crossover_prob': 0.9,
        'final_crossover_prob': 0.7,
        'initial_mutation_prob': 0.3,
        'final_mutation_prob': 0.1,
        'elite_size': 3,
    },
    {
        'encoding_type': 'binary',
        'population_size': 100,
        'max_generations': 200,
        'initial_crossover_prob': 0.9,
        'final_crossover_prob': 0.7,
        'initial_mutation_prob': 0.3,
        'final_mutation_prob': 0.1,
        'elite_size': 3,
        'bits': 24,
    }
]

# 실험 실행
for setting in experiment_settings:
    print(f"\n{setting['encoding_type'].upper()} 인코딩 실험 시작...")
    
    ga = AdaptiveGA(**setting)
    results = []
    
    # 실수 인코딩용 연산자 조합
    if setting['encoding_type'] == 'real':
        real_crossover_methods = [
            (crossover_blend, {'alpha': 0.5}),
            (crossover_linear, {'alpha': 0.7}),
            (crossover_uniform, {'prop': 0.5})
        ]
        real_mutation_methods = [
            (mutation_random_deviation, {'mu': 0, 'sigma': 1.0, 'p': 0.1}),
            (mutation_fitness_driven_random_deviation, {'mu': 0, 'sigma': 1.0, 'p': 0.1, 'max_tries': 3})
        ]
        combinations = [
            (selection_rank_with_elite, real_crossover_methods[0], real_mutation_methods[0]),
            (selection_tournament, real_crossover_methods[1], real_mutation_methods[0]),
            ('hybrid', real_crossover_methods[2], real_mutation_methods[1]),
        ]
    # 이진 인코딩용 연산자 조합
    else:
        binary_crossover_methods = [
            (crossover_one_point, {}),
            (crossover_n_point, {'n': 2}),
            (crossover_uniform, {'prop': 0.5})
        ]
        binary_mutation_methods = [
            (mutation_bit_flip, {}),
            (mutation_fitness_driven_bit_flip, {'max_tries': 3})
        ]
        combinations = [
            (selection_rank_with_elite, binary_crossover_methods[0], binary_mutation_methods[0]),
            (selection_tournament, binary_crossover_methods[1], binary_mutation_methods[0]),
            ('hybrid', binary_crossover_methods[2], binary_mutation_methods[1]),
        ]
    
    for selection, crossover, mutation in combinations:
        print(f"\n실행 중:")
        print(f"- 연산자: {selection if isinstance(selection, str) else selection.__name__} / {crossover[0].__name__} / {mutation[0].__name__}")
        
        result = ga.run(selection, crossover, mutation)
        results.append({
            'selection': selection if isinstance(selection, str) else selection.__name__,
            'crossover': crossover[0].__name__,
            'mutation': mutation[0].__name__,
            **result
        })
        
        print(f"- 결과: 세대 수={result['generations']}, f(x)={result['best_fx']:.6f}, x={result['best_x']:.6f}")
        if setting['encoding_type'] == 'binary':
            print(f"- 이진 표현: {result['binary']}")

    # 결과 저장
    result_dir = r"C:\Users\brigh\Documents\GitHub\Machine-Learning\genetic_algorithm\2025_중간시험\1-4.문제_결과"
    
    # 기존 파일에서 가장 큰 번호 찾기
    def get_next_number(files):
        max_num = 0
        for f in files:
            if f.startswith('개선된_ga_results_'):
                try:
                    # 파일 이름에서 숫자만 추출
                    num_str = ''.join(c for c in f.split('_')[2] if c.isdigit())
                    if num_str:
                        max_num = max(max_num, int(num_str))
                except ValueError:
                    continue
        return max_num + 1
    
    files = [f for f in os.listdir(result_dir) if f.endswith('.md')]
    result_number = get_next_number(files)
    
    filename = os.path.join(result_dir, f'개선된_ga_results_{result_number}_{setting["encoding_type"]}.md')
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# 개선된 유전 알고리즘 실험 결과 #{result_number} ({setting['encoding_type']} 인코딩)\n\n")
        f.write(f"실험 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 실험 설정\n")
        for key, value in setting.items():
            f.write(f"- {key}: {value}\n")
        f.write("\n## 실험 결과\n\n")
        
        sorted_results = sorted(results, key=lambda x: x['best_fx'])
        
        for i, result in enumerate(sorted_results, 1):
            f.write(f"### 실험 {i}\n")
            f.write(f"- 연산자 조합:\n")
            f.write(f"  - 선택: {result['selection']}\n")
            f.write(f"  - 교차: {result['crossover']}\n")
            f.write(f"  - 돌연변이: {result['mutation']}\n")
            f.write(f"- 결과:\n")
            f.write(f"  - 수렴 세대 수: {result['generations']}\n")
            f.write(f"  - 최소값 f(x): {result['best_fx']:.6f}\n")
            f.write(f"  - 최적 x: {result['best_x']:.6f}\n")
            if setting['encoding_type'] == 'binary':
                f.write(f"  - 이진 표현: {result['binary']}\n")
            f.write("\n")
        
        avg_generations = sum(r['generations'] for r in results) / len(results)
        best_result = min(results, key=lambda x: x['best_fx'])
        
        f.write("## 통계 분석\n")
        f.write(f"- 평균 수렴 세대 수: {avg_generations:.2f}\n")
        f.write(f"- 전체 최소값: {best_result['best_fx']:.6f}\n")
        f.write("- 최적의 연산자 조합:\n")
        f.write(f"  - 선택 연산자: {best_result['selection']}\n")
        f.write(f"  - 교차 연산자: {best_result['crossover']}\n")
        f.write(f"  - 돌연변이 연산자: {best_result['mutation']}\n")
        f.write(f"  - 최적 x: {best_result['best_x']:.6f}\n")
        if setting['encoding_type'] == 'binary':
            f.write(f"  - 이진 표현: {best_result['binary']}")

    print(f"\n결과가 {filename} 파일에 저장되었습니다.")