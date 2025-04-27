import random
import numpy as np
from itertools import product
from toolbox import (
    # 선택 연산자
    selection_proportional, selection_rank, selection_rank_with_elite,
    selection_stochastic_universal_sampling, selection_tournament,
    # 교차 연산자
    crossover_blend, crossover_linear, crossover_uniform,
    # 돌연변이 연산자
    mutation_random_deviation, mutation_fitness_driven_random_deviation
)
import os
from datetime import datetime

def func(x):
    """최소화할 목적 함수: f(x) = 2sin(x) - 0.5x"""
    return 2 * np.sin(x) - 0.5 * x

class Individual:
    def __init__(self, gene_list) -> None:
        if isinstance(gene_list, Individual):
            self.gene_list = gene_list.gene_list
        else:
            self.gene_list = gene_list if isinstance(gene_list, list) else [gene_list]
        # 제약조건 적용: -15 ≤ x ≤ 15
        self.gene_list[0] = max(min(self.gene_list[0], 15), -15)
        # 최소값을 찾는 문제이므로 적합도는 함수값의 음수
        self.fitness = -func(self.gene_list[0])

    def __str__(self):
        return f'x: {self.gene_list[0]}, f(x): {func(self.gene_list[0])}'

# 초기화
POPULATION_SIZE = 20
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.2
MAX_GENERATIONS = 100
CONVERGENCE_THRESHOLD = 1e-9  # 수렴 판단 임계값
CONVERGENCE_GENERATIONS = 5   # 연속적으로 변화가 적은 세대 수

# 개체 생성 함수
def create_random_individual():
    x = random.uniform(-15, 15)
    return Individual([x])

# 실수 최적화에 적합한 연산자만 선택
selection_methods = [
    selection_proportional,
    selection_rank,
    selection_rank_with_elite,
    selection_stochastic_universal_sampling,
    selection_tournament
]

crossover_methods = [
    crossover_blend,
    crossover_linear,
    crossover_uniform
]

mutation_methods = [
    mutation_random_deviation,
    mutation_fitness_driven_random_deviation
]

# 연산자별 파라미터 설정
def get_crossover_params(crossover):
    if crossover == crossover_blend:
        return {'alpha': 0.5}
    elif crossover == crossover_linear:
        return {'alpha': 0.7}
    elif crossover == crossover_uniform:
        return {'prop': 0.5}
    return {}

def get_mutation_params(mutation):
    if mutation == mutation_random_deviation:
        return {'mu': 0, 'sigma': 1.0, 'p': 0.1}
    elif mutation == mutation_fitness_driven_random_deviation:
        return {'mu': 0, 'sigma': 1.0, 'p': 0.1, 'max_tries': 3}
    return {}

# 모든 조합 생성
combinations = list(product(selection_methods, crossover_methods, mutation_methods))
results = []

print("실험 시작...\n")

for selection, crossover, mutation in combinations:
    # 초기 개체군 생성
    population = [create_random_individual() for _ in range(POPULATION_SIZE)]
    best_ever = min(population, key=lambda x: func(x.gene_list[0]))
    
    # 조기 종료를 위한 변수들
    prev_best_fx = float('inf')
    convergence_count = 0
    generation_completed = 0
    
    # 연산자별 파라미터 가져오기
    crossover_params = get_crossover_params(crossover)
    mutation_params = get_mutation_params(mutation)

    for generation in range(MAX_GENERATIONS):
        # 선택
        selected = selection(population)

        # 교차
        offspring = []
        for p1, p2 in zip(selected[::2], selected[1::2]):
            if random.random() < CROSSOVER_PROBABILITY:
                c1, c2 = crossover(p1.gene_list, p2.gene_list, **crossover_params)
                offspring.extend([Individual(c1), Individual(c2)])
            else:
                offspring.extend([p1, p2])

        # 돌연변이
        mutated_offspring = []
        for ind in offspring:
            if random.random() < MUTATION_PROBABILITY:
                mutated = mutation(ind.gene_list, **mutation_params)
                mutated_offspring.append(Individual(mutated))
            else:
                mutated_offspring.append(ind)

        # 새로운 세대 설정
        population = mutated_offspring
        
        # 현재 세대의 최소값 갱신
        current_best = min(population, key=lambda x: func(x.gene_list[0]))
        current_fx = func(current_best.gene_list[0])
        
        if current_fx < func(best_ever.gene_list[0]):
            best_ever = current_best

        # 수렴 검사
        if abs(current_fx - prev_best_fx) < CONVERGENCE_THRESHOLD:
            convergence_count += 1
        else:
            convergence_count = 0
        
        prev_best_fx = current_fx
        generation_completed = generation + 1

        # 조기 종료 조건
        if convergence_count >= CONVERGENCE_GENERATIONS:
            break

    # 결과 저장
    results.append({
        'generations': generation_completed,
        'selection': selection.__name__,
        'crossover': crossover.__name__,
        'mutation': mutation.__name__,
        'best_fx': func(best_ever.gene_list[0]),
        'best_x': best_ever.gene_list[0]
    })
    
    # 현재 조합 결과 출력
    print(f"실행 완료:")
    print(f"- 연산자: {selection.__name__} / {crossover.__name__} / {mutation.__name__}")
    print(f"- 세대 수: {generation_completed}")
    print(f"- 최소값 f(x): {func(best_ever.gene_list[0]):.6f}")
    print(f"- 최적 x: {best_ever.gene_list[0]:.6f}")
    print("-" * 50)

# 결과 파일 저장 경로 설정
result_dir = r"C:\Users\brigh\Documents\GitHub\Machine-Learning\genetic_algorithm\중간시험\결과"

# 다음 결과 파일 번호 찾기
def get_next_result_number():
    files = [f for f in os.listdir(result_dir) if f.startswith('ga_results_') and f.endswith('.md')]
    if not files:
        return 1
    numbers = [int(f.split('_')[2].split('.')[0]) for f in files]
    return max(numbers) + 1

# 결과 파일 이름 생성
result_number = get_next_result_number()
result_filename = os.path.join(result_dir, f'ga_results_{result_number}.md')

# 최종 결과를 markdown 파일로 저장
with open(result_filename, 'w', encoding='utf-8') as f:
    f.write(f"# 유전 알고리즘 실험 결과 #{result_number}\n\n")
    f.write(f"실험 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("## 실험 설정\n")
    f.write(f"- 인구 크기: {POPULATION_SIZE}\n")
    f.write(f"- 교차 확률: {CROSSOVER_PROBABILITY}\n")
    f.write(f"- 돌연변이 확률: {MUTATION_PROBABILITY}\n")
    f.write(f"- 최대 세대 수: {MAX_GENERATIONS}\n")
    f.write(f"- 수렴 임계값: {CONVERGENCE_THRESHOLD}\n")
    f.write(f"- 수렴 판단 세대 수: {CONVERGENCE_GENERATIONS}\n\n")
    
    f.write("## 실험 결과\n")
    f.write("| 선택 연산자 | 교차 연산자 | 돌연변이 연산자 | 세대 수 | 최소값 f(x) | 최적 x |\n")
    f.write("|------------|------------|----------------|---------|------------|--------|\n")
    
    # 결과를 최소값 기준으로 정렬
    sorted_results = sorted(results, key=lambda x: x['best_fx'])
    
    for result in sorted_results:
        f.write(f"| {result['selection']} | {result['crossover']} | {result['mutation']} | ")
        f.write(f"{result['generations']} | {result['best_fx']:.6f} | {result['best_x']:.6f} |\n")
    
    # 전체 실험의 통계
    avg_generations = sum(r['generations'] for r in results) / len(results)
    best_fx = min(r['best_fx'] for r in results)
    best_combination = next(r for r in results if r['best_fx'] == best_fx)
    
    f.write("\n## 통계 분석\n")
    f.write(f"- 평균 수렴 세대 수: {avg_generations:.2f}\n")
    f.write(f"- 전체 최소값: {best_fx:.6f}\n")
    f.write("- 최적의 연산자 조합:\n")
    f.write(f"  - 선택 연산자: {best_combination['selection']}\n")
    f.write(f"  - 교차 연산자: {best_combination['crossover']}\n")
    f.write(f"  - 돌연변이 연산자: {best_combination['mutation']}\n")
    f.write(f"  - 최적 x: {best_combination['best_x']:.6f}")

print(f"\n결과가 {result_filename} 파일에 저장되었습니다.")