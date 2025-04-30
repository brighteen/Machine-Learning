import random
import numpy as np
from itertools import product
from toolbox import (
    selection_proportional, selection_rank, selection_rank_with_elite,
    selection_stochastic_universal_sampling, selection_tournament,
    crossover_one_point, crossover_n_point, crossover_uniform,
    mutation_bit_flip, mutation_fitness_driven_bit_flip
)
import os
from datetime import datetime

class BinaryIndividual:
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
        self.fitness = -self._evaluate()  # 최소화 문제이므로 음수화

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

    def _evaluate(self):
        """목적 함수: f(x) = 2sin(x) - 0.5x"""
        return 2 * np.sin(self.x) - 0.5 * self.x

    def __str__(self):
        binary = ''.join(map(str, self.gene_list))
        return f'Binary: {binary}, x: {self.x:.4f}, f(x): {self._evaluate():.4f}'

# 초기화 설정
POPULATION_SIZE = 20
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.2
MAX_GENERATIONS = 100
CONVERGENCE_THRESHOLD = 1e-9
CONVERGENCE_GENERATIONS = 5
BITS = 16  # 이진 표현 비트 수

# 이진 개체 생성 함수
def create_random_binary_individual():
    return BinaryIndividual(bits=BITS)

# 연산자 목록
selection_methods = [
    selection_proportional,
    selection_rank,
    selection_rank_with_elite,
    selection_stochastic_universal_sampling,
    selection_tournament
]

crossover_methods = [
    crossover_one_point,
    crossover_n_point,
    crossover_uniform
]

mutation_methods = [
    mutation_bit_flip,
    mutation_fitness_driven_bit_flip
]

# 연산자별 파라미터 설정
def get_crossover_params(crossover):
    if crossover == crossover_n_point:
        return {'n': 2}
    elif crossover == crossover_uniform:
        return {'prop': 0.5}
    return {}

def get_mutation_params(mutation):
    if mutation == mutation_fitness_driven_bit_flip:
        return {'max_tries': 3}
    return {}

# 실험 시작
print("이진 인코딩을 사용한 실험 시작...\n")

combinations = list(product(selection_methods, crossover_methods, mutation_methods))
results = []

for selection, crossover, mutation in combinations:
    # 초기 개체군 생성
    population = [create_random_binary_individual() for _ in range(POPULATION_SIZE)]
    best_ever = min(population, key=lambda x: x._evaluate())
    
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
                offspring.extend([BinaryIndividual(c1, BITS), BinaryIndividual(c2, BITS)])
            else:
                offspring.extend([p1, p2])

        # 돌연변이
        mutated_offspring = []
        for ind in offspring:
            if random.random() < MUTATION_PROBABILITY:
                mutated = mutation(ind.gene_list, **mutation_params)
                mutated_offspring.append(BinaryIndividual(mutated, BITS))
            else:
                mutated_offspring.append(ind)

        # 새로운 세대 설정
        population = mutated_offspring
        
        # 현재 세대의 최소값 갱신
        current_best = min(population, key=lambda x: x._evaluate())
        current_fx = current_best._evaluate()
        
        if current_fx < best_ever._evaluate():
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
        'best_fx': best_ever._evaluate(),
        'best_x': best_ever.x,
        'binary': ''.join(map(str, best_ever.gene_list))
    })
    
    # 현재 조합 결과 출력
    print(f"실행 완료:")
    print(f"- 연산자: {selection.__name__} / {crossover.__name__} / {mutation.__name__}")
    print(f"- 세대 수: {generation_completed}")
    print(f"- 최소값 f(x): {best_ever._evaluate():.6f}")
    print(f"- 최적 x: {best_ever.x:.6f}")
    print(f"- 이진 표현: {best_ever.gene_list}")
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
    f.write(f"# 유전 알고리즘 실험 결과 #{result_number} (이진 인코딩)\n\n")
    f.write(f"실험 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("## 실험 설정\n")
    f.write(f"- 인구 크기: {POPULATION_SIZE}\n")
    f.write(f"- 교차 확률: {CROSSOVER_PROBABILITY}\n")
    f.write(f"- 돌연변이 확률: {MUTATION_PROBABILITY}\n")
    f.write(f"- 최대 세대 수: {MAX_GENERATIONS}\n")
    f.write(f"- 수렴 임계값: {CONVERGENCE_THRESHOLD}\n")
    f.write(f"- 수렴 판단 세대 수: {CONVERGENCE_GENERATIONS}\n")
    f.write(f"- 이진 표현 비트 수: {BITS}\n\n")
    
    f.write("## 실험 결과\n")
    f.write("| 선택 연산자 | 교차 연산자 | 돌연변이 연산자 | 세대 수 | 최소값 f(x) | 최적 x | 이진 표현 |\n")
    f.write("|------------|------------|----------------|---------|------------|--------|----------|\n")
    
    # 결과를 최소값 기준으로 정렬
    sorted_results = sorted(results, key=lambda x: x['best_fx'])
    
    for result in sorted_results:
        f.write(f"| {result['selection']} | {result['crossover']} | {result['mutation']} | ")
        f.write(f"{result['generations']} | {result['best_fx']:.6f} | {result['best_x']:.6f} | {result['binary']} |\n")
    
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
    f.write(f"  - 최적 x: {best_combination['best_x']:.6f}\n")
    f.write(f"  - 이진 표현: {best_combination['binary']}")

print(f"\n결과가 {result_filename} 파일에 저장되었습니다.")