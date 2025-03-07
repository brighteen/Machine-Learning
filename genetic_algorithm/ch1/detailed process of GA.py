import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt

def _utils_constraints(g, min, max):
    if max and g > max:
        g = max
    if min and g < min:
        g = min
    return g

def crossover_blend(g1, g2, alpha, min = None, max = None):
    shift = (1. + 2. * alpha) * random.random() - alpha
    new_g1 = (1. - shift) * g1 + shift * g2
    new_g2 = shift * g1 + (1. - shift) * g2
    return _utils_constraints(new_g1, min, max), _utils_constraints(new_g2, min, max)

def mutate_gaussian(g, mu, sigma, min = None, max = None):
    mutated_gene = g + random.gauss(mu, sigma)
    return _utils_constraints(mutated_gene, min, max)

def select_tournament(population, tournament_size):
    new_offspring = []
    for _ in range(len(population)):
        candidates = [random.choice(population) for _ in
        range(tournament_size)]
        new_offspring.append(max(candidates, key = lambda ind:
        ind.fitness))
    return new_offspring

def func(x):
    return np.sin(x) - .2 * abs(x)

def get_best(population):
    best = population[0]
    for ind in population:
        if ind.fitness > best.fitness:
            best = ind
    return best

class Individual: # 개체를 나타내는 클래스
    def __init__(self, gene_list: List[float]) -> None:
        self.gene_list = gene_list
        self.fitness = func(self.gene_list[0])
    def get_gene(self):
        return self.gene_list[0]
    @classmethod
    def crossover(cls, parent1, parent2): # 교차차
        child1_gene, child2_gene = crossover_blend(parent1.get_gene(),
        parent2.get_gene(), 1, -10, 10)
        return Individual([child1_gene]), Individual([child2_gene])
    @classmethod
    def mutate(cls, ind): # 돌연변이
        mutated_gene = mutate_gaussian(ind.get_gene(), 0, 1, -10, 10)
        return Individual([mutated_gene])
    @classmethod
    def select(cls, population): # 자연 선택
        return select_tournament(population, tournament_size = 3)
    @classmethod
    def create_random(cls):
        return Individual([random.randrange(-1000, 1000) / 100])

def plot_population_extended(
    population, 
    title="", 
    parent_child_links=None, 
    mutated_indices=None, 
    show_duplicates=False,
    label_fitness=False
):
    """
    - population: 현재 개체 리스트
    - title: 그래프 제목
    - parent_child_links: 교차 과정에서 (부모→자식) 정보를 담은 리스트
                         [((px, pf), (cx, cf)), ...]
    - mutated_indices: 돌연변이가 일어난 개체 인덱스 목록
    - show_duplicates: 동일 유전자값의 개체 수 표시 여부
    - label_fitness: True이면 각 개체 옆에 (유전자, 적합도) 라벨을 출력
    """
    x_vals = np.linspace(-10, 10, 400)
    plt.plot(x_vals, [func(x) for x in x_vals], '--', color='blue')
    
    coords = [(ind.get_gene(), ind.fitness) for ind in population]
    gene_counter = None
    if show_duplicates:
        from collections import Counter
        gene_counter = Counter([round(c[0], 2) for c in coords])
    
    for i, (xv, fv) in enumerate(coords):
        plt.plot(xv, fv, 'o', color='orange')
        
        # fitness 라벨 표시 옵션
        if label_fitness:
            plt.text(xv, fv, f"({xv:.2f}, {fv:.2f})", fontsize=8, color='black')
        
        # 중복 개체 표시
        if show_duplicates and gene_counter:
            rounded_gene = round(xv, 2)
            count_val = gene_counter[rounded_gene]
            if count_val > 1:
                plt.text(xv, fv-0.15, f"count={count_val}", color='gray', fontsize=7)
        
        # 돌연변이가 발생한 개체라면 'mutate' 라벨 표시
        if mutated_indices and i in mutated_indices:
            plt.text(xv+0.05, fv+0.05, "mutate", fontsize=8, color='red')
    
    # 부모→자식 연결선 표시 (교차 시각화)
    if parent_child_links:
        for (px, pf), (cx, cf) in parent_child_links:
            plt.plot([px, cx], [pf, cf], color='gray', linestyle='--', linewidth=1)
    
    # 최고 적합도 개체 강조 (녹색 사각형)
    best = get_best(population)
    plt.plot(best.get_gene(), best.fitness, 's', color='green')
    
    plt.title(title)
    plt.show()
    plt.close()

def run_one_generation(population, generation_number):
    # 1) 초기 염색체 생성 (초기 상태 시각화)
    plot_population_extended(
        population, 
        title=f"Step 1: Initial (Gen {generation_number})", 
        parent_child_links=None, 
        mutated_indices=None, 
        show_duplicates=False,
        label_fitness=False  # 초기 단계에서는 라벨 없이 출력
    )
    
    # 2) 적합도 계산 (Fitness Calculation)
    #    이 단계에서는 각 개체 옆에 (유전자, 적합도) 라벨을 표시
    plot_population_extended(
        population, 
        title=f"Step 2: Fitness Calculation (Gen {generation_number})", 
        parent_child_links=None, 
        mutated_indices=None, 
        show_duplicates=False,
        label_fitness=True  # 라벨 표시 활성화
    )
    
    # 3) 선택 (Selection)
    selected_population = Individual.select(population)
    plot_population_extended(
        selected_population, 
        title=f"Step 3: Selection (Gen {generation_number})", 
        parent_child_links=None, 
        mutated_indices=None, 
        show_duplicates=True,
        label_fitness=False
    )
    
    # 4) 교차 (Crossover)
    crossed_population = []
    parent_child_links = []
    for ind1, ind2 in zip(selected_population[::2], selected_population[1::2]):
        if random.random() < CROSSOVER_PROBABILITY:
            kid1, kid2 = Individual.crossover(ind1, ind2)
            crossed_population.extend([kid1, kid2])
            parent_child_links.append(((ind1.get_gene(), ind1.fitness), (kid1.get_gene(), kid1.fitness)))
            parent_child_links.append(((ind2.get_gene(), ind2.fitness), (kid2.get_gene(), kid2.fitness)))
        else:
            crossed_population.extend([ind1, ind2])
    plot_population_extended(
        crossed_population, 
        title=f"Step 4: Crossover (Gen {generation_number})", 
        parent_child_links=parent_child_links, 
        mutated_indices=None, 
        show_duplicates=False,
        label_fitness=False
    )
    
    # 5) 돌연변이 (Mutation)
    mutated_population = []
    mutated_indices = []
    for i, ind in enumerate(crossed_population):
        if random.random() < MUTATION_PROBABILITY:
            mutated_population.append(Individual.mutate(ind))
            mutated_indices.append(i)
        else:
            mutated_population.append(ind)
    plot_population_extended(
        mutated_population, 
        title=f"Step 5: Mutation (Gen {generation_number})", 
        parent_child_links=None, 
        mutated_indices=mutated_indices, 
        show_duplicates=False,
        label_fitness=False
    )
    
    return mutated_population

POPULATION_SIZE = 10
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.1
MAX_GENERATIONS = 10

# 초기 인구 생성
population = [Individual.create_random() for _ in range(POPULATION_SIZE)]
# 0세대(또는 generation_number=1로 간주) 업데이트
population = run_one_generation(population, 1)