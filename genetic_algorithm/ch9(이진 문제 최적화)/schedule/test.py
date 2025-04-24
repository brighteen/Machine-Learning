import random
import copy
import pandas as pd
from toolbox import (
    selection_rank_with_elite,
    crossover_n_point,
    crossover_fitness_driven_one_point,
    mutation_bit_flip,
    mutation_shuffle,
    mutation_fitness_driven_bit_flip,
    crossover_operation,
    mutation_operation,
)
from schedule_analyzer import shift_deviations, shift_relax
from individual import Individual

# ── 문제 및 GA 파라미터 설정 ─────────────────────────────────────────
POP_SIZE       = 10
CX_PROB        = 0.8
MUT_PROB       = 0.5
MAX_GENERATIONS= 40

random.seed(1)
Individual.set_employees(3)
Individual.set_period(3)

def fitness_fn(df):
    dev   = shift_deviations(df, mor_min=1, mor_max=4,
                                 day_min=2, day_max=5,
                                 evn_min=1, evn_max=2)
    relax = shift_relax(df, 1, 1, 3)
    return -(dev + relax * 5)

Individual.set_fitness_function(fitness_fn)

# ── 연산자 조합 정의 ─────────────────────────────────────────────────
crossover_methods = {
    "n_point": lambda p1,p2: [Individual(g) 
                               for g in crossover_n_point(p1.gene_list, p2.gene_list, 3)],
    "one_point": crossover_fitness_driven_one_point,
}

mutation_methods = {
    "bit_flip":        lambda ind: Individual(mutation_bit_flip(ind.gene_list)),
    "shuffle":         lambda ind: Individual(mutation_shuffle(ind.gene_list)),
    "fitness_driven":  mutation_fitness_driven_bit_flip,
}

# ── 실험 루프 ─────────────────────────────────────────────────────────
results = []
for cx_name, cx_func in crossover_methods.items():
    for mut_name, mut_func in mutation_methods.items():
        # 초기화
        random.seed(1)
        population = [Individual.generate_random() for _ in range(POP_SIZE)]
        best_ind   = max(population, key=lambda ind: ind.fitness)

        # 세대별 통계 저장 리스트
        best_list, avg_list, worst_list, std_list = [], [], [], []

        for gen in range(MAX_GENERATIONS):
            fitness_vals = [ind.fitness for ind in population]
            avg = sum(fitness_vals) / len(fitness_vals)
            var = sum((x - avg)**2 for x in fitness_vals) / len(fitness_vals)

            best_list.append(max(fitness_vals))
            avg_list.append(avg)
            worst_list.append(min(fitness_vals))
            std_list.append(var**0.5)

            # 선택→교차→돌연변이
            selected  = selection_rank_with_elite(population, elite_size=2)
            offspring = crossover_operation(selected, cx_func, CX_PROB)
            population= mutation_operation(offspring, mut_func, MUT_PROB)

            # 베스트 개체 갱신
            gen_best = max(population, key=lambda ind: ind.fitness)
            if gen_best.fitness > best_ind.fitness:
                best_ind = gen_best

        # 개선 속도 지표 계산
        first_imp = next((i for i,v in enumerate(best_list) if v>best_list[0]), None)
        last_imp  = max(i for i,v in enumerate(best_list) if v==best_list[-1])
        delta_imp = best_list[-1] - best_list[0]

        # 패널티 분해 (최종 베스트 스케줄)
        sched    = best_ind.create_schedule()
        dev_pen  = shift_deviations(sched, mor_min=1, mor_max=4, day_min=2, day_max=5, evn_min=1, evn_max=2)
        relax_pen= shift_relax(sched, 1, 1, 3)

        # 요약 결과 저장
        results.append({
            "crossover"          : cx_name,
            "mutation"           : mut_name,
            "final_best"         : best_list[-1],
            "final_avg"          : avg_list[-1],
            "final_worst"        : worst_list[-1],
            "final_std"          : std_list[-1],
            "first_imp_gen"      : first_imp,
            "last_imp_gen"       : last_imp,
            "improvement_delta"  : delta_imp,
            "deviation_penalty"  : dev_pen,
            "relax_penalty"      : relax_pen,
        })

# ── 결과 출력 ────────────────────────────────────────────────────────
df = pd.DataFrame(results)
print(df.to_string(index=False))