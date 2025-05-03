import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
np.random.seed(0)
random.seed(0)
# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# --- 문제 데이터 ---
days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
courses = [
    (1, "영상이해", "송주환", "1.5"),
    (1, "인공지능기초와활용", "이근호", "1.5"),
    (1, "인공지능수학기초", "민정익", "1.5"),
    (1, "소프트웨어적사고", "김영수", "1.5"),
    (3, "리빙랩1", "이근호", "3h"),
    (2, "문제해결과알고리즘", "송주환", "1.5"),
    (2, "리눅스운영체제", "김영수", "1.5"),
    (2, "기계학습", "권수태", "1.5"),
    (2, "딥러닝", "고선우", "1.5"),
    (3, "웹프로그래밍", "송주환", "1.5"),
    (3, "자연어처리", "김영수", "1.5"),
    (3, "강화학습", "민정익", "1.5"),
    (3, "첨단신경망", "고선우", "1.5"),
    (4, "인공지능세미나", "고선우", "3h"),
    (4, "리빙랩2", "권수태", "3h"),
    (4, "프로젝트1", "민정익", "1.5"),
    (4, "프로젝트2", "김영수", "1.5"),
]

# 고정 불가 슬롯
forbidden = {
    1: {(0,p) for p in [12,13,14]} | {(2,p) for p in [9,10,11]} | {(1,p) for p in [4,5,6]},
    2: {(0,p) for p in [9,10,11]} | {(2,p) for p in [12,13,14]} | {(1,p) for p in [1,2,3]} | {(3,p) for p in [4,5,6]} | {(1,p) for p in [9,10,11,12,13,14]} | {(3,p) for p in [9,10,11]},
    3: {(0,p) for p in [9,10,11]} | {(2,p) for p in [12,13,14]} | {(1,p) for p in [1,2,3]} | {(3,p) for p in [4,5,6]},
    4: set()
}

# 슬롯 정의
slot_pairs_1_5 = [
    ((0,1),(2,4)), ((0,4),(2,1)), ((0,9),(2,12)), ((0,12),(2,9)),
    ((1,1),(3,4)), ((1,4),(3,1)), ((1,9),(3,12)), ((1,12),(3,9))
]

# 가능한 슬롯 생성 (3h: 월~목 우선, 그 다음 금 오전·오후)
possible_slots = []
for year, _, _, typ in courses:
    if typ == "1.5":
        choices = slot_pairs_1_5
    else:
        choices = []
        # 월~목
        for d in range(4):
            for start in range(1, 15-6+1):
                if any((d, start+off) in forbidden[year] for off in range(6)):
                    continue
                choices.append(((d, start),))
        # 금요일 오전, 오후
        for start in (1, 9):
            if all((4, start+off) not in forbidden[year] for off in range(6)):
                choices.append(((4, start),))
    # forbidden 필터
    filtered = []
    length = 3 if typ=="1.5" else 6
    for ch in choices:
        if all((d, p+off) not in forbidden[year] for d, p in ch for off in range(length)):
            filtered.append(ch)
    possible_slots.append(filtered)

# 유전자 디코드
def decode_gene(gene):
    assignments = []
    for idx, g in enumerate(gene):
        year, name, prof, typ = courses[idx]
        ch = possible_slots[idx][g]
        if typ == "1.5":
            for d, p in ch:
                for off in range(3):
                    assignments.append((year, name, prof, d, p+off))
        else:
            d, p = ch[0]
            for off in range(6):
                assignments.append((year, name, prof, d, p+off))
    return assignments

# 적합도 및 위반 내역 계산 (non3h_day 제거)
def fitness_and_violations(gene):
    assignments = decode_gene(gene)
    violations = {
        'year_conflict': 0,
        'prof_conflict': 0,
        'lunch_violation': 0,
        'forbidden_slot': 0,
        'soft_prof_days': 0
    }
    occ_y, occ_p, prof_days = {}, {}, {}
    # 하드 위반 체크
    for year, name, prof, d, p in assignments:
        if occ_y.get((year,d,p)): violations['year_conflict'] += 1
        occ_y.setdefault((year,d,p), []).append(name)
        if (prof,d,p) in occ_p: violations['prof_conflict'] += 1
        occ_p[(prof,d,p)] = name
        if p in (7,8): violations['lunch_violation'] += 1
        if (d,p) in forbidden[year]: violations['forbidden_slot'] += 1
    # 소프트 위반: 교수 4일 미달
    for _,_,prof,d,_ in assignments:
        prof_days.setdefault(prof, set()).add(d)
    for ds in prof_days.values():
        if len(ds) < 4:
            violations['soft_prof_days'] += (4 - len(ds))
    hard = sum(violations[k] for k in ['year_conflict','prof_conflict','lunch_violation','forbidden_slot'])
    soft = violations['soft_prof_days']
    fitness = -(100 * hard + soft)
    return fitness, violations

# GA 파라미터 설정
NUM = len(courses)
POP, GENS, TOURN, CROSS, MUT, ELITE = 100, 200, 3, 0.8, 0.2, 2

# 초기 개체군
def random_gene():
    return [random.randrange(len(possible_slots[i])) for i in range(NUM)]

population = [(*fitness_and_violations(random_gene()), random_gene()) for _ in range(POP)]
best = max(population, key=lambda x: x[0])

# GA 주 루프
# GA 메인 루프
for _ in range(GENS):
    new = sorted(population, key=lambda x: x[0], reverse=True)[:ELITE]
    while len(new) < POP:
        p1 = max(random.sample(population, TOURN), key=lambda x: x[0])[2]
        p2 = max(random.sample(population, TOURN), key=lambda x: x[0])[2]
        if random.random() < CROSS:
            pt = random.randint(1, NUM-1)
            c1, c2 = p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
        else:
            c1, c2 = p1[:], p2[:]
        for child in (c1, c2):
            for i in range(NUM):
                if random.random() < MUT:
                    child[i] = random.randrange(len(possible_slots[i]))
            new.append((*fitness_and_violations(child), child))
            if len(new) >= POP:
                break
    population = new
    curr = max(population, key=lambda x: x[0])
    if curr[0] > best[0]:
        best = curr


# 디버그 출력
best_f, best_violations, best_gene = best[0], best[1], best[2]
print("Best Fitness:", best_f)
print("Violation breakdown:")
for k, v in best_violations.items():
    print(f"  {k}: {v}")

# 시간표 구축 및 시각화 (셀 병합 포함)
schedule = {y: [["" for _ in days] for _ in range(14)] for y in range(1,5)}
schedule_prof = {y: [["" for _ in days] for _ in range(14)] for y in range(1,5)}
for year,name,prof,d,p in decode_gene(best_gene):
    schedule[year][p-1][d] = name
    schedule_prof[year][p-1][d] = prof

profs = list({prof for *_ ,prof,_ in courses})
cmap = plt.cm.get_cmap('tab10', len(profs))
colors = {prof: cmap(i) for i, prof in enumerate(profs)}

fig, axes = plt.subplots(2,2, figsize=(12,10))
axes = axes.flatten()
for idx, ax in enumerate(axes, start=1):
    ax.set_title(f"{idx}학년")
    ax.set_xticks(range(len(days))); ax.set_xticklabels(days)
    ax.xaxis.tick_top()
    ax.set_yticks(range(14)); ax.set_yticklabels(range(1,15))
    ax.set_ylim(13.5, -0.5); ax.set_xlim(-0.5, 4.5)
    ax.set_xticks(np.arange(-0.5,5,1), minor=True)
    ax.set_yticks(np.arange(-0.5,15,1), minor=True)
    ax.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)
    for day in range(5):
        per = 0
        while per < 14:
            name = schedule[idx][per][day]
            prof = schedule_prof[idx][per][day]
            if name:
                run = 1
                while per+run < 14 and schedule[idx][per+run][day] == name:
                    run += 1
                color = colors[prof]
                rect = Rectangle((day-0.5, per-0.5), 1, run, facecolor=color, edgecolor='black')
                ax.add_patch(rect)
                ax.text(day, per + (run-1)/2, f"{name}\n{prof}", ha="center", va="center", fontsize=8)
                per += run
            else:
                per += 1
plt.tight_layout()
plt.show()
