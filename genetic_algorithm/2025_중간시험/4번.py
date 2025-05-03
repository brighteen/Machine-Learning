import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 시드 고정
np.random.seed(0)
random.seed(0)

# --- 문제 데이터 ---
days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
rooms = {1: "423호", 2: "424호", 3: "514호", 4: "425호"}
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

# 학년별 고정 수업 (진탐 포함)
fixed_slots = {
    1: [(0, 11, 3, "외국어"), (2, 8, 3, "외국어"), (1, 3, 3, "채플"), (2, 14, 4, "진탐")],
    2: [(0, 8, 3, "핵심교양A"), (2, 11, 3, "핵심교양A"),
        (1, 0, 3, "핵심교양B"), (3, 3, 3, "핵심교양B"),
        (1, 8, 3, "채플"), (1, 11, 3, "기독교"), (3, 8, 3, "기독교"),
        (2, 14, 4, "진탐")],
    3: [(0, 8, 3, "핵심교양A"), (2, 11, 3, "핵심교양A"),
        (1, 0, 3, "핵심교양B"), (3, 3, 3, "핵심교양B"),
        (2, 14, 4, "진탐")],
    4: [(2, 14, 4, "진탐")]
}

# forbidden 시간대
forbidden = {
    1: {(0, p) for p in [12,13,14]} | {(2, p) for p in [9,10,11,15,16,17,18]} | {(1, p) for p in [4,5,6]},
    2: {(0, p) for p in [9,10,11]} | {(2, p) for p in [12,13,14,15,16,17,18]} |
       {(1, p) for p in [1,2,3]} | {(3, p) for p in [4,5,6]} |
       {(1, p) for p in [9,10,11,12,13,14]} | {(3, p) for p in [9,10,11]},
    3: {(0, p) for p in [9,10,11]} | {(2, p) for p in [12,13,14,15,16,17,18]} |
       {(1, p) for p in [1,2,3]} | {(3, p) for p in [4,5,6]},
    4: {(2, p) for p in [15,16,17,18]}
}

# 1.5h 짝꿍 슬롯 정의
slot_pairs_1_5 = [
    ((0,1),(2,4)), ((0,4),(2,1)), ((0,9),(2,12)), ((0,12),(2,9)),
    ((1,1),(3,4)), ((1,4),(3,1)), ((1,9),(3,12)), ((1,12),(3,9))
]

# 가능한 슬롯 생성 (3h: 월~목 우선, 그 다음 금 오전/오후)
possible_slots = []
for year, _, _, typ in courses:
    if typ == "1.5":
        choices = slot_pairs_1_5
    else:
        choices = []
        for d in range(4):
            for start in range(1, 10):
                if any((d, start+off) in forbidden[year] for off in range(6)):
                    continue
                choices.append(((d, start),))
        for start in (1, 9):
            if all((4, start+off) not in forbidden[year] for off in range(6)):
                choices.append(((4, start),))
    length = 3 if typ=="1.5" else 6
    filtered = [
        ch for ch in choices
        if all((d, p+off) not in forbidden[year] for d,p in ch for off in range(length))
    ]
    possible_slots.append(filtered)

# 크로모솜 디코드
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

# 적합도 및 위반 내역 계산
def fitness_and_violations(gene):
    assign = decode_gene(gene)
    vio = {'year_conflict':0,'prof_conflict':0,'lunch':0,'forbidden':0,'soft_days':0}
    occ_y, occ_p, prof_days = {}, {}, {}
    for year,name,prof,d,p in assign:
        if occ_y.get((year,d,p)): vio['year_conflict']+=1
        occ_y.setdefault((year,d,p),[]).append(name)
        if (prof,d,p) in occ_p: vio['prof_conflict']+=1
        occ_p[(prof,d,p)] = name
        if p in (7,8): vio['lunch']+=1
        if (d,p) in forbidden[year]: vio['forbidden']+=1
        prof_days.setdefault(prof,set()).add(d)
    for ds in prof_days.values():
        if len(ds)<4: vio['soft_days'] += (4-len(ds))
    hard = vio['year_conflict']+vio['prof_conflict']+vio['lunch']+vio['forbidden']
    soft = vio['soft_days']
    return -(100*hard + soft), vio

# GA 파라미터 및 실행
NUM = len(courses)
POP, GENS, TOURN, CROSS, MUT, ELITE = 100, 200, 3, 0.8, 0.2, 2
population = [(*fitness_and_violations([random.randrange(len(ps)) for ps in possible_slots]), [random.randrange(len(ps)) for ps in possible_slots]) for _ in range(POP)]
best = max(population, key=lambda x: x[0])
for _ in range(GENS):
    new_pop = sorted(population, key=lambda x: x[0], reverse=True)[:ELITE]
    while len(new_pop) < POP:
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
            new_pop.append((*fitness_and_violations(child), child))
            if len(new_pop) >= POP:
                break
    population = new_pop
    curr = max(population, key=lambda x: x[0])
    if curr[0] > best[0]:
        best = curr

bf, bv, bg = best
print("=== 최종 결과 ===")
print("Best Fitness:", bf)
print("Violation breakdown:")
for k, v in bv.items():
    print(f"  {k}: {v}")

# 시간표 구축 (18교시)
schedule = {y: [["" for _ in days] for _ in range(18)] for y in range(1,5)}
schedule_prof = {y: [["" for _ in days] for _ in range(18)] for y in range(1,5)}
for year,name,prof,d,p in decode_gene(bg):
    schedule[year][p-1][d] = name
    schedule_prof[year][p-1][d] = prof

# 교수색 & y축 라벨 정의
profs = list({prof for *_ ,prof,_ in courses})
cmap = plt.cm.get_cmap('tab10', len(profs))
colors = {prof: cmap(i) for i, prof in enumerate(profs)}
yticks = list(range(18))
ylabels = [f"{i+1} ({9 + (i//2):02d}:{(i%2)*30:02d})" for i in yticks]

# 시각화 (2x2)
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes = axes.flatten()
for idx, ax in enumerate(axes, start=1):
    ax.set_title(f"{idx}학년 시간표({rooms[idx]})", pad=20)
    ax.xaxis.tick_top()
    ax.set_xticks(range(5)); ax.set_xticklabels(days)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels)
    ax.set_ylim(17.5, -0.5); ax.set_xlim(-0.5, 4.5)
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 19, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)

    # 고정 수업 (회색)
    for day, st, span, nm in fixed_slots[idx]:
        rect = Rectangle((day-0.5, st-0.5), 1, span, facecolor='lightgrey', edgecolor='black')
        ax.add_patch(rect)
        ax.text(day, st + (span-1)/2, nm, ha="center", va="center", fontsize=8)

    # 가변 수업 (병합, 색상, 강의실 포함)
    for day in range(5):
        per = 0
        while per < 18:
            nm = schedule[idx][per][day]
            pf = schedule_prof[idx][per][day]
            if nm:
                run = 1
                while per+run < 18 and schedule[idx][per+run][day] == nm:
                    run += 1
                room = rooms[idx]
                color = colors[pf]
                rect = Rectangle((day-0.5, per-0.5), 1, run, facecolor=color, edgecolor='black')
                ax.add_patch(rect)
                ax.text(day, per + (run-1)/2, f"{nm}\n{room}\n{pf}", ha="center", va="center", fontsize=7)
                per += run
            else:
                per += 1

plt.tight_layout()
plt.show()
