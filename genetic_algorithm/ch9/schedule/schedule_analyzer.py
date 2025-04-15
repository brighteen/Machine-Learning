# 스케줄의 각 근무조에 대해 총 근무자 수 계산
def number_of_shifts(df):
    return df.sum()

# shift_deviations: 각 근무조(아침, 낮, 저녁)별로 최소/최대 인원 차이를 계산하고,
# 모든 근무조가 비어 있을 경우 추가 벌점 부여
def shift_deviations(df, mor_min, mor_max, day_min, day_max, evn_min, evn_max):
    min_mor_dev = 0
    max_mor_dev = 0
    min_day_dev = 0
    max_day_dev = 0
    min_evn_dev = 0
    max_evn_dev = 0
    empty_penalty = 0
    for i in range(0, len(df)):
        shift_ord = i % 3  # 0: 아침, 1: 낮, 2: 저녁
        empl_per_shift = df.sum(axis=1)[i]
        if shift_ord == 0:
            min_mor_dev += max(mor_min - empl_per_shift, 0)
            max_mor_dev += max(empl_per_shift - mor_max, 0)
        elif shift_ord == 1:
            min_day_dev += max(day_min - empl_per_shift, 0)
            max_day_dev += max(empl_per_shift - day_max, 0)
        elif shift_ord == 2:
            min_evn_dev += max(evn_min - empl_per_shift, 0)
            max_evn_dev += max(empl_per_shift - evn_max, 0)
        if empl_per_shift == 0:
            empty_penalty += 100   # 근무조가 비면 큰 벌점 추가
    return min_mor_dev + max_mor_dev + min_day_dev + max_day_dev + min_evn_dev + max_evn_dev + empty_penalty

# shift_relax: 일정 근무 후 적절한 휴식이 이루어지지 않은 경우 벌점 계산
def shift_relax(df, relax_after_mon, relax_after_day, relax_after_evn):
    violations = 0
    for e in range(0, len(df.columns)):
        relax_counter = 0
        for s in range(0, len(df)):
            shift = df.iloc[s, e]
            if shift == 1:
                if relax_counter > 0:
                    violations += 1
                shift_order = s % 3
                if shift_order == 0:
                    relax_counter = relax_after_mon
                elif shift_order == 1:
                    relax_counter = relax_after_day
                elif shift_order == 2:
                    relax_counter = relax_after_evn
            else:
                relax_counter = max(0, relax_counter - 1)
    return violations
