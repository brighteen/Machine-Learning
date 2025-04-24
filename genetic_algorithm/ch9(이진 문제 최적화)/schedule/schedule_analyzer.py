# 주어진 스케줄 DataFrame에서 각 근무조(shift)마다 할당된 직원 수를 계산하는 함수
def number_of_shifts(df):
    return df.sum()

  
# shift_deviations 함수: 각 근무조별로 최소/최대 직원 수 조건에 어긋나는 정도를 계산  
# (예: 아침 근무는 최소 1, 최대 4명이어야 함 등)
def shift_deviations(df, mor_min, mor_max, day_min, day_max, evn_min, evn_max):
    min_mor_dev = 0
    max_mor_dev = 0
    min_day_dev = 0
    max_day_dev = 0
    min_evn_dev = 0
    max_evn_dev = 0
    empty_penalty = 0
    # 각 근무 슬롯(행)별로 반복 (총 3 * period 행)
    for i in range(0, len(df)):
        shift_ord = i % 3  # 근무조 순서: 0 → 아침, 1 → 낮, 2 → 저녁
        empl_per_shift = df.sum(axis = 1)[i]
        if shift_ord == 0:
            # 아침 근무: 최소 인원 미달 및 과다 배정 평가
            min_mor_dev += max(mor_min - empl_per_shift, 0)
            max_mor_dev += max(empl_per_shift - mor_max, 0)
        elif shift_ord == 1:
            # 낮 근무 평가
            min_day_dev += max(day_min - empl_per_shift, 0)
            max_day_dev += max(empl_per_shift - day_max, 0)
        elif shift_ord == 2:
            # 저녁 근무 평가
            min_evn_dev += max(evn_min - empl_per_shift, 0)
            max_evn_dev += max(empl_per_shift - evn_max, 0)
        # 만약 근무 슬롯에 직원이 전혀 배정되지 않았다면 큰 패널티 부여
        if empl_per_shift == 0:
            empty_penalty += 100

    # 각 조건 위반 정도를 합산하여 총 편차 점수 반환
    return min_mor_dev + max_mor_dev + min_day_dev + max_day_dev + min_evn_dev + max_evn_dev + empty_penalty

  
# shift_relax 함수: 연속 근무 후 휴식이 주어지지 않는 경우 위반 횟수를 계산  
# relax_after_mon, relax_after_day, relax_after_evn: 각 근무조 후에 필요한 휴식 횟수
def shift_relax(df, relax_after_mon, relax_after_day, relax_after_evn):
    violations = 0
    # 각 직원별로 스케줄 열(column) 반복
    for e in range(0, len(df.columns)):
        relax_counter = 0
        # 각 근무 슬롯(행)마다 반복
        for s in range(0, len(df)):
            shift = df.iloc[s, e]
            if shift == 1:
                # 근무가 시작되면 이전의 휴식 조건이 남아있다면 위반 처리
                if relax_counter > 0:
                    violations += 1
                shift_order = s % 3
                # 각 근무조별로 휴식 카운터 재설정
                if shift_order == 0:
                    relax_counter = relax_after_mon
                elif shift_order == 1:
                    relax_counter = relax_after_day
                elif shift_order == 2:
                    relax_counter = relax_after_evn
            else:
                # 근무가 없으면 휴식 카운터 감소 (최소 0)
                relax_counter = max(0, relax_counter - 1)
    return violations
