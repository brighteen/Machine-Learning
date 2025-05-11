def number_of_shifts(df):  # 총 근무 교대 수를 계산하는 함수
    return df.sum()  # 모든 1 값의 합계 반환


def shift_deviations(df, mor_min, mor_max, day_min, day_max, evn_min, evn_max):  # 교대별 필요 인원 요구사항 위반 점수 계산 함수
    min_mor_dev = 0  # 아침 교대 최소 인원 위반 점수 초기화
    max_mor_dev = 0  # 아침 교대 최대 인원 위반 점수 초기화
    min_day_dev = 0  # 점심 교대 최소 인원 위반 점수 초기화
    max_day_dev = 0  # 점심 교대 최대 인원 위반 점수 초기화
    min_evn_dev = 0  # 저녁 교대 최소 인원 위반 점수 초기화
    max_evn_dev = 0  # 저녁 교대 최대 인원 위반 점수 초기화
    empty_penalty = 0  # 교대에 직원이 없을 경우 페널티 초기화
    for i in range(0, len(df)):  # 모든 교대에 대해 반복
        shift_ord = i % 3  # 교대 유형 결정 (0:아침, 1:점심, 2:저녁)
        empl_per_shift = df.sum(axis = 1)[i]  # 해당 교대에 배정된 직원 수 계산
        if shift_ord == 0:  # 아침 교대인 경우
            min_mor_dev += max(mor_min - empl_per_shift, 0)  # 최소 인원 미달 시 위반 점수 증가
            max_mor_dev += max(empl_per_shift - mor_max, 0)  # 최대 인원 초과 시 위반 점수 증가
        elif shift_ord == 1:  # 점심 교대인 경우
            min_day_dev += max(day_min - empl_per_shift, 0)  # 최소 인원 미달 시 위반 점수 증가
            max_day_dev += max(empl_per_shift - day_max, 0)  # 최대 인원 초과 시 위반 점수 증가
        elif shift_ord == 2:  # 저녁 교대인 경우
            min_evn_dev += max(evn_min - empl_per_shift, 0)  # 최소 인원 미달 시 위반 점수 증가
            max_evn_dev += max(empl_per_shift - evn_max, 0)  # 최대 인원 초과 시 위반 점수 증가
        if empl_per_shift == 0:  # 교대에 직원이 없는 경우
            empty_penalty += 100  # 큰 페널티 부여

    return min_mor_dev + max_mor_dev + min_day_dev + max_day_dev + min_evn_dev + max_evn_dev + empty_penalty  # 모든 위반 점수 합산하여 반환


def shift_relax(df, relax_after_mon, relax_after_day, relax_after_evn):  # 휴식 위반을 계산하는 함수
    violations = 0  # 위반 카운터 초기화
    for e in range(0, len(df.columns)):  # 각 직원에 대해 반복
        relax_counter = 0  # 필요 휴식 카운터 초기화
        for s in range(0, len(df)):  # 각 교대에 대해 반복
            shift = df.iloc[s, e]  # 해당 직원의 특정 교대 근무 여부 가져오기
            if shift == 1:  # 근무하는 경우
                if relax_counter > 0:  # 아직 휴식이 필요한 상태라면
                    violations += 1  # 위반 카운트 증가
                shift_order = s % 3  # 교대 유형 결정 (0:아침, 1:점심, 2:저녁)
                if shift_order == 0:  # 아침 교대인 경우
                    relax_counter = relax_after_mon  # 아침 교대 후 필요한 휴식 시간 설정
                elif shift_order == 1:  # 점심 교대인 경우
                    relax_counter = relax_after_day  # 점심 교대 후 필요한 휴식 시간 설정
                elif shift_order == 2:  # 저녁 교대인 경우
                    relax_counter = relax_after_evn  # 저녁 교대 후 필요한 휴식 시간 설정
            else:  # 근무하지 않는 경우
                relax_counter = max(0, relax_counter - 1)  # 필요 휴식 카운터 감소 (최소 0)
    return violations  # 총 휴식 위반 횟수 반환
