import copy  # 깊은 복사를 위한 라이브러리
import random  # 난수 생성을 위한 라이브러리
import matplotlib.pyplot as plt  # 그래프 작성을 위한 라이브러리

from individual import Individual  # Individual 클래스 가져오기


def selection_rank_with_elite(individuals, elite_size = 0):  # 엘리트 보존과 랭크 기반 선택 함수
    sorted_individuals = sorted(individuals, key = lambda ind: ind.fitness, reverse = True)  # 적합도 기준 내림차순 정렬
    rank_distance = 1 / len(individuals)  # 랭크 간 거리 계산
    ranks = [(1 - i * rank_distance) for i in range(len(individuals))]  # 선형적으로 감소하는 랭크 값 생성
    ranks_sum = sum(ranks)  # 모든 랭크 값의 합계
    selected = sorted_individuals[0:elite_size]  # 최상위 elite_size 개체를 무조건 선택

    for i in range(len(sorted_individuals) - elite_size):  # 나머지 자리 채우기
        shave = random.random() * ranks_sum  # 0에서 rank_sum 사이의 임의 값
        rank_sum = 0  # 누적 랭크 합계 초기화
        for i in range(len(sorted_individuals)):  # 룰렛휠 선택 방식으로 개체 선택
            rank_sum += ranks[i]  # 랭크 값 누적
            if rank_sum > shave:  # 누적 합이 임의 값을 초과하면
                selected.append(sorted_individuals[i])  # 해당 개체 선택
                break  # 반복 중단

    return selected  # 선택된 개체 리스트 반환


def crossover_n_point(p1, p2, n):  # n점 교차 함수
    ps = random.sample(range(1, len(p1) - 1), n)  # n개의 무작위 교차 지점 선택
    ps.append(0)  # 시작점 추가
    ps.append(len(p1))  # 끝점 추가
    ps = sorted(ps)  # 교차 지점 오름차순 정렬
    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)  # 부모 유전자 깊은 복사
    for i in range(0, n + 1):  # 각 세그먼트에 대해 교차 수행
        if i % 2 == 0:  # 짝수 인덱스 세그먼트는 건너뛰기
            continue
        c1[ps[i]:ps[i + 1]] = p2[ps[i]:ps[i + 1]]  # 첫 번째 자식의 해당 세그먼트를 두 번째 부모 것으로 교체
        c2[ps[i]:ps[i + 1]] = p1[ps[i]:ps[i + 1]]  # 두 번째 자식의 해당 세그먼트를 첫 번째 부모 것으로 교체
    return [c1, c2]  # 두 자식 유전자 반환


def crossover_fitness_driven_one_point(p1, p2):  # 적합도 기반 1점 교차 함수
    point = random.randint(1, len(p1.gene_list) - 1)  # 무작위 교차 지점 선택
    c1, c2 = copy.deepcopy(p1.gene_list), copy.deepcopy(p2.gene_list)  # 부모 유전자 깊은 복사
    c1[point:], c2[point:] = p2.gene_list[point:], p1.gene_list[point:]  # 교차 지점 이후 부분 교환
    child1 = Individual(c1)  # 첫 번째 자식 개체 생성
    child2 = Individual(c2)  # 두 번째 자식 개체 생성
    candidates = [child1, child2, p1, p2]  # 자식과 부모 모두 후보로 포함

    best = sorted(candidates, key = lambda ind: ind.fitness, reverse = True)  # 적합도 기준 내림차순 정렬

    return best[0:2]  # 상위 2개 개체만 선택하여 반환


def mutation_bit_flip(ind):  # 비트 플립 돌연변이 함수
    mut = copy.deepcopy(ind)  # 개체 유전자 깊은 복사
    pos = random.randint(0, len(ind) - 1)  # 무작위 위치 선택
    g1 = mut[pos]  # 선택된 위치의 유전자 값 저장
    mut[pos] = (g1 + 1) % 2  # 이진 비트 반전 (0->1, 1->0)
    return mut  # 변형된 유전자 반환


def mutation_shuffle(ind):  # 구간 셔플 돌연변이 함수
    mut = copy.deepcopy(ind)  # 개체 유전자 깊은 복사
    pos = sorted(random.sample(range(0, len(mut)), 2))  # 두 개의 무작위 위치 선택 (오름차순 정렬)
    subrange = mut[pos[0]:pos[1] + 1]  # 선택된 위치 사이의 유전자 부분 추출
    random.shuffle(subrange)  # 추출된 부분을 무작위로 섞기
    mut[pos[0]:pos[1] + 1] = subrange  # 섞인 부분을 원래 위치에 삽입

    return mut  # 변형된 유전자 반환


def mutation_fitness_driven_bit_flip(ind, max_tries = 3):  # 적합도 기반 비트 플립 돌연변이 함수
    for t in range(0, max_tries):  # 최대 시도 횟수만큼 반복
        mut = copy.deepcopy(ind.gene_list)  # 개체 유전자 깊은 복사
        pos = random.randint(0, len(ind.gene_list) - 1)  # 무작위 위치 선택
        g1 = mut[pos]  # 선택된 위치의 유전자 값 저장
        mut[pos] = (g1 + 1) % 2  # 이진 비트 반전
        mutated = Individual(mut)  # 변형된 유전자로 새 개체 생성
        if mutated.fitness > ind.fitness:  # 적합도가 향상되었는지 확인
            return mutated  # 적합도가 향상된 경우 해당 개체 반환
    return ind  # 어떤 돌연변이도 적합도 향상을 가져오지 않으면 원본 개체 반환


def crossover_operation(population, method, prob):  # 교차 연산 적용 함수
    crossed_offspring = []  # 자식 개체를 저장할 리스트
    for ind1, ind2 in zip(population[::2], population[1::2]):  # 인구를 페어링 (짝수-홀수 인덱스 쌍)
        if random.random() < prob:  # 교차 확률에 따라 교차 여부 결정
            kid1, kid2 = method(ind1, ind2)  # 교차 메서드 적용
            crossed_offspring.append(kid1)  # 첫 번째 자식 추가
            crossed_offspring.append(kid2)  # 두 번째 자식 추가
        else:  # 교차가 발생하지 않는 경우
            crossed_offspring.append(ind1)  # 첫 번째 부모를 그대로 전달
            crossed_offspring.append(ind2)  # 두 번째 부모를 그대로 전달
    return crossed_offspring  # 교차 후 개체 리스트 반환


def mutation_operation(population, method, prob):  # 돌연변이 연산 적용 함수
    mutated_offspring = []  # 돌연변이 적용 개체를 저장할 리스트
    for mutant in population:  # 모든 개체에 대해 반복
        if random.random() < prob:  # 돌연변이 확률에 따라 돌연변이 여부 결정
            new_mutant = method(mutant)  # 돌연변이 메서드 적용
            mutated_offspring.append(new_mutant)  # 돌연변이 개체 추가
        else:  # 돌연변이가 발생하지 않는 경우
            mutated_offspring.append(mutant)  # 원래 개체를 그대로 전달
    return mutated_offspring  # 돌연변이 후 개체 리스트 반환


def stats(population, best_ind, fit_avg, fit_best):  # 통계 정보 업데이트 함수
    best_of_generation = max(population, key = lambda ind: ind.fitness)  # 현재 세대 최고 개체 찾기
    if best_ind.fitness < best_of_generation.fitness:  # 역대 최고 개체와 비교
        best_ind = best_of_generation  # 더 좋은 개체가 있으면 업데이트
    fit_avg.append(sum([ind.fitness for ind in population]) / len(population))  # 현재 세대 평균 적합도 기록
    fit_best.append(best_ind.fitness)  # 역대 최고 적합도 기록

    return best_ind, fit_avg, fit_best  # 업데이트된 통계 정보 반환


def plot_stats(fit_avg, fit_best, title):  # 적합도 통계 그래프 출력 함수
    plt.plot(fit_avg, label = "Average Fitness of Generation")  # 평균 적합도 그래프
    plt.plot(fit_best, label = "Best Fitness")  # 최고 적합도 그래프
    plt.title(title)  # 그래프 제목 설정
    plt.legend(loc = "lower right")  # 범례 위치 설정
    plt.show()  # 그래프 표시
    plt.close()  # 그래프 창 닫기
