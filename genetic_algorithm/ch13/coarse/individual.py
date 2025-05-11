class Individual:  # 개체 클래스 정의
    cache = {}  # 적합도 값을 저장하는 캐시 딕셔너리
    cache_hit = 0  # 캐시 히트 카운터
    counter = 0  # 생성된 개체 수 카운터

    @classmethod
    def set_fitness_function(cls, fun):  # 적합도 함수 설정 클래스 메서드
        cls.fitness_function = fun  # 클래스 변수에 적합도 함수 할당

    def __init__(self, gene_list) -> None:  # 생성자 메서드
        coarsed_gene_list = [round(g) for g in gene_list]  # 유전자 값을 반올림하여 거칠게(coarse) 만듦
        self.gene_list = coarsed_gene_list  # 거친 유전자 리스트 저장
        gene_hash = ','.join([str(g) for g in coarsed_gene_list])  # 유전자를 문자열로 변환하여 해시 키 생성
        cache = self.__class__.cache  # 클래스 캐시 참조
        if gene_hash not in cache.keys():  # 캐시에 없으면 적합도를 계산하여 저장
            cache[gene_hash] =\
                self.__class__.fitness_function(coarsed_gene_list)  # 적합도 계산 및 캐시 저장
        else:  # 이미 계산된 적합도 값이 있으면 재사용
            self.__class__.cache_hit += 1  # 캐시 히트 카운터 증가

        self.fitness = cache[gene_hash]  # 개체의 적합도 값 저장
        self.__class__.counter += 1  # 개체 생성 카운터 증가
