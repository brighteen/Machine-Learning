import json  # JSON 파일 처리를 위한 라이브러리
import random  # 난수 생성을 위한 라이브러리


class Individual:  # 개체 클래스 정의 (간단한 버전)

    def __init__(self, gene_list) -> None:  # 생성자 메서드
        self.gene_list = gene_list  # 유전자 리스트 저장


def dump_population(population, path):  # 인구를 JSON 파일로 저장하는 함수
    ind_genes = [ind.gene_list for ind in population]  # 각 개체의 유전자만 추출
    with open(path, 'w') as f:  # 파일 열기
        json.dump(ind_genes, f)  # 유전자 리스트를 JSON 형식으로 저장


def restore_population(path):  # JSON 파일에서 인구를 복원하는 함수
    population = []  # 복원된 인구를 저장할 빈 리스트
    with open(path) as json_file:  # JSON 파일 열기
        ind_genes = json.load(json_file)  # JSON에서 유전자 리스트 로드
        for gene_list in ind_genes:  # 각 유전자 리스트에 대해
            population.append(Individual(gene_list))  # 개체 생성 및 인구에 추가
    return population  # 복원된 인구 반환


if __name__ == '__main__':  # 스크립트가 직접 실행될 때만 실행

    population = [Individual([random.randint(0, 100)]) for _ in range(100)]  # 100개의 무작위 개체로 인구 생성
    path = '/tmp/population_genes.json'  # 저장 경로 설정
    dump_population(population, path)  # 인구를 JSON 파일로 저장
    restored_population = restore_population(path)  # JSON 파일에서 인구 복원
