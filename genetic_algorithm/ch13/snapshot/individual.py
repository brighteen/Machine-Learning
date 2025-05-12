import json  # JSON 파일 처리를 위한 라이브러리
import random  # 난수 생성을 위한 라이브러리


class Individual:  # 개체 클래스 정의 (간단한 버전)

    def __init__(self, gene_list) -> None:  # 생성자 메서드
        if not isinstance(gene_list, list):  # 유전자 리스트 유효성 검사
            raise TypeError("gene_list는 리스트 형태여야 합니다")
        self.gene_list = gene_list  # 유전자 리스트 저장
        
    def __str__(self) -> str:  # 문자열 표현 메서드
        return f"Individual(genes={self.gene_list})"


def dump_population(population, path):  # 인구를 JSON 파일로 저장하는 함수
    ind_genes = [ind.gene_list for ind in population]  # 각 개체의 유전자만 추출
    try:
        # 디렉토리가 존재하는지 확인하고 생성
        import os
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(path, 'w') as f:  # 파일 열기
            json.dump(ind_genes, f)  # 유전자 리스트를 JSON 형식으로 저장
    except Exception as e:
        print(f"파일 저장 중 오류 발생: {e}")


def restore_population(path):  # JSON 파일에서 인구를 복원하는 함수
    population = []  # 복원된 인구를 저장할 빈 리스트
    try:
        with open(path) as json_file:  # JSON 파일 열기
            ind_genes = json.load(json_file)  # JSON에서 유전자 리스트 로드
            for gene_list in ind_genes:  # 각 유전자 리스트에 대해
                population.append(Individual(gene_list))  # 개체 생성 및 인구에 추가
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {path}")
    except json.JSONDecodeError:
        print(f"유효하지 않은 JSON 형식입니다: {path}")
    except Exception as e:
        print(f"파일 복원 중 오류 발생: {e}")
    return population  # 복원된 인구 반환


if __name__ == '__main__':  # 스크립트가 직접 실행될 때만 실행
    import os
    
    population = [Individual([random.randint(0, 100)]) for _ in range(100)]  # 100개의 무작위 개체로 인구 생성
    
    # 현재 파일 위치 기준으로 tmp 폴더 생성 및 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(script_dir, 'tmp')
    
    # tmp 디렉토리가 없으면 생성
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
        
    path = os.path.join(tmp_dir, 'population_genes.json')  # tmp 폴더 안에 JSON 파일 저장
    
    print(f"파일 저장 경로: {path}")
    dump_population(population, path)  # 인구를 JSON 파일로 저장
    
    restored_population = restore_population(path)  # JSON 파일에서 인구 복원
    
    if restored_population:
        print(f"복원된 인구 수: {len(restored_population)}")
        # 첫 번째 개체의 유전자 출력 (예시)
        if restored_population:
            print(f"첫 번째 개체의 유전자: {restored_population[0].gene_list}")
