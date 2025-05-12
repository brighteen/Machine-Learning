# filepath: c:\Users\brigh\Documents\GitHub\Machine-Learning\genetic_algorithm\ch13\snapshot\test.py
import os
import json
from individual import Individual, restore_population

# JSON 파일 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, 'tmp', 'population_genes.json')

print(f"불러올 파일 경로: {json_path}")

# 1. 기본적인 방법으로 JSON 파일을 직접 읽기
try:
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    
    print("\n1. 직접 JSON 파일 읽기 결과:")
    print(f"총 {len(raw_data)}개의 개체 정보가 저장되어 있습니다.")
    print(f"처음 5개 개체의 유전자: {raw_data[:5]}")
    
    # 저장된 유전자의 통계 정보 계산
    gene_values = [gene[0] for gene in raw_data]  # 각 개체의 첫 번째(유일한) 유전자 값
    avg_gene = sum(gene_values) / len(gene_values)
    max_gene = max(gene_values)
    min_gene = min(gene_values)
    
    print(f"\n유전자 통계:")
    print(f"- 평균값: {avg_gene:.2f}")
    print(f"- 최대값: {max_gene}")
    print(f"- 최소값: {min_gene}")
    
except Exception as e:
    print(f"JSON 파일 읽기 중 오류 발생: {e}")

# 2. individual.py의 restore_population 함수를 사용하여 개체 복원
print("\n2. restore_population() 함수로 불러오기:")
try:
    restored_population = restore_population(json_path)
    
    if restored_population:
        print(f"복원된 인구 수: {len(restored_population)}")
        print("처음 5개 개체 정보:")
        
        for i, ind in enumerate(restored_population[:5]):
            print(f"개체 {i+1}: {ind}")
        
        # 무작위 선택 시뮬레이션 (간단한 예시)
        import random
        selected = random.choice(restored_population)
        print(f"\n무작위로 선택된 개체: {selected}")
        
except Exception as e:
    print(f"개체 복원 중 오류 발생: {e}")