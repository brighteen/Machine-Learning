#!/usr/bin/env python3
"""
신경망의 노드 가지치기를 위한 유전 알고리즘 (NNPGA)
여러 데이터셋에 대한 일괄 실행 스크립트
"""

import subprocess
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.font_manager as fm

# 한글 폰트 설정
def configure_matplotlib_for_korean():
    """matplotlib에서 한글을 사용할 수 있도록 설정합니다."""
    # 기본 폰트를 sans-serif로 설정
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['NanumGothic', 'DejaVu Sans', 'Arial', 'Verdana']
    
    # 폰트 경로 탐색
    font_path = None
    
    # 일반적인 폰트 경로들 탐색
    paths_to_check = [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/TTF/DejaVuSans.ttf',
        '/usr/share/fonts/dejavu/DejaVuSans.ttf'
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            font_path = path
            break
    
    # 폰트를 찾았으면 설정
    if font_path:
        font_prop = fm.FontProperties(fname=font_path)
        matplotlib.rcParams['font.family'] = font_prop.get_name()
    else:
        # 한글 폰트를 찾지 못했으면 영어로 레이블 사용
        print("경고: 한글 폰트를 찾을 수 없습니다. 영어 레이블을 사용합니다.")
        
    # matplotlib에서 유니코드 문자를 사용할 수 있도록 설정
    matplotlib.rcParams['axes.unicode_minus'] = False

def run_experiment(dataset, prune_rate=0.1, pop_size=20, generations=30, epochs=50, 
                  crossover_rate=1.0, mutation_rate=0.01, selection_pressure=0.25,
                  k_folds=10, output_dir='results'):
    """
    단일 데이터셋에 대한 실험 실행
    """
    cmd = [
        'python', 'main.py',
        '--dataset', dataset,
        '--prune_rate', str(prune_rate),
        '--pop_size', str(pop_size),
        '--generations', str(generations),
        '--epochs', str(epochs),
        '--crossover_rate', str(crossover_rate),
        '--mutation_rate', str(mutation_rate),
        '--selection_pressure', str(selection_pressure),
        '--k_folds', str(k_folds),
        '--output_dir', output_dir
    ]
    
    print(f"\n\n{'='*50}")
    print(f"데이터셋 '{dataset}' 실험 시작")
    print(f"{'='*50}")
    
    start_time = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    
    # 실시간으로 출력 표시
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    # 오류 메시지 출력
    for line in process.stderr:
        print(line, end='')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n{'='*50}")
    print(f"데이터셋 '{dataset}' 실험 완료 (소요시간: {elapsed_time:.2f}초)")
    print(f"{'='*50}")
    
    return elapsed_time

def collect_results(output_dir='results'):
    """
    모든 결과 파일에서 최적 가지치기 비율과 정확도 수집
    """
    result_files = [f for f in os.listdir(output_dir) if f.endswith('_results.txt')]
    
    if not result_files:
        print("결과 파일을 찾을 수 없습니다.")
        return None
    
    results = []
    
    for file in result_files:
        dataset = file.replace('_results.txt', '')
        file_path = os.path.join(output_dir, file)
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # 최적 가지치기 비율 찾기
        for i, line in enumerate(lines):
            if "최적의 가지치기 비율:" in line:
                # 다음 3줄에서 정보 추출
                input_hidden = lines[i+1].strip().split(":")
                input_only = lines[i+2].strip().split(":")
                hidden_only = lines[i+3].strip().split(":")
                
                # 비율과 정확도 추출
                ih_rate = float(input_hidden[1].split()[0])
                ih_acc = float(input_hidden[1].split()[1].strip('()%'))
                
                i_rate = float(input_only[1].split()[0])
                i_acc = float(input_only[1].split()[1].strip('()%'))
                
                h_rate = float(hidden_only[1].split()[0])
                h_acc = float(hidden_only[1].split()[1].strip('()%'))
                
                results.append({
                    'dataset': dataset,
                    'ih_rate': ih_rate,
                    'ih_acc': ih_acc,
                    'i_rate': i_rate,
                    'i_acc': i_acc,
                    'h_rate': h_rate,
                    'h_acc': h_acc
                })
                break
    
    return pd.DataFrame(results)

def plot_summary(results_df, output_dir='results'):
    """
    모든 데이터셋의 결과를 요약한 그래프 생성
    """
    # 한글 폰트 설정
    configure_matplotlib_for_korean()
    
    # 항상 영어 레이블 사용
    use_english_labels = True
    
    # 1. 데이터셋별 최적 정확도 비교
    plt.figure(figsize=(12, 6))
    
    datasets = results_df['dataset']
    x = np.arange(len(datasets))
    width = 0.25
    
    # 항상 영어 레이블 사용
    plt.bar(x - width, results_df['ih_acc'], width, label='Input+Hidden Layers')
    plt.bar(x, results_df['i_acc'], width, label='Input Layer')
    plt.bar(x + width, results_df['h_acc'], width, label='Hidden Layer')
    
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison by Dataset')
    
    plt.xticks(x, datasets, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()
    
    # 2. 데이터셋별 최적 가지치기 비율 비교
    plt.figure(figsize=(12, 6))
    
    # 항상 영어 레이블 사용
    plt.bar(x - width, results_df['ih_rate'], width, label='Input+Hidden Layers')
    plt.bar(x, results_df['i_rate'], width, label='Input Layer')
    plt.bar(x + width, results_df['h_rate'], width, label='Hidden Layer')
    
    plt.xlabel('Dataset')
    plt.ylabel('Optimal Pruning Rate')
    plt.title('Optimal Pruning Rate Comparison by Dataset')
    
    plt.xticks(x, datasets, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prune_rate_comparison.png'))
    plt.close()
    
    return True

def export_results_to_csv(results_df, output_dir='results'):
    """
    결과를 CSV 파일로 내보내기
    """
    output_path = os.path.join(output_dir, 'all_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"모든 결과가 '{output_path}'에 저장되었습니다.")
    
    # 요약 통계도 저장
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    
    # 방법별 평균 정확도 및 표준편차 계산
    summary = {
        'metric': ['Average Accuracy', 'Standard Deviation', 'Min Value', 'Max Value', 'Average Pruning Rate'],
        'input_hidden': [
            results_df['ih_acc'].mean(),
            results_df['ih_acc'].std(),
            results_df['ih_acc'].min(),
            results_df['ih_acc'].max(),
            results_df['ih_rate'].mean()
        ],
        'input_only': [
            results_df['i_acc'].mean(),
            results_df['i_acc'].std(),
            results_df['i_acc'].min(),
            results_df['i_acc'].max(),
            results_df['i_rate'].mean()
        ],
        'hidden_only': [
            results_df['h_acc'].mean(),
            results_df['h_acc'].std(),
            results_df['h_acc'].min(),
            results_df['h_acc'].max(),
            results_df['h_rate'].mean()
        ]
    }
    
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"요약 통계가 '{summary_path}'에 저장되었습니다.")

def main():
    """
    메인 함수 - 여러 데이터셋에 대한 실험 실행
    """
    # 한글 폰트 설정
    configure_matplotlib_for_korean()
    
    # 실험할 데이터셋 목록
    datasets = ['iris', 'wine', 'digits', 'breast_cancer']  # 논문과 동일하게 설정
    
    # 결과 디렉토리 생성
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 데이터셋별 실행 시간 기록
    execution_times = {}
    
    # 모든 데이터셋에 대해 실험 실행
    for dataset in datasets:
        elapsed_time = run_experiment(
            dataset, 
            generations=150, # 논문과 동일하게 설정
            epochs=100,      # 논문과 동일하게 설정
            pop_size=20,     # 논문과 동일하게 설정
            k_folds=10,      # 10겹 교차검증 (논문과 동일)
            output_dir=output_dir
        )
        execution_times[dataset] = elapsed_time
    
    # 실행 시간 출력
    print("\n모든 데이터셋 실험 완료!")
    print("\n데이터셋별 실행 시간:")
    for dataset, elapsed_time in execution_times.items():
        print(f"{dataset}: {elapsed_time:.2f}초")
    
    # 결과 수집 및 시각화
    results_df = collect_results(output_dir)
    
    if results_df is not None:
        # 결과 출력
        print("\n모든 데이터셋 실험 결과:")
        print(results_df)
        
        # 그래프 생성
        plot_summary(results_df, output_dir)
        
        # CSV로 내보내기
        export_results_to_csv(results_df, output_dir)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n총 소요 시간: {total_time:.2f}초")
