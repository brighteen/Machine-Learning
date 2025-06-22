#!/usr/bin/env python3
"""
신경망의 노드 가지치기를 위한 유전 알고리즘 (NNPGA)
작성자: 
날짜: 2025년 6월 22일
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split

from neural_network import NeuralNetwork
from genetic_algorithm import GeneticAlgorithm, Chromosome
from data_processor import DataProcessor
from evaluator import Evaluator

def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='신경망의 노드 가지치기를 위한 유전 알고리즘 (NNPGA)')
    
    parser.add_argument('--dataset', type=str, default='iris', choices=['iris', 'wine', 'digits', 'breast_cancer'], 
                        help='사용할 데이터셋 (기본값: iris)')
    parser.add_argument('--prune_rate', type=float, default=0.1, 
                        help='가지치기 비율 (0~1 사이, 기본값: 0.1)')
    parser.add_argument('--pop_size', type=int, default=20, 
                        help='유전 알고리즘 인구 크기 (기본값: 20)')
    parser.add_argument('--generations', type=int, default=150, 
                        help='유전 알고리즘 세대 수 (기본값: 150)')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='신경망 학습 에포크 수 (기본값: 100)')
    parser.add_argument('--crossover_rate', type=float, default=1.0, 
                        help='교차 확률 (0~1 사이, 기본값: 1.0)')
    parser.add_argument('--mutation_rate', type=float, default=0.01, 
                        help='돌연변이 확률 (0~1 사이, 기본값: 0.01)')
    parser.add_argument('--selection_pressure', type=float, default=0.25, 
                        help='선택 압력 (0~1 사이, 기본값: 0.25)')
    parser.add_argument('--test_size', type=float, default=0.2, 
                        help='테스트 세트 비율 (0~1 사이, 기본값: 0.2)')
    parser.add_argument('--random_state', type=int, default=42, 
                        help='랜덤 시드 (기본값: 42)')
    parser.add_argument('--verbose', action='store_true', 
                        help='상세 출력 모드')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='결과 저장 디렉토리 (기본값: results)')
    parser.add_argument('--k_folds', type=int, default=10, 
                        help='K-겹 교차검증의 폴드 수 (기본값: 10)')
    
    return parser.parse_args()

def load_dataset(dataset_name, random_state=42):
    """데이터셋 로드 및 전처리"""
    data_processor = DataProcessor()
    
    if dataset_name == 'iris':
        data = load_iris()
    elif dataset_name == 'wine':
        data = load_wine()
    elif dataset_name == 'digits':
        data = load_digits()
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
    else:
        raise ValueError(f"지원되지 않는 데이터셋: {dataset_name}")
    
    X, y = data.data, data.target
    num_features = X.shape[1]
    num_classes = len(np.unique(y))
    
    # 데이터 정규화
    X = data_processor.normalize_data(X)
    
    # 원-핫 인코딩
    y_onehot = data_processor.one_hot_encode(y, num_classes)
    
    # 데이터 분할
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=random_state
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'num_features': num_features,
        'num_classes': num_classes,
        'feature_names': data.feature_names,
        'target_names': data.target_names
    }

def run_nnpga_with_kfold(args):
    """10-겹 교차검증을 이용한 NNPGA 알고리즘 실행"""
    # 데이터셋 로드
    print(f"데이터셋 '{args.dataset}' 로드 중...")
    data_processor = DataProcessor()
    
    # UCI 데이터 로드
    if args.dataset == 'iris':
        data = load_iris()
    elif args.dataset == 'wine':
        data = load_wine()
    elif args.dataset == 'digits':
        data = load_digits()
    elif args.dataset == 'breast_cancer':
        data = load_breast_cancer()
    else:
        raise ValueError(f"지원되지 않는 데이터셋: {args.dataset}")
    
    X, y = data.data, data.target
    num_features = X.shape[1]
    num_classes = len(np.unique(y))
    
    # 데이터 정규화
    X = data_processor.normalize_data(X)
    
    # 원-핫 인코딩
    y_onehot = data_processor.one_hot_encode(y, num_classes)
    
    # 신경망 구조 설정
    input_size = num_features
    hidden_size = num_classes * 4  # 은닉층 노드 수는 출력층 노드 수의 4배로 설정
    output_size = num_classes
    
    print(f"신경망 구조: {input_size}-{hidden_size}-{output_size}")
    
    # 가지치기 비율에 따른 성능 비교 실험
    evaluator = Evaluator()
    prune_rates = [0.04, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    
    # 결과 저장할 디렉토리 생성
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 결과 저장할 파일
    result_file = os.path.join(args.output_dir, f"{args.dataset}_results.txt")
    
    with open(result_file, 'w') as f:
        f.write(f"데이터셋: {args.dataset}\n")
        f.write(f"신경망 구조: {input_size}-{hidden_size}-{output_size}\n")
        f.write(f"인구 크기: {args.pop_size}, 세대 수: {args.generations}, 에포크 수: {args.epochs}\n")
        f.write(f"교차 확률: {args.crossover_rate}, 돌연변이 확률: {args.mutation_rate}, 선택 압력: {args.selection_pressure}\n")
        f.write(f"{args.k_folds}-겹 교차검증 사용\n")
        f.write("\n가지치기 비율에 따른 성능 비교:\n")
        f.write("가지치기 비율 | 입력+은닉층 | 입력층 | 은닉층\n")
        f.write("-" * 50 + "\n")
    
    # K-겹 교차검증을 위한 데이터 분할
    fold_data = data_processor.k_fold_split(X, y_onehot, n_splits=args.k_folds, random_state=args.random_state)
    
    # 가지치기 방법별 결과 저장
    accuracies_ih = []  # 입력+은닉층 동시 가지치기
    accuracies_i = []   # 입력층만 가지치기
    accuracies_h = []   # 은닉층만 가지치기
    
    for prune_rate in prune_rates:
        print(f"\n가지치기 비율: {prune_rate:.2f}")
        
        # 각 가지치기 방법별 정확도 저장
        fold_accuracies_ih = []
        fold_accuracies_i = []
        fold_accuracies_h = []
        
        # 각 폴드에 대해 실험 수행
        for fold_idx, (X_train, y_train, X_val, y_val, X_test, y_test) in enumerate(fold_data):
            print(f"폴드 {fold_idx+1}/{args.k_folds} 실험 중...")
            
            # 1. 입력+은닉층 동시 가지치기
            print("입력+은닉층 동시 가지치기 실험 중...")
            
            # 유전 알고리즘 초기화
            ga_ih = GeneticAlgorithm(
                population_size=args.pop_size,
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                prune_rate=prune_rate,
                crossover_rate=args.crossover_rate,
                mutation_rate=args.mutation_rate,
                selection_pressure=args.selection_pressure
            )
            
            # 진화 실행
            best_chromosome_ih = ga_ih.evolve(
                X_train, y_train,
                X_val, y_val,
                generations=args.generations,
                epochs=args.epochs
            )
            
            # 성능 평가
            accuracy_ih, rmse_ih = evaluator.evaluate_model(
                best_chromosome_ih.neural_network,
                X_test,
                y_test
            )
            
            fold_accuracies_ih.append(accuracy_ih * 100)  # 퍼센트로 변환
            
            # 2. 입력층만 가지치기
            print("입력층만 가지치기 실험 중...")
            
            # 염색체 생성 (입력층만 가지치기)
            chromosome_i = Chromosome(input_size, hidden_size, prune_rate)
            chromosome_i.genes[input_size:] = 1  # 은닉층은 가지치기 안함
            
            # 신경망 생성 및 학습
            chromosome_i.neural_network = NeuralNetwork(input_size, hidden_size, output_size)
            chromosome_i.neural_network.train(
                X_train, y_train,
                epochs=args.epochs,
                validation_data=(X_val, y_val)
            )
            
            # 성능 평가
            accuracy_i, rmse_i = evaluator.evaluate_model(
                chromosome_i.neural_network,
                X_test,
                y_test
            )
            
            fold_accuracies_i.append(accuracy_i * 100)  # 퍼센트로 변환
            
            # 3. 은닉층만 가지치기
            print("은닉층만 가지치기 실험 중...")
            
            # 염색체 생성 (은닉층만 가지치기)
            chromosome_h = Chromosome(input_size, hidden_size, prune_rate)
            chromosome_h.genes[:input_size] = 1  # 입력층은 가지치기 안함
            
            # 신경망 생성 및 학습
            chromosome_h.neural_network = NeuralNetwork(input_size, hidden_size, output_size)
            chromosome_h.neural_network.train(
                X_train, y_train,
                epochs=args.epochs,
                validation_data=(X_val, y_val)
            )
            
            # 성능 평가
            accuracy_h, rmse_h = evaluator.evaluate_model(
                chromosome_h.neural_network,
                X_test,
                y_test
            )
            
            fold_accuracies_h.append(accuracy_h * 100)  # 퍼센트로 변환
        
        # 각 가지치기 방법별 평균 및 표준편차 계산
        mean_ih = np.mean(fold_accuracies_ih)
        std_ih = np.std(fold_accuracies_ih)
        
        mean_i = np.mean(fold_accuracies_i)
        std_i = np.std(fold_accuracies_i)
        
        mean_h = np.mean(fold_accuracies_h)
        std_h = np.std(fold_accuracies_h)
        
        # 평균 정확도 저장
        accuracies_ih.append(mean_ih)
        accuracies_i.append(mean_i)
        accuracies_h.append(mean_h)
        
        # 결과 저장
        with open(result_file, 'a') as f:
            f.write(f"{prune_rate:.2f} | {mean_ih:.2f}±{std_ih:.2f} | {mean_i:.2f}±{std_i:.2f} | {mean_h:.2f}±{std_h:.2f}\n")
        
        print(f"입력+은닉층: {mean_ih:.2f}±{std_ih:.2f}%, 입력층: {mean_i:.2f}±{std_i:.2f}%, 은닉층: {mean_h:.2f}±{std_h:.2f}%")
    
    # 그래프 그리기
    evaluator.plot_accuracy_vs_prune_rate(prune_rates, accuracies_ih, accuracies_i, accuracies_h)
    print(f"그래프가 '{args.output_dir}/accuracy_vs_prune_rate.png'에 저장되었습니다.")
    
    # 최적의 가지치기 비율 찾기
    best_idx_ih = np.argmax(accuracies_ih)
    best_idx_i = np.argmax(accuracies_i)
    best_idx_h = np.argmax(accuracies_h)
    
    best_rate_ih = prune_rates[best_idx_ih]
    best_rate_i = prune_rates[best_idx_i]
    best_rate_h = prune_rates[best_idx_h]
    
    print("\n최적의 가지치기 비율:")
    print(f"입력+은닉층: {best_rate_ih:.2f} ({accuracies_ih[best_idx_ih]:.2f}%)")
    print(f"입력층: {best_rate_i:.2f} ({accuracies_i[best_idx_i]:.2f}%)")
    print(f"은닉층: {best_rate_h:.2f} ({accuracies_h[best_idx_h]:.2f}%)")
    
    # 최종 결과 저장
    with open(result_file, 'a') as f:
        f.write("\n최적의 가지치기 비율:\n")
        f.write(f"입력+은닉층: {best_rate_ih:.2f} ({accuracies_ih[best_idx_ih]:.2f}%)\n")
        f.write(f"입력층: {best_rate_i:.2f} ({accuracies_i[best_idx_i]:.2f}%)\n")
        f.write(f"은닉층: {best_rate_h:.2f} ({accuracies_h[best_idx_h]:.2f}%)\n")
    
    print(f"결과가 '{result_file}'에 저장되었습니다.")
    
    return {
        'best_rate_ih': best_rate_ih,
        'best_acc_ih': accuracies_ih[best_idx_ih],
        'best_rate_i': best_rate_i,
        'best_acc_i': accuracies_i[best_idx_i],
        'best_rate_h': best_rate_h,
        'best_acc_h': accuracies_h[best_idx_h]
    }

def run_nnpga(args):
    """NNPGA 알고리즘 실행"""
    # 데이터셋 로드
    print(f"데이터셋 '{args.dataset}' 로드 중...")
    data = load_dataset(args.dataset, args.random_state)
    
    # 신경망 구조 설정
    input_size = data['num_features']
    hidden_size = data['num_classes'] * 4  # 은닉층 노드 수는 출력층 노드 수의 4배로 설정
    output_size = data['num_classes']
    
    print(f"신경망 구조: {input_size}-{hidden_size}-{output_size}")
    
    # 가지치기 비율에 따른 성능 비교 실험
    evaluator = Evaluator()
    prune_rates = [0.04, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    
    # 결과 저장할 디렉토리 생성
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 결과 저장할 파일
    result_file = os.path.join(args.output_dir, f"{args.dataset}_results.txt")
    
    with open(result_file, 'w') as f:
        f.write(f"데이터셋: {args.dataset}\n")
        f.write(f"신경망 구조: {input_size}-{hidden_size}-{output_size}\n")
        f.write(f"인구 크기: {args.pop_size}, 세대 수: {args.generations}, 에포크 수: {args.epochs}\n")
        f.write(f"교차 확률: {args.crossover_rate}, 돌연변이 확률: {args.mutation_rate}, 선택 압력: {args.selection_pressure}\n")
        f.write("\n가지치기 비율에 따른 성능 비교:\n")
        f.write("가지치기 비율 | 입력+은닉층 | 입력층 | 은닉층\n")
        f.write("-" * 50 + "\n")
    
    # 가지치기 방법별 결과 저장
    accuracies_ih = []  # 입력+은닉층 동시 가지치기
    accuracies_i = []   # 입력층만 가지치기
    accuracies_h = []   # 은닉층만 가지치기
    
    for prune_rate in prune_rates:
        print(f"\n가지치기 비율: {prune_rate:.2f}")
        
        # 1. 입력+은닉층 동시 가지치기
        print("입력+은닉층 동시 가지치기 실험 중...")
        
        # 유전 알고리즘 초기화
        ga_ih = GeneticAlgorithm(
            population_size=args.pop_size,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            prune_rate=prune_rate,
            crossover_rate=args.crossover_rate,
            mutation_rate=args.mutation_rate,
            selection_pressure=args.selection_pressure
        )
        
        # 진화 실행
        best_chromosome_ih = ga_ih.evolve(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            generations=args.generations,
            epochs=args.epochs
        )
        
        # 성능 평가
        accuracy_ih, rmse_ih = evaluator.evaluate_model(
            best_chromosome_ih.neural_network,
            data['X_test'],
            data['y_test']
        )
        
        accuracies_ih.append(accuracy_ih * 100)  # 퍼센트로 변환
        
        # 2. 입력층만 가지치기
        print("입력층만 가지치기 실험 중...")
        
        # 염색체 생성 (입력층만 가지치기)
        chromosome_i = Chromosome(input_size, hidden_size, prune_rate)
        chromosome_i.genes[input_size:] = 1  # 은닉층은 가지치기 안함
        
        # 신경망 생성 및 학습
        chromosome_i.neural_network = NeuralNetwork(input_size, hidden_size, output_size)
        chromosome_i.neural_network.train(
            data['X_train'], data['y_train'],
            epochs=args.epochs,
            validation_data=(data['X_val'], data['y_val'])
        )
        
        # 성능 평가
        accuracy_i, rmse_i = evaluator.evaluate_model(
            chromosome_i.neural_network,
            data['X_test'],
            data['y_test']
        )
        
        accuracies_i.append(accuracy_i * 100)  # 퍼센트로 변환
        
        # 3. 은닉층만 가지치기
        print("은닉층만 가지치기 실험 중...")
        
        # 염색체 생성 (은닉층만 가지치기)
        chromosome_h = Chromosome(input_size, hidden_size, prune_rate)
        chromosome_h.genes[:input_size] = 1  # 입력층은 가지치기 안함
        
        # 신경망 생성 및 학습
        chromosome_h.neural_network = NeuralNetwork(input_size, hidden_size, output_size)
        chromosome_h.neural_network.train(
            data['X_train'], data['y_train'],
            epochs=args.epochs,
            validation_data=(data['X_val'], data['y_val'])
        )
        
        # 성능 평가
        accuracy_h, rmse_h = evaluator.evaluate_model(
            chromosome_h.neural_network,
            data['X_test'],
            data['y_test']
        )
        
        accuracies_h.append(accuracy_h * 100)  # 퍼센트로 변환
        
        # 결과 저장
        with open(result_file, 'a') as f:
            f.write(f"{prune_rate:.2f} | {accuracy_ih*100:.2f}±{0:.2f} | {accuracy_i*100:.2f}±{0:.2f} | {accuracy_h*100:.2f}±{0:.2f}\n")
        
        print(f"입력+은닉층: {accuracy_ih*100:.2f}%, 입력층: {accuracy_i*100:.2f}%, 은닉층: {accuracy_h*100:.2f}%")
    
    # 그래프 그리기
    evaluator.plot_accuracy_vs_prune_rate(prune_rates, accuracies_ih, accuracies_i, accuracies_h)
    print(f"그래프가 '{args.output_dir}/accuracy_vs_prune_rate.png'에 저장되었습니다.")
    
    # 최적의 가지치기 비율 찾기
    best_idx_ih = np.argmax(accuracies_ih)
    best_idx_i = np.argmax(accuracies_i)
    best_idx_h = np.argmax(accuracies_h)
    
    best_rate_ih = prune_rates[best_idx_ih]
    best_rate_i = prune_rates[best_idx_i]
    best_rate_h = prune_rates[best_idx_h]
    
    print("\n최적의 가지치기 비율:")
    print(f"입력+은닉층: {best_rate_ih:.2f} ({accuracies_ih[best_idx_ih]:.2f}%)")
    print(f"입력층: {best_rate_i:.2f} ({accuracies_i[best_idx_i]:.2f}%)")
    print(f"은닉층: {best_rate_h:.2f} ({accuracies_h[best_idx_h]:.2f}%)")
    
    # 최종 결과 저장
    with open(result_file, 'a') as f:
        f.write("\n최적의 가지치기 비율:\n")
        f.write(f"입력+은닉층: {best_rate_ih:.2f} ({accuracies_ih[best_idx_ih]:.2f}%)\n")
        f.write(f"입력층: {best_rate_i:.2f} ({accuracies_i[best_idx_i]:.2f}%)\n")
        f.write(f"은닉층: {best_rate_h:.2f} ({accuracies_h[best_idx_h]:.2f}%)\n")
    
    print(f"결과가 '{result_file}'에 저장되었습니다.")
    
    return {
        'best_rate_ih': best_rate_ih,
        'best_acc_ih': accuracies_ih[best_idx_ih],
        'best_rate_i': best_rate_i,
        'best_acc_i': accuracies_i[best_idx_i],
        'best_rate_h': best_rate_h,
        'best_acc_h': accuracies_h[best_idx_h]
    }

def main():
    """메인 함수"""
    # 명령줄 인수 파싱
    args = parse_arguments()
    
    # 시작 시간 기록
    start_time = time.time()
    
    # 10-겹 교차검증을 이용한 NNPGA 알고리즘 실행
    results = run_nnpga_with_kfold(args)
    
    # 종료 시간 기록
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n총 실행 시간: {elapsed_time:.2f}초")
    
    # 최종 결과 출력
    print("\n최종 결과:")
    print(f"데이터셋: {args.dataset}")
    print(f"입력+은닉층 최적 가지치기 비율: {results['best_rate_ih']:.2f} (정확도: {results['best_acc_ih']:.2f}%)")
    print(f"입력층 최적 가지치기 비율: {results['best_rate_i']:.2f} (정확도: {results['best_acc_i']:.2f}%)")
    print(f"은닉층 최적 가지치기 비율: {results['best_rate_h']:.2f} (정확도: {results['best_acc_h']:.2f}%)")

if __name__ == "__main__":
    main()
