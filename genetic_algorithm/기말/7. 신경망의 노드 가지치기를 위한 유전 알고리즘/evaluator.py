import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
from scipy import stats
import os

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

class Evaluator:
    def __init__(self):
        """성능 평가기 초기화"""
        # 한글 폰트 설정
        configure_matplotlib_for_korean()
    
    def calculate_accuracy(self, y_true, y_pred):
        """
        정확도 계산
        :param y_true: 실제 레이블 (원-핫 인코딩)
        :param y_pred: 예측 레이블 (원-핫 인코딩)
        :return: 정확도
        """
        # 원-핫 인코딩된 레이블을 클래스 인덱스로 변환
        y_true_classes = np.argmax(y_true, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # 정확도 계산
        return accuracy_score(y_true_classes, y_pred_classes)
    
    def calculate_rmse(self, y_true, y_pred):
        """
        RMSE 계산
        :param y_true: 실제 레이블
        :param y_pred: 예측 레이블
        :return: RMSE
        """
        return np.sqrt(np.mean(np.square(y_true - y_pred)))
    
    def evaluate_model(self, model, X_test, y_test):
        """
        모델 평가
        :param model: 신경망 모델
        :param X_test: 테스트 데이터
        :param y_test: 테스트 레이블
        :return: 정확도, RMSE
        """
        # 예측
        y_pred = model.predict(X_test)
        
        # 정확도 계산
        accuracy = self.calculate_accuracy(y_test, y_pred)
        
        # RMSE 계산
        rmse = self.calculate_rmse(y_test, y_pred)
        
        return accuracy, rmse
    
    def print_classification_report(self, y_true, y_pred, target_names=None):
        """
        분류 보고서 출력
        :param y_true: 실제 레이블 (원-핫 인코딩)
        :param y_pred: 예측 레이블 (원-핫 인코딩)
        :param target_names: 클래스 이름 목록
        """
        # 원-핫 인코딩된 레이블을 클래스 인덱스로 변환
        y_true_classes = np.argmax(y_true, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # 분류 보고서 출력
        print(classification_report(y_true_classes, y_pred_classes, target_names=target_names))
        
        # 혼동 행렬 출력
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        print("Confusion Matrix:")
        print(cm)
    
    def plot_accuracy_vs_prune_rate(self, prune_rates, accuracies_ih, accuracies_i, accuracies_h):
        """
        가지치기 비율에 따른 정확도 그래프 출력
        :param prune_rates: 가지치기 비율 목록
        :param accuracies_ih: 입력+은닉층 동시 가지치기 정확도
        :param accuracies_i: 입력층 가지치기 정확도
        :param accuracies_h: 은닉층 가지치기 정확도
        """
        plt.figure(figsize=(10, 6))
        
        # 항상 영어 레이블 사용
        plt.plot(prune_rates, accuracies_ih, 'ro-', label='Input+Hidden Layers')
        plt.plot(prune_rates, accuracies_i, 'go-', label='Input Layer')
        plt.plot(prune_rates, accuracies_h, 'bo-', label='Hidden Layer')
        plt.xlabel('Pruning Rate (%)')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy vs Pruning Rate')
        
        plt.grid(True)
        plt.legend()
        plt.savefig('accuracy_vs_prune_rate.png')
        plt.close()
    
    def t_test(self, method1_acc, method2_acc, alpha=0.01):
        """
        두 방법의 정확도에 대한 t-검정
        :param method1_acc: 방법 1의 정확도 리스트
        :param method2_acc: 방법 2의 정확도 리스트
        :param alpha: 유의수준
        :return: t 통계량, p 값, 방법 1이 방법 2보다 우수한지 여부
        """
        # t-검정 수행
        t_stat, p_value = stats.ttest_ind(method1_acc, method2_acc)
        
        # 방법 1이 방법 2보다 우수한지 검정
        # 단측 검정으로 변환
        p_value_one_tailed = p_value / 2
        
        # 방법 1의 평균이 방법 2보다 크고, p 값이 유의수준보다 작으면 방법 1이 우수
        is_method1_better = np.mean(method1_acc) > np.mean(method2_acc) and p_value_one_tailed < alpha
        
        return t_stat, p_value, is_method1_better
    
    def compare_methods(self, method_names, method_accuracies, alpha=0.01):
        """
        여러 방법의 정확도 비교
        :param method_names: 방법 이름 리스트
        :param method_accuracies: 방법별 정확도 리스트
        :param alpha: 유의수준
        """
        n_methods = len(method_names)
        
        # 결과 테이블 헤더 출력
        print("방법 비교 결과:")
        print("-" * 50)
        print("방법1 vs 방법2 | t-통계량 | p-값 | 방법1 우수?")
        print("-" * 50)
        
        # 모든 방법 쌍에 대해 t-검정
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                t_stat, p_value, is_better = self.t_test(method_accuracies[i], method_accuracies[j], alpha)
                
                print(f"{method_names[i]} vs {method_names[j]} | {t_stat:.4f} | {p_value:.4f} | {is_better}")
        
        print("-" * 50)
