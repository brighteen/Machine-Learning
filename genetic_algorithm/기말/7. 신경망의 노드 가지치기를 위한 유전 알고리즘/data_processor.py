import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class DataProcessor:
    def __init__(self):
        """데이터 처리기 초기화"""
        self.scaler = MinMaxScaler()
    
    def load_data(self, file_path, has_header=True, delimiter=','):
        """
        데이터 로드
        :param file_path: 데이터 파일 경로
        :param has_header: 헤더 유무
        :param delimiter: 구분자
        :return: 로드된 데이터
        """
        try:
            header = 0 if has_header else None
            data = pd.read_csv(file_path, header=header, delimiter=delimiter)
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def normalize_data(self, X):
        """
        데이터 정규화 (0~1 범위)
        :param X: 입력 데이터
        :return: 정규화된 데이터
        """
        return self.scaler.fit_transform(X)
    
    def one_hot_encode(self, y, num_classes=None):
        """
        원-핫 인코딩
        :param y: 레이블 데이터
        :param num_classes: 클래스 수
        :return: 원-핫 인코딩된 데이터
        """
        if num_classes is None:
            num_classes = len(np.unique(y))
        
        # 원-핫 인코딩
        one_hot = np.zeros((y.shape[0], num_classes))
        for i, val in enumerate(y):
            one_hot[i, int(val)] = 1
            
        return one_hot
    
    def prepare_data(self, data, target_column, one_hot=True, num_classes=None):
        """
        데이터 준비
        :param data: 입력 데이터프레임
        :param target_column: 타겟 열 이름 또는 인덱스
        :param one_hot: 원-핫 인코딩 여부
        :param num_classes: 클래스 수
        :return: X, y
        """
        # 타겟 데이터 분리
        if isinstance(target_column, str):
            y = data[target_column].values
            X = data.drop(target_column, axis=1).values
        else:
            y = data.iloc[:, target_column].values
            X = data.drop(data.columns[target_column], axis=1).values
        
        # 데이터 정규화
        X = self.normalize_data(X)
        
        # 원-핫 인코딩
        if one_hot:
            y = self.one_hot_encode(y, num_classes)
            
        return X, y
    
    def k_fold_split(self, X, y, n_splits=10, random_state=42):
        """
        K-겹 교차 검증 분할
        :param X: 입력 데이터
        :param y: 레이블 데이터
        :param n_splits: 분할 수
        :param random_state: 랜덤 시드
        :return: 분할된 데이터 세트
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_data = []
        
        for train_index, test_index in kf.split(X):
            # 각 폴드에서 훈련, 검증, 테스트 세트 분할
            X_train_val, X_test = X[train_index], X[test_index]
            y_train_val, y_test = y[train_index], y[test_index]
            
            # 훈련 데이터의 8/9는 훈련, 1/9는 검증 세트로 사용
            n = len(X_train_val)
            n_train = int(n * 8/9)
            
            X_train, X_val = X_train_val[:n_train], X_train_val[n_train:]
            y_train, y_val = y_train_val[:n_train], y_train_val[n_train:]
            
            fold_data.append((X_train, y_train, X_val, y_val, X_test, y_test))
        
        return fold_data
    
    def prepare_uci_data(self, dataset_name):
        """
        UCI 데이터 준비
        :param dataset_name: 데이터셋 이름
        :return: 데이터셋 정보 (특징 수, 클래스 수, 입력 데이터, 타겟 데이터)
        """
        # 실제 UCI 데이터를 로드하는 코드는 데이터셋마다 다를 수 있음
        # 여기서는 예시만 제공
        
        if dataset_name == "iris":
            from sklearn.datasets import load_iris
            data = load_iris()
            X, y = data.data, data.target
            num_features = X.shape[1]
            num_classes = len(np.unique(y))
            
            # 데이터 정규화
            X = self.normalize_data(X)
            
            # 원-핫 인코딩
            y_onehot = self.one_hot_encode(y, num_classes)
            
            return num_features, num_classes, X, y_onehot
        
        elif dataset_name == "wine":
            from sklearn.datasets import load_wine
            data = load_wine()
            X, y = data.data, data.target
            num_features = X.shape[1]
            num_classes = len(np.unique(y))
            
            # 데이터 정규화
            X = self.normalize_data(X)
            
            # 원-핫 인코딩
            y_onehot = self.one_hot_encode(y, num_classes)
            
            return num_features, num_classes, X, y_onehot
        
        else:
            print(f"Dataset {dataset_name} not supported yet.")
            return None, None, None, None
