import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        신경망 초기화
        :param input_size: 입력층 노드 수
        :param hidden_size: 은닉층 노드 수
        :param output_size: 출력층 노드 수
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 가중치 초기화 (-1, 1) 사이의 랜덤 값
        self.weights_ih = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.weights_ho = np.random.uniform(-1, 1, (output_size, hidden_size))
        
        # 바이어스 초기화
        self.bias_h = np.random.uniform(-1, 1, (hidden_size, 1))
        self.bias_o = np.random.uniform(-1, 1, (output_size, 1))
        
    def sigmoid(self, x):
        """시그모이드 활성화 함수"""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """시그모이드 함수의 미분"""
        return x * (1 - x)
    
    def forward(self, X):
        """
        순방향 전파
        :param X: 입력 데이터 (배치 크기 x 입력 크기)
        :return: 출력 데이터
        """
        # 입력층 -> 은닉층
        self.hidden_input = np.dot(self.weights_ih, X.T) + self.bias_h
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        # 은닉층 -> 출력층
        self.output_input = np.dot(self.weights_ho, self.hidden_output) + self.bias_o
        self.output = self.sigmoid(self.output_input)
        
        return self.output.T
    
    def backward(self, X, y, learning_rate=0.1, momentum=0.9):
        """
        역전파 알고리즘
        :param X: 입력 데이터
        :param y: 타겟 데이터
        :param learning_rate: 학습률
        :param momentum: 모멘텀 계수
        """
        # 출력층 오차
        output_error = y.T - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)
        
        # 은닉층 오차
        hidden_error = np.dot(self.weights_ho.T, output_delta)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        # 가중치 및 바이어스 업데이트
        # 출력층 -> 은닉층
        self.weights_ho += learning_rate * np.dot(output_delta, self.hidden_output.T)
        self.bias_o += learning_rate * np.sum(output_delta, axis=1, keepdims=True)
        
        # 은닉층 -> 입력층
        self.weights_ih += learning_rate * np.dot(hidden_delta, X)
        self.bias_h += learning_rate * np.sum(hidden_delta, axis=1, keepdims=True)
    
    def train(self, X, y, epochs=1000, learning_rate=0.1, momentum=0.9, validation_data=None, early_stopping=True):
        """
        신경망 학습
        :param X: 입력 데이터
        :param y: 타겟 데이터
        :param epochs: 학습 횟수
        :param learning_rate: 학습률
        :param momentum: 모멘텀 계수
        :param validation_data: 검증 데이터 (X_val, y_val)
        :param early_stopping: 조기 종료 여부
        :return: 에포크별 훈련 및 검증 오차
        """
        train_errors = []
        val_errors = []
        
        # 조기 종료를 위한 변수
        best_val_error = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            # 순방향 전파
            output = self.forward(X)
            
            # RMSE 계산
            train_error = np.sqrt(np.mean(np.square(y - output)))
            train_errors.append(train_error)
            
            # 역전파
            self.backward(X, y, learning_rate, momentum)
            
            # 검증 데이터가 있으면 검증 오차 계산
            if validation_data is not None:
                X_val, y_val = validation_data
                val_output = self.forward(X_val)
                val_error = np.sqrt(np.mean(np.square(y_val - val_output)))
                val_errors.append(val_error)
                
                # 조기 종료 검사
                if early_stopping:
                    if val_error < best_val_error:
                        best_val_error = val_error
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
        
        return train_errors, val_errors
    
    def predict(self, X):
        """
        예측
        :param X: 입력 데이터
        :return: 예측 결과
        """
        return self.forward(X)
    
    def rmse(self, y_true, y_pred):
        """
        RMSE 계산
        :param y_true: 실제 값
        :param y_pred: 예측 값
        :return: RMSE
        """
        return np.sqrt(np.mean(np.square(y_true - y_pred)))
    
    def accuracy(self, y_true, y_pred):
        """
        정확도 계산 (분류 문제에 사용)
        :param y_true: 실제 값 (원-핫 인코딩)
        :param y_pred: 예측 값
        :return: 정확도
        """
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
        return np.mean(y_pred_classes == y_true_classes)
    
    def copy_weights_from(self, other_nn, input_mask, hidden_mask):
        """
        다른 신경망에서 가중치를 복사
        :param other_nn: 가중치를 복사할 다른 신경망
        :param input_mask: 입력층 마스크 (1: 유지, 0: 제거)
        :param hidden_mask: 은닉층 마스크 (1: 유지, 0: 제거)
        """
        # 입력층 -> 은닉층 가중치 복사
        for h in range(self.hidden_size):
            if hidden_mask[h] == 1:  # 은닉층 노드가 유지되는 경우만
                for i in range(self.input_size):
                    if input_mask[i] == 1:  # 입력층 노드가 유지되는 경우만
                        self.weights_ih[h, i] = other_nn.weights_ih[h, i]
                
                # 은닉층 바이어스 복사
                self.bias_h[h] = other_nn.bias_h[h]
        
        # 은닉층 -> 출력층 가중치 복사
        for o in range(self.output_size):
            for h in range(self.hidden_size):
                if hidden_mask[h] == 1:  # 은닉층 노드가 유지되는 경우만
                    self.weights_ho[o, h] = other_nn.weights_ho[o, h]
            
            # 출력층 바이어스 복사
            self.bias_o[o] = other_nn.bias_o[o]
