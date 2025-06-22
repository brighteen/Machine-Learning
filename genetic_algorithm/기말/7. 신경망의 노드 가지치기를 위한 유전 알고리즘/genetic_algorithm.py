import numpy as np
import random
from neural_network import NeuralNetwork

class Chromosome:
    def __init__(self, input_size, hidden_size, prune_rate=0.1):
        """
        염색체 초기화
        :param input_size: 입력층 노드 수
        :param hidden_size: 은닉층 노드 수
        :param prune_rate: 가지치기 비율 (0~1 사이)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.prune_rate = prune_rate
        self.gene_length = input_size + hidden_size
        
        # 유전자 초기화 (1: 노드 유지, 0: 노드 제거)
        self.genes = np.ones(self.gene_length, dtype=int)
        
        # 입력층과 은닉층의 가지치기 비율 계산
        input_prune_count = int(input_size * prune_rate)
        hidden_prune_count = int(hidden_size * prune_rate)
        
        # 입력층 가지치기
        prune_indices = np.random.choice(input_size, input_prune_count, replace=False)
        for idx in prune_indices:
            self.genes[idx] = 0
            
        # 은닉층 가지치기
        prune_indices = np.random.choice(hidden_size, hidden_prune_count, replace=False) + input_size
        for idx in prune_indices:
            self.genes[idx] = 0
            
        self.fitness = 0.0  # 적합도
        self.neural_network = None  # 해당 염색체에 대응하는 신경망
        
    def get_input_mask(self):
        """입력층 마스크 반환"""
        return self.genes[:self.input_size]
    
    def get_hidden_mask(self):
        """은닉층 마스크 반환"""
        return self.genes[self.input_size:]
    
    def get_active_input_count(self):
        """활성화된 입력 노드 수 반환"""
        return np.sum(self.get_input_mask())
    
    def get_active_hidden_count(self):
        """활성화된 은닉 노드 수 반환"""
        return np.sum(self.get_hidden_mask())
    
    def adjust_prune_rate(self):
        """가지치기 비율 조정"""
        # 입력층 가지치기 비율 조정
        input_mask = self.get_input_mask()
        target_input_prune = int(self.input_size * self.prune_rate)
        current_input_prune = self.input_size - np.sum(input_mask)
        
        if current_input_prune > target_input_prune:
            # 0 -> 1 변환 (추가)
            zeros = np.where(input_mask == 0)[0]
            to_change = np.random.choice(zeros, current_input_prune - target_input_prune, replace=False)
            for idx in to_change:
                self.genes[idx] = 1
        elif current_input_prune < target_input_prune:
            # 1 -> 0 변환 (제거)
            ones = np.where(input_mask == 1)[0]
            to_change = np.random.choice(ones, target_input_prune - current_input_prune, replace=False)
            for idx in to_change:
                self.genes[idx] = 0
        
        # 은닉층 가지치기 비율 조정
        hidden_mask = self.get_hidden_mask()
        target_hidden_prune = int(self.hidden_size * self.prune_rate)
        current_hidden_prune = self.hidden_size - np.sum(hidden_mask)
        
        if current_hidden_prune > target_hidden_prune:
            # 0 -> 1 변환 (추가)
            zeros = np.where(hidden_mask == 0)[0]
            to_change = np.random.choice(zeros, current_hidden_prune - target_hidden_prune, replace=False)
            for idx in to_change:
                self.genes[self.input_size + idx] = 1
        elif current_hidden_prune < target_hidden_prune:
            # 1 -> 0 변환 (제거)
            ones = np.where(hidden_mask == 1)[0]
            to_change = np.random.choice(ones, target_hidden_prune - current_hidden_prune, replace=False)
            for idx in to_change:
                self.genes[self.input_size + idx] = 0

class GeneticAlgorithm:
    def __init__(self, population_size, input_size, hidden_size, output_size, prune_rate=0.1, crossover_rate=1.0, mutation_rate=0.01, selection_pressure=0.25):
        """
        유전 알고리즘 초기화
        :param population_size: 인구 크기
        :param input_size: 입력층 노드 수
        :param hidden_size: 은닉층 노드 수
        :param output_size: 출력층 노드 수
        :param prune_rate: 가지치기 비율
        :param crossover_rate: 교차 확률
        :param mutation_rate: 돌연변이 확률
        :param selection_pressure: 선택 압력
        """
        self.population_size = population_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.prune_rate = prune_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_pressure = selection_pressure
        
        # 인구 초기화
        self.population = [Chromosome(input_size, hidden_size, prune_rate) for _ in range(population_size)]
        
    def create_neural_network(self, chromosome):
        """
        염색체에 대응하는 신경망 생성
        :param chromosome: 염색체
        :return: 신경망
        """
        return NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
    
    def calculate_fitness(self, chromosome, X_val, y_val):
        """
        적합도 계산
        :param chromosome: 염색체
        :param X_val: 검증 데이터
        :param y_val: 검증 데이터 레이블
        :return: 적합도
        """
        if chromosome.neural_network is None:
            return 0.0
        
        # 신경망 예측
        y_pred = chromosome.neural_network.predict(X_val)
        
        # RMSE 계산
        rmse = chromosome.neural_network.rmse(y_val, y_pred)
        
        # 적합도 계산 (RMSE가 낮을수록 적합도가 높음)
        fitness = 1.0 / (1.0 + rmse)
        
        return fitness
    
    def select_parent(self):
        """
        룰렛 휠 선택법으로 부모 선택
        :return: 선택된 부모 염색체
        """
        # 순위 기반 선택 확률 계산
        sorted_population = sorted(self.population, key=lambda c: c.fitness, reverse=True)
        
        # 순위 기반 적합도 할당
        q = self.selection_pressure
        rank_probabilities = np.array([q * (1 - q) ** i for i in range(self.population_size)])
        rank_probabilities /= np.sum(rank_probabilities)  # 정규화
        
        # 룰렛 휠 선택
        point = random.random()
        cumulative_prob = 0
        
        for i, chromosome in enumerate(sorted_population):
            cumulative_prob += rank_probabilities[i]
            if point < cumulative_prob:
                return chromosome
        
        # 예외 처리
        return sorted_population[0]
    
    def crossover(self, parent1, parent2):
        """
        교차 연산
        :param parent1: 부모 염색체 1
        :param parent2: 부모 염색체 2
        :return: 자식 염색체
        """
        if random.random() > self.crossover_rate:
            # 교차가 일어나지 않으면 부모 중 하나를 복제
            return Chromosome(self.input_size, self.hidden_size, self.prune_rate) if random.random() < 0.5 else Chromosome(self.input_size, self.hidden_size, self.prune_rate)
        
        # 자식 염색체 생성
        child = Chromosome(self.input_size, self.hidden_size, self.prune_rate)
        
        # 3점 교차
        points = sorted(random.sample(range(1, parent1.gene_length), 3))
        
        for i in range(parent1.gene_length):
            if i < points[0] or (points[1] <= i < points[2]):
                child.genes[i] = parent1.genes[i]
            else:
                child.genes[i] = parent2.genes[i]
        
        return child
    
    def mutate(self, chromosome):
        """
        돌연변이 연산
        :param chromosome: 염색체
        """
        for i in range(chromosome.gene_length):
            if random.random() < self.mutation_rate:
                # 유전자 반전 (0 -> 1, 1 -> 0)
                chromosome.genes[i] = 1 - chromosome.genes[i]
        
        # 가지치기 비율 유지
        chromosome.adjust_prune_rate()
    
    def inherit_weights(self, child, parent1, parent2):
        """
        부모의 가중치를 자식에게 상속
        :param child: 자식 염색체
        :param parent1: 부모 염색체 1
        :param parent2: 부모 염색체 2
        """
        # 자식 신경망 생성
        child.neural_network = self.create_neural_network(child)
        
        # 부모 신경망이 없으면 새로 생성
        if parent1.neural_network is None or parent2.neural_network is None:
            return
        
        # 입력층과 은닉층 마스크
        input_mask = child.get_input_mask()
        hidden_mask = child.get_hidden_mask()
        
        # 가중치 상속 - 입력층 -> 은닉층
        for h in range(self.hidden_size):
            if hidden_mask[h] == 1:  # 은닉층 노드가 유지되는 경우만
                for i in range(self.input_size):
                    if input_mask[i] == 1:  # 입력층 노드가 유지되는 경우만
                        # 부모1과 부모2 모두에 해당 가중치가 있는 경우
                        if parent1.get_hidden_mask()[h] == 1 and parent1.get_input_mask()[i] == 1 and \
                           parent2.get_hidden_mask()[h] == 1 and parent2.get_input_mask()[i] == 1:
                            # 두 부모의 가중치 평균
                            child.neural_network.weights_ih[h, i] = (parent1.neural_network.weights_ih[h, i] + 
                                                                      parent2.neural_network.weights_ih[h, i]) / 2
                        # 부모1에만 해당 가중치가 있는 경우
                        elif parent1.get_hidden_mask()[h] == 1 and parent1.get_input_mask()[i] == 1:
                            child.neural_network.weights_ih[h, i] = parent1.neural_network.weights_ih[h, i]
                        # 부모2에만 해당 가중치가 있는 경우
                        elif parent2.get_hidden_mask()[h] == 1 and parent2.get_input_mask()[i] == 1:
                            child.neural_network.weights_ih[h, i] = parent2.neural_network.weights_ih[h, i]
                        # 두 부모 모두에 해당 가중치가 없는 경우
                        else:
                            child.neural_network.weights_ih[h, i] = np.random.uniform(-1, 1)
        
        # 가중치 상속 - 은닉층 -> 출력층
        for o in range(self.output_size):
            for h in range(self.hidden_size):
                if hidden_mask[h] == 1:  # 은닉층 노드가 유지되는 경우만
                    # 부모1과 부모2 모두에 해당 가중치가 있는 경우
                    if parent1.get_hidden_mask()[h] == 1 and parent2.get_hidden_mask()[h] == 1:
                        # 두 부모의 가중치 평균
                        child.neural_network.weights_ho[o, h] = (parent1.neural_network.weights_ho[o, h] + 
                                                                  parent2.neural_network.weights_ho[o, h]) / 2
                    # 부모1에만 해당 가중치가 있는 경우
                    elif parent1.get_hidden_mask()[h] == 1:
                        child.neural_network.weights_ho[o, h] = parent1.neural_network.weights_ho[o, h]
                    # 부모2에만 해당 가중치가 있는 경우
                    elif parent2.get_hidden_mask()[h] == 1:
                        child.neural_network.weights_ho[o, h] = parent2.neural_network.weights_ho[o, h]
                    # 두 부모 모두에 해당 가중치가 없는 경우
                    else:
                        child.neural_network.weights_ho[o, h] = np.random.uniform(-1, 1)
        
        # 바이어스 상속 - 은닉층
        for h in range(self.hidden_size):
            if hidden_mask[h] == 1:  # 은닉층 노드가 유지되는 경우만
                # 부모1과 부모2 모두에 해당 바이어스가 있는 경우
                if parent1.get_hidden_mask()[h] == 1 and parent2.get_hidden_mask()[h] == 1:
                    # 두 부모의 바이어스 평균
                    child.neural_network.bias_h[h] = (parent1.neural_network.bias_h[h] + 
                                                      parent2.neural_network.bias_h[h]) / 2
                # 부모1에만 해당 바이어스가 있는 경우
                elif parent1.get_hidden_mask()[h] == 1:
                    child.neural_network.bias_h[h] = parent1.neural_network.bias_h[h]
                # 부모2에만 해당 바이어스가 있는 경우
                elif parent2.get_hidden_mask()[h] == 1:
                    child.neural_network.bias_h[h] = parent2.neural_network.bias_h[h]
                # 두 부모 모두에 해당 바이어스가 없는 경우
                else:
                    child.neural_network.bias_h[h] = np.random.uniform(-1, 1)
        
        # 출력층 바이어스는 모든 자식이 상속받음 (부모1과 부모2의 평균)
        child.neural_network.bias_o = (parent1.neural_network.bias_o + parent2.neural_network.bias_o) / 2
    
    def evolve(self, X_train, y_train, X_val, y_val, generations=150, epochs=100):
        """
        진화 과정 수행
        :param X_train: 학습 데이터
        :param y_train: 학습 데이터 레이블
        :param X_val: 검증 데이터
        :param y_val: 검증 데이터 레이블
        :param generations: 세대 수
        :param epochs: 신경망 학습 에포크 수
        :return: 최적의 염색체
        """
        # 초기 인구 평가
        for chromosome in self.population:
            # 신경망 생성 및 훈련
            chromosome.neural_network = self.create_neural_network(chromosome)
            chromosome.neural_network.train(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))
            
            # 적합도 계산
            chromosome.fitness = self.calculate_fitness(chromosome, X_val, y_val)
        
        # 세대 진화
        for generation in range(generations):
            print(f"Generation {generation+1}/{generations}")
            
            # 자식 생성
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            
            # 교차
            child = self.crossover(parent1, parent2)
            
            # 돌연변이
            self.mutate(child)
            
            # 가중치 상속
            self.inherit_weights(child, parent1, parent2)
            
            # 자식 신경망 훈련
            child.neural_network.train(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))
            
            # 자식 적합도 계산
            child.fitness = self.calculate_fitness(child, X_val, y_val)
            
            # 인구 대체
            if child.fitness > parent1.fitness or child.fitness > parent2.fitness:
                # 자식이 부모보다 우수하면 부모 중 열등한 것을 대체
                if parent1.fitness < parent2.fitness:
                    self.population[self.population.index(parent1)] = child
                else:
                    self.population[self.population.index(parent2)] = child
            else:
                # 자식이 부모보다 열등하면 인구 중 가장 열등한 것을 대체
                worst_idx = min(range(self.population_size), key=lambda i: self.population[i].fitness)
                if child.fitness > self.population[worst_idx].fitness:
                    self.population[worst_idx] = child
            
            # 현재 세대의 최고 적합도 출력
            best_fitness = max(chromosome.fitness for chromosome in self.population)
            print(f"Best fitness: {best_fitness:.6f}")
        
        # 최적의 염색체 반환
        return max(self.population, key=lambda c: c.fitness)
