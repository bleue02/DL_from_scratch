import sys, os
import numpy as np
from Common_func.common_main import softmax, sigmoid, cross_entropy_error
from Gradients.gradients import multidimensional_numerical_gradients
# from Gradients.gradients import gradient_descent
sys.path.append(os.pardir)

class TwoLayerNet:
    # 인수는 순서대로 입력층의 뉴런수, 은닉충의 뉴런 수, 출력층의 뉴런 수
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # Reset Weight
        # params: 신경망의 매개변수를 보관하는 딕셔너리 변수(Instance variable)
        # grads: Instance variable을 갖는다.
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        # params['W1']은 1번쨰 층의 가중치, params['b1']은 1층의 편향
        # params에 저장될 떄는 Numpy array로 저장됨
        # 접근은 Key!
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        # params['W2']은 2번쨰 층의 가중치, params['b2']은 2층의 편향

    # predict :각 레이블의 확률을 넘파이 배열로 반환
    # EX) [0.1, 0.3, 0.2, ... 0.3]식으로 해석
    def predict(self, x): # 예측(추론) 수행, x: 이미지 데이터(Image Data)
        W1, W2 = self.params['W1'], self.params['W2']

        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t): # 손실 함수의 값을 구한다
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):  # 손실 함수의 값을 구한다
        y = self.predict(x)
        y = np.argmax(y, axis=1) # 배열에서 값이 가장큰(확률이 높은) 원ㅅ소의 인덱스를 구한다 -> " 예측 결과 "
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0]) # 신경망에서 어ㅖ측한 답변과 정답 레이블을 비교하여 맞힌 숫자를 넣는다
        return accuracy  # predict()의 결과와 정답 레이블을 바탕으로 교차 엔트로피 오차를 구현하도록 구현됨

    # x: 입력 데이터,  t: 정답 레이블
    # numerical_gradient: 각 매개변수의 기울기를 계산
    # numerical_gradient 메소드는 오차역전ㅁ파법을 사용하여 기울기를 효율적으로 계산한다
    # 즉 수치 미분 방식으로 각 매개변수의 손실 함수에 대한 기울기를 계산한다.
    def numerical_gradient(self, x, t): # 가중치 매개변수의 기울기를 구한다.
        # gradoemt(self, x, t) 가중치 매개변수의 기울기를 구한다 <-- \
        # numerical_gradient()의 성능 개선판.
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = multidimensional_numerical_gradients(loss_W, self.params['W1'])
        grads['b1'] = multidimensional_numerical_gradients(loss_W, self.params['b1'])
        grads['W2'] = multidimensional_numerical_gradients(loss_W, self.params['W2'])
        grads['b2'] = multidimensional_numerical_gradients(loss_W, self.params['b2'])

        return grads

if __name__ == '__main__':
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    print(net.params['W1'].shape)  # (784, 100)
    print(net.params['b1'].shape)  # (100,)
    print(net.params['W2'].shape)  # (100, 10)
    print(net.params['b2'].shape)  # (10,)
