# 오차역전파법을 적용한 신경망 구현
# 2층 신경망 TwoLayerNet 클래스

import sys
import os
from collections import OrderedDict
sys.path.append(os.pardir)
from Common_func.common_main import *
from Gradients.Gradient_Descent import numerical_gradients
from External_Module.mnist import load_mnist
from Affine_Layer import Affine
from Soft_Max_With_Loss_Layer import SoftWithLoss

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
        weight_init_std=0.01):
        # 가중치 초기화
        self.params = {'W1': weight_init_std * \
                             np.random.randn(input_size, hidden_size), 'b1': np.zeros(hidden_size),
                       'W2': weight_init_std * \
                             np.random.randn(hidden_size, output_size), 'b2': np.zeros(output_size)}

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = \
            Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = \
            Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {'W1': numerical_gradients(loss_W, self.params['W1']),
                 'b1': numerical_gradients(loss_W, self.params['b1']),
                 'W2': numerical_gradients(loss_W, self.params['W2']),
                 'b2': numerical_gradients(loss_W, self.params['b2'])}

        return grads
    def gradient(self, x, t):
        # 순전파
        self.loss(x, t)

        # 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {'W1': self.layers['Affine1'].dW, 'b1': self.layers['Affine1'].db,
                 'W2': self.layers['Affine2'].dW, 'b2': self.layers['Affine2'].db}

        return grads


# 신경망의 계층을 순서가 있는 딕셔너리에서 보관,
# 따라서 순전파때는 추가한 순서대로 각 계층의 forward()를 호출하기만 하면 된다.
# 역전파때는 계층을 반대 순서로 호출하기만 하면 된다.
# 신경망의 구성 요소를 모듈화하여 계층으로 구현했기 때문에 구축이 쉬워진다.


# 5.7.3 오차역전파법으로 구한 기울기 검증하기

# 기울기를 구하는데는 두 가지 방법이 있다.
# 1. 수치 미분 : 느리다. 구현이 쉽다.
# 2. 해석적으로 수식을 풀기(오차 역전파법) : 빠르지만 실수가 있을 수 있다.
# 두 기울기 결과를 비교해서 오차역전파법을 제대로 구현했는지 검증한다.
# 이 작업을 기울기 확인gradient check라고 한다.

if __name__ == '__main__':
    # 데이터 읽기
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    x_batch = x_train[:3]
    t_batch = t_train[:3]

    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    grad_backprop = network.gradient(x_batch, t_batch)

# 각 가중치의 차이의 절댓값을 구한 후, 그 절댓값들의 평균을 낸다.
    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print("\n", key + ":" + str(diff))
# 수치 미분과 오차역전파법으로 구한 기울기의 차이가 매우 작다.
# 실수 없이 구현되었을 확률이 높다.
# 정밀도가 유한하기 때문에 오차가 0이 되지는 않는다.
