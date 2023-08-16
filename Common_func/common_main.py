import numpy as np


def function_1(x):
    return 0.01 * x**2 + 0.1 * x

def function_2(x):
    return x[0] **2 + x[1] ** 2

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def partial_derivative_1(x0):
    return x0*x0 + 4.0 ** 2.0
# print(numerical_diff(partial_derivative_1, 3.0))

def partial_derivative_2(x1):
    return 3.0 ** 2.0 + x1 * x1
# print(numerical_diff(partial_derivative_2, 4.0))

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
#
#     batch_size = y.shape[0]
#     return -np.sum(t * np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]

    # 원-핫 인코딩 경우 정수 인덱스를 가져옴
    if t.size == y.size:
        t = t.argmax(axis=1) # np.clip 함수는 넘파이 배열의 원소를 지정된 범위 내로 제한(클립)하는 함수
    y_clipped = np.clip(y, 1e-10, 1 - 1e-10) # Y의 원소중 1e-10보다 작은 경우 1e-10, = "0.0000000001"
                                             # Y의 원소중 1e-10보다 큰 경우 1 - 1e-10 = "0.9999999999"
    return -np.sum(np.log(y_clipped[np.arange(batch_size), t])) / batch_size

    # return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size  # 원래 코드



class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

