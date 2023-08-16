import numpy as np


class Sigmoid: # 순전파의 출력을 인스턴스 변수 out 에 보관하고, 역전파 계산 떄 그 값을 사용
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

