from Common_func.common_main import cross_entropy_error
import numpy as np
class SoftWithLoss:
    def __init__(self):
        self.loss = None # 손실
        self.y = None # sotfmax의 출력
        self.t = None # 정답 레이블(원-핫 벡터)

    def forward(self, x, t):
        self.t = t
        self.y = np.sort(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx