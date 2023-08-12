# 경사 하강
import numpy as np
from Common_func.common_main import function_2


def numerical_gradients(f, x): # f는 함수, x는 넘파이 배열
    # 즉 x는 넘파이 배열 x의 각 원소에 대해서 수치 미분을 구한다
    h = 1e-4
    # grad: x와 같은 크기의 0으로 채워진 배열
    grad = np.zeros_like(x) # x와 형상이 같고 그 원사가 모두 0인 배열을 만든다

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x # init_x: 초기값

    for i in range(step_num): # step_num: 경사법에 따른 반복 횟수
        grad = numerical_gradients(f, x)
        x -= lr * grad # lr: learning rate(학습률)
        # 기울기에 학습률을 곱한 값으로 갱신하는 처리를 step_num 번 반복
    return x

init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr = 0.1, step_num=100))

