# 경사 하강
import numpy as np
from Common_func.common_main import function_2


def numerical_gradients(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    # 1차원 배열을 위한 경우
    if x.ndim == 1:
        for idx in range(x.size):
            tmp_val = x[idx]
            x[idx] = tmp_val + h
            fxh1 = f(x)
            x[idx] = tmp_val - h2
            fxh2 = f(x)
            grad[idx] = (fxh1 - fxh2) / (2 * h)
            x[idx] = tmp_val
        return grad

    # 2차원 배열을 위한 경우
    elif x.ndim == 2:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                tmp_val = x[i, j]
                x[i, j] = tmp_val + h
                fxh1 = f(x)
                x[i, j] = tmp_val - h
                fxh2 = f(x)
                grad[i, j] = (fxh1 - fxh2) / (2 * h)
                x[i, j] = tmp_val
        return grad

    # 그 외의 경우
    else:
        raise ValueError("x should be 1 or 2 dimensional array.")


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x # init_x: 초기값

    for i in range(step_num): # step_num: 경사법에 따른 반복 횟수
        grad = numerical_gradients(f, x)
        x -= lr * grad # lr: learning rate(학습률)
        # 기울기에 학습률을 곱한 값으로 갱신하는 처리를 step_num 번 반복
    return x

init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr = 0.1, step_num=100))

