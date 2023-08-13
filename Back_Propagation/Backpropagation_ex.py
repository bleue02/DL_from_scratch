# forward: 순전판
# backward: 역전파
# 곱셈계층: MulLayer
# 덧셈계층: AddLayer

class MulLayer:
    def __init__(self): # Reset Instance variable x and y
        self.x = None # x, y는 순전파 시의 입력값을 유지하기 위하여 사용됨
        self.y = None

    def forward(self, x, y): # x와 y를 인수로 받고 두 값을 곱해서 반환
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout): # 상류에서 넘어온 미분(Dout)에 순전파 떄의 값을 서로 바꾼후 곱하고 하류로 흘린다.
        dx = dout * self.y # x와 y를 바꾼다
        dy = dout * self.x
        return dx, dy

class AddLayer:
    def __init__(self):
        pass

    @staticmethod
    def forward(x, y):
        out = x + y
        return out

    @staticmethod
    def backward(dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


if __name__ == '__main__':
    # 문제1의 예시
    apple = 100
    apple_num = 2
    tax = 1.1

    # 계층들
    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()

    # 순전파
    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax)

    print(price)  # 220.0

    # 역전파
    dprice = 1
    dapple_price, dtax = mul_tax_layer.backward(dprice) # backward의 호출 순서는 forward의 반대
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)

    print(dapple, dapple_num, dtax)  # 2.2 110.0 200

    # 문제2의 예시
    orange = 150
    orange_num = 3

    # 계층들
    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()

    # 순전파
    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)
    price = mul_tax_layer.forward(all_price, tax)

    print(price)  # 715.0

    # 역전파
    dprice = 1
    dall_price, dtax = mul_tax_layer.backward(dprice)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
    dornage, dorange_num = mul_orange_layer.backward(dorange_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)

    print(dapple_num, dapple, dornage, dorange_num, dtax)
    # 110.0 2.2 3.3 165.0 650