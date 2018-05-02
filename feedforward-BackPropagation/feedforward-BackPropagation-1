import numpy as np

class FullyConnectLay1:
    def __init__(self, l_x, l_y):  # 两个参数
        self.weights = np.random.randn(l_y, l_x)  # 使用随机数初始化参数
        self.bias = np.random.randn(1)  # 使用随机数初始化参数

    def forward(self, x):
        self.x = x  # 把中间结果保存下来，以备反向传播时使用
        self.y = np.dot(self.weights, x) + self.bias  # 计算w11*a1+w12*a2+bias1
        return self.y  # 将这一层计算的结果向前传递

    def backward(self, d):
        self.dw = d * self.x  # 根据链式法则，将反向传递回来的导数值乘以x，得到对参数的梯度
        self.db = d
        self.dx = d * self.weights
        print('Hidden层dw: ', self.dw)
        self.weights = self.weights - 0.01*self.dw #假定学习率为0.01
        print('更新weights: ', self.weights)
        return self.dw, self.db  # 返回求得的参数梯度

class FullyConnectLay2:
    def __init__(self, l_x, l_y):
        self.weights = np.random.randn(l_y, l_x)  # 使用随机数初始化参数
        self.bias = np.random.randn(1)  # 使用随机数初始化参数

    def forward(self, x):
        self.x = x  # 把中间结果保存下来，以备反向传播时使用
        self.y = np.dot(self.weights, x) + self.bias  # 计算w11*a1+w12*a2+bias1
        return self.y  # 将这一层计算的结果向前传递

    def backward(self, d):
        self.dw = d * self.x  # 根据链式法则，将反向传递回来的导数值乘以x，得到对参数的梯度
        self.db = d
        self.dx = d * self.weights
        print('out层dw: ', self.dw)
        self.weights = self.weights - 0.01*self.dw #假定学习率为0.01
        print('更新weights: ', self.weights)
        return self.dx  # 返回self.dx

class SigmoidHidden:
    def __init__(self):  # 无参数，不需初始化
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.x = x
        self.y = self.sigmoid(x)
        return self.y

    def backward(self,d):  # 这里sigmoid是最后一层，所以从这里开始反向计算梯度
        sig = self.sigmoid(self.x)
        self.dx = d * sig * (1 - sig)
        return self.dx  # 反向传递梯度

class Sigmoid:
    def __init__(self):  # 无参数，不需初始化
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.x = x
        self.y = self.sigmoid(x)
        return self.y

    def backward(self):  # 这里sigmoid是最后一层，所以从这里开始反向计算梯度
        sig = self.sigmoid(self.x)
        self.dx = sig * (1 - sig)
        return self.dx  # 反向传递梯度

def main():
        fc1 = FullyConnectLay1(2, 1)
        fc = FullyConnectLay2(1,1)
        sigmoid1 = SigmoidHidden()
        sigmoid = Sigmoid()
        x = np.array([[1], [2]])
        print('weights:', fc1.weights,  fc.weights,' bias:', fc1.bias,  fc.bias,' input: ', x)

        # 执行前向计算
        y1 = fc1.forward(x)
        y2 = sigmoid1.forward(y1)
        y3 = fc.forward(y2)
        y4 = sigmoid.forward(y3)

        print('forward result: ', y4)

        # 执行反向传播
        d1 = sigmoid.backward()
        dx = fc.backward(d1)
        dx = sigmoid1.backward(dx)
        dw = fc1.backward(dx)


main()