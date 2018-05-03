#隐含层有两个节点
import numpy as np

class FullyConnect:
    def __init__(self, l_x, l_y,layer):  # 两个参数
        self.weights = np.random.randn(l_y, l_x)  # 使用随机数初始化参数
        self.bias = np.random.randn(1)  # 使用随机数初始化参数
        self.layer = layer

    def forward(self, x):
        self.x = x  # 把中间结果保存下来，以备反向传播时使用
        self.y = np.dot(self.weights, x) + self.bias  # 计算w11*a1+w12*a2+bias1
        return self.y  # 将这一层计算的结果向前传递

    def backward(self, d):
        self.dw = (d * self.x).reshape(1,2)  # 根据链式法则，将反向传递回来的导数值乘以x，得到对参数的梯度
        self.db = d
        self.dx = d * self.weights
        print(self.layer,'层dw: ', self.dw)
        self.weights = self.weights - 0.1*self.dw #假定学习率为0.1
        print('更新weights: ', self.weights)
        if(self.layer == 'layer1'):
            return self.dw, self.db  # 返回求得的参数梯度
        else:
            return self.dx

class Sigmoid:
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
        self.dx = d *sig * (1 - sig)
        return self.dx  # 反向传递梯度

def main():
        fc11 = FullyConnect(2, 1,'layer1')
        fc12 = FullyConnect(2, 1,'layer1')  #隐含层有两个节点
        fc = FullyConnect(2,1,'output')
        sigmoid11= Sigmoid()
        sigmoid12 = Sigmoid()
        sigmoid = Sigmoid()
        x = np.array([[1], [2]])
        print('weights:', fc11.weights, fc12.weights,fc.weights,' bias:', fc11.bias, fc12.bias , fc.bias,' input: ', x)

        # 执行前向计算
        y11 = fc11.forward(x)
        y11_sig = sigmoid11.forward(y11)
        y12 = fc12.forward(x)
        y12_sig = sigmoid12.forward(y12)
        y1= np.vstack((y11_sig,y12_sig))
        y3 = fc.forward(y1)
        y4 = sigmoid.forward(y3)

        print('forward result: ', y4)

        # 执行反向传播
        d1 = sigmoid.backward(1) #先假定均方误差为1
        dx = fc.backward(d1)
        dx11 = sigmoid11.backward(dx[0,0])
        dx12 = sigmoid12.backward(dx[0,1])
        dw11 = fc11.backward(dx11)
        dw12 = fc12.backward(dx12)


main()