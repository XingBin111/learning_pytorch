import torch as t
from matplotlib import pyplot as plt
import numpy as np

t.manual_seed(1000)

def get_fake_data(batch_size=8):
    x = t.rand(batch_size, 1) * 5
    y = x * 2 + 3 + t.randn(batch_size, 1)
    return x, y

x, y = get_fake_data()
plt.scatter(x.squeeze().numpy(), y.squeeze().numpy())

w = t.rand(1, 1, requires_grad=True)
b = t.rand(1, 1, requires_grad=True)
losses = np.zeros(500)

lr = 0.005

for i in range(500):
    print(i)
    x, y = get_fake_data(32)

    y_pred = x * w + b

    loss = 0.5 * (y_pred - y) ** 2
    loss = loss.sum()
    losses[i] = loss.item()

    loss.backward()

    # 参数已经更新完毕, 后面所有操作都不涉及梯度, 所有都是tensor.data
    w.data.sub_(lr*w.grad.data)
    b.data.sub_(lr*b.grad.data)

    w.grad.data.zero_()
    b.grad.data.zero_()


    if i % 50 == 0:
        x = t.arange(0, 6, dtype=t.float32).view(-1, 1)
        y = x * w.data + b.data

        plt.plot(x.numpy(), y.numpy())

        x2, y2 = get_fake_data(20)
        plt.scatter(x2.numpy(), y2.numpy())

        plt.xlim(0, 5)
        plt.ylim(0, 13)
        plt.show()
        # plt.pause(0.5)