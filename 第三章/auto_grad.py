import torch as t


def f(x):
    y = x ** 2 * t.exp(x)
    return y


def grad_f(x):
    dx = 2 * x * t.exp(x) + x ** 2 * t.exp(x)
    return dx


x = t.randn(3, 4, requires_grad=True)
y = f(x)

y.backward(t.ones(y.size()))        # gradient形状与y一致

print(x.grad)
print(t.norm(t.sub(x.grad, grad_f(x))))


# 如果想要修改tensor的数值, 但又不希望被autograd记录, 可以对tensor.data, 或tensor.detach()进行操作
print(x.data.requires_grad)         # False
print(x.detach().requires_grad)     # False

# 统计一些不希望被记录的指标

tensor = x.detach()
mean = tensor.mean()
std = tensor.std()

x = t.ones(3, requires_grad=True)
w = t.rand(3, requires_grad=True)
y = x * w
z = y.sum()
print(x.requires_grad, w.requires_grad, y.requires_grad)

z.backward()

# 非叶子节点grad计算完之后自动清空, y.grad=None
print(x.grad, w.grad, y.grad)       # tensor([0.1867, 0.2548, 0.1631]) tensor([1., 1., 1.]) None

# 获取z对y的梯度
t.autograd.grad(z, y)       # (tensor([1., 1., 1.]),)





