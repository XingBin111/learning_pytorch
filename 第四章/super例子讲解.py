# 在类的继承中，如果你想要重写父类的方法而不是覆盖的父类方法，这个时候我们可以使用super()方法来实现


# eaxmple 1
class C(object):
    def minus(self, x):
        return x / 2


class D(C):
    def minus(self, x):
        super(D, self).minus(x)
        print('hello')

d = D()
d.minus(2)      # hello, 相当于在父类上面增加了print功能




# example 2
class A(object):
    def __init__(self):
        self.n = 10

    def minus(self, m):
        self.n -= m


class B(A):
    def __init__(self):
        self.n = 7

    def minus(self, m):
        super(B, self).minus(m)
        self.n -= 2

b=B()
b.minus(2)
print(b.n)      # b.n = 3


class C(A):
 def __init__(self):
  self.n = 12

 def minus(self, m):
  super(C,self).minus(m)
  self.n -= 5


class D(B, C):
 def __init__(self):
  self.n = 15

 def minus(self, m):
  super(D,self).minus(m)
  self.n -= 2

d=D()
d.minus(2)
print(d.n)      # b.n = 4
D.__mro__       # 类继承的顺序