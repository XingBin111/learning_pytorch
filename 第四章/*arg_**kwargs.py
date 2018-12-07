# *args将参数打包成tuple给函数调用
def f(*args):
    print(args, type(args))


f(1)


def f(x, y, *args):
    print(x, y, args)


f(1, 2, 3, 4, 5)

# **kwargs将关键字参数成dict给函数体调用


def f(**kwargs):
    print(kwargs, type(kwargs))


f(a=2)


def f(**kwargs):
    print(kwargs)


f(a=1, b=2, c=3)


# 注意点：参数arg、*args、**kwargs三个参数的位置必须是一定的。必须是(arg,*args,**kwargs)这个顺序，否则程序会报错。
def f(arg, *args, **kwargs):
    print(arg, args, kwargs)


f(6, 7, 8, 9, a=1, b=2, c=2)
