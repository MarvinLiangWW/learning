import time

'''自身不传入参数的装饰器（采用两层函数定义装饰器）'''
def login(func):
    def wrapper(*args,**kwargs):
        print('function name %s'%func.__name__)
        return func(*args,**kwargs)
    return wrapper

'''自身传入参数的装饰器（采用三层函数定义装饰器）'''
def login(text):
    def decorator(func):
        def wrapper(*args,**kwargs):
            print('function name %s'%func.__name__)
            return func(*args,**kwargs)
        return wrapper
    return decorator
# 等价于 -> (login(text))(f) -> return wrapper

'''类中的装饰器 @classmethod @staticmethod @property
普通方法：对象.method(self)
@classmethod： 类名.class_method(cls), 对象.class_method(cls)
@staticmethod： 类名.static_method(), 对象.static_method()

使用场景：
不需要用到与类相关的属性与方法时，就用静态方法@staticmethod
需要用到与类相关的属性或方法，又想表明这个方法是整个类通用的，而不是对象特异的，使用类方法@classmethod

'''

'''多个装饰器的调用顺序与执行结果'''
def decorator_a(func):
    print('Get in decorator_a')
    def inner_a(*args,**kwargs):
        print('Get in inner_a')
        return func(*args,**kwargs)
    return inner_a

def decorator_b(func):
    print('Get in decorator_b')
    def inner_b(*args,**kwargs):
        print('Get in inner_b')
        return func(*args,**kwargs)
    return inner_b

@decorator_b
@decorator_a
def f(x):
    print('Get in f')
    return x**2

f(1)

'''
the corresponding output result:
Get in decorator_a
Get in decorator_b
Get in inner_b
Get in inner_a
Get in f
1

explaination: https://segmentfault.com/a/1190000007837364
'''


'''
实际的应用场景中，不需要太过研究细节，只需要按照逻辑进行调用
example：
写了两个装饰方法比如
1.先验证有没有登录@login_required
2.再验证权限不够使@permission_allowed
采用下面的顺序来装饰函数：
@login_required
@permission_allowed
def f():
    # Do something
    return
'''
    




# 常用的函数计时装饰器。
def fn_timer(function):
    '''
    usage:
    @fn_timer
    def test():
        print('main')
    :param function: any function name 
    :return:
    '''
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" % (function.func_name, str(t1 - t0)))
        return result
    return function_timer
