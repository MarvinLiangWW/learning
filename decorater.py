import time

'''自身不传入参数的装饰器（采用两层函数定义装饰器）'''
def login(func):
    def wrapper(*args,**kargs):
        print('function name %s'%func.__name__)
        return func(*args,**kargs)
    return wrapper

'''自身传入参数的装饰器（采用三层函数定义装饰器）'''
def login(text):
    def decorator(func):
        def wrapper(*args,**kargs):
            print('function name %s'%func.__name__)
            return func(*args,**kargs)
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
