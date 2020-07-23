import time

'''自身不传入参数的装饰器（采用两层函数定义装饰器）'''
def login(func):
    def wrapper(*args,**kargs):
        print('function name %s'%func.__name__)
        return func(*args,**kargs)
    return wrapper



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
