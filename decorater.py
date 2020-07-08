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
