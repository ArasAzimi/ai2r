from functools import wraps


def time_it(func):
    import time

    @wraps(func)
    def time_it_decorator(*args, **kwargs):
        print(">ia> {} was called.".format(func.__name__))
        start_time = time.time()
        ret = func(*args, **kwargs)
        print(">ia> {} took {} seconds".format(func.__name__, time.time() - start_time))
        return ret

    return time_it_decorator
