import time
from functools import wraps


"""
使用装饰器模式进行时间统计
"""
def time_counter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        exec_time = end_time - start_time
        print(f"{func.__name__} 函数执行时间：{exec_time:.2f}秒")
        return result
    return wrapper


# 使用示例
@time_counter
def create_collection(self):
    # 方法实现
    pass

@time_counter
def add_docs(self, datas):
    # 方法实现
    pass