# 5. 가변 키워드 인자(kwargs)
def print_user(**kwargs):
    print(type(kwargs))
    return kwargs

print(print_user(name="Kim", age=20))
