# 4. 가변 위치 인자(args)
def sum_all(*args):
    print(type(args))
    return sum(args)

print(sum_all(1,2,3,4))
