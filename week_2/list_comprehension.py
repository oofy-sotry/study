# 기본형
def basic():
    result = [x for x in range(10)]
    print("기본형:", result)

# 조건 포함
def with_condition():
    result = [x for x in range(10) if x % 2 == 0]
    print("조건 포함:", result)

# 변형 포함
def with_transform():
    result = [x*x for x in range(10)]
    print("변형 포함:", result)

if __name__ == "__main__":
    basic()
    with_condition()
    with_transform()
