def divide(a, b):
    if b == 0:
        raise ValueError("0으로 나눌 수 없음")
    return a / b

if __name__ == "__main__":
    divide(10, 0)
