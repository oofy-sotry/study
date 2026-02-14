def num_input():
    try:
        return float(input("숫자 입력:"))
    except ValueError:
        print("숫자를 입력하세요")
        return None

def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return "0으로 나눌 수 없습니다"

if __name__ == "__main__":
    num1 = num_input()
    num2 = num_input()

    if num1 is not None and num2 is not None:
        print(divide(num1, num2))
