def num_input():
    return float(input("숫자 입력:"))

def divide(a, b):
    return a / b

if __name__ == "__main__":
    try:
        num1 = num_input()
        num2 = num_input()
        result = divide(num1, num2)
        print(result)
    except ValueError:
        print("숫자를 입력하세요")
    except ZeroDivisionError:
        print("0으로 나눌 수 없습니다")
