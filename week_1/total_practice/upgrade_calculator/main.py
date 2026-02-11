from calculator import add, subtract, multiply, divide

def get_choice():
    print("=== 계산기 프로그램 ===")
    print("1. 더하기")
    print("2. 빼기")
    print("3. 곱하기")
    print("4. 나누기")
    print("0. 종료")
    return input("원하는 연산 번호를 선택하세요:")

def get_numbers():
    try:
        num1 = float(input("첫 번째 숫자 입력: "))
        num2 = float(input("두 번째 숫자 입력: "))
        return num1, num2
    except ValueError:
        print("숫자를 입력해주세요.")
        return None

def calculate(choice, num1, num2):
    if choice == "1":
        return add(num1, num2)
    elif choice == "2":
        return subtract(num1, num2)
    elif choice == "3":
        return multiply(num1, num2)
    elif choice == "4":
        return divide(num1, num2)
    else:
        print("잘못된 선택입니다.")
        return None

def main():
    while True:
        choice = get_choice()

        if choice == "0":
            print("프로그램 종료")
            break

        numbers = get_numbers()
        if numbers is None:
            continue

        num1, num2 = numbers

        try:
            result = calculate(choice, num1, num2)
            if result is not None:
                print("결과:", result)
        except ZeroDivisionError:
            print("0으로 나눌 수 없습니다.")


if __name__ == "__main__":
    main()
