def add (a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def devide(a, b):
    return a / b

def main():
    while True:
        print("=== 계산기 프로그램 ===")
        print("1. 더하기")
        print("2. 빼기")
        print("3. 곱하기")
        print("4. 나누기")

        if choice == "0":
            print("프로그램 종료")
            break

        num1 = int(input("첫 번째 숫자 입력: "))
        num2 = int(input("두 번째 숫자 입력: "))

        if choice == "1":
            print("결과:", add(num1, num2))
        elif choice == "2":
            print("결과:", subtract(num1, num2))
        elif choice == "3":
            print("결과:", multiply(num1, num2))
        elif choice == "4":
            print("결과:", divide(num1, num2))
        else:
            print("잘못된 선택입니다.")

if __name__ == "__main__":
    main()
