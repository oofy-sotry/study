while True:
    print("=== 계산기 프로그램 ===")
    print("1. 더하기")
    print("2. 빼기")
    print("3. 곱하기")
    print("4. 나누기")

    # input()은 항상 문자열(str)반환
    # 만약 숫자로 사용하려면 형변환 필요
    choice = input("원하는 연산 번호를 선택하세요")
    print("선택한 값:", choice)

    num1 = int(input("첫 번째 숫자 입력:"))
    num2 = int(input("두 번째 숫자 입력:"))

    if choice == "1":
        print("결과:", num1 + num2)
    elif choice == "2":
        print("결과:", num1 - num2)
    elif choice == "3":
        print("결과:", num1 * num2)
    elif choice == "4":
        print("결과:", num1 / num2)
    else:
        print("잘못된 선택입니다")
