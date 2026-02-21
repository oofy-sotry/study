from account import Account, SavingsAccount

def main():
    owner = input("이름을 입력하세요: ")
    balance = float(input("초기 금액을 입력하세요: "))

    account_type = input("계좌 종류 선택 (1: 일반, 2: 저축): ")

    if account_type == "2":
        acc = SavingsAccount(owner, balance)
    else:
        acc = Account(owner, balance)

    while True:
        print("\n1. 입금")
        print("2. 출금")
        print("3. 잔액 확인")
        print("4. 이자 추가 (저축계좌만)")
        print("0. 종료")

        choice = input("선택: ")

        try:
            if choice == "1":
                amount = float(input("입금 금액: "))
                acc.deposit(amount)

            elif choice == "2":
                amount = float(input("출금 금액: "))
                acc.withdraw(amount)

            elif choice == "3":
                print(acc)

            elif choice == "4":
                if isinstance(acc, SavingsAccount):
                    rate = float(input("이자율 입력 (예: 0.05): "))
                    acc.add_interest(rate)
                else:
                    print("저축계좌만 가능합니다.")

            elif choice == "0":
                print("프로그램 종료")
                break

            else:
                print("잘못된 선택입니다.")

        except ValueError as e:
            print("오류:", e)


if __name__ == "__main__":
    main()
