from account import Account, SavingsAccount

accounts = []
next_account_number = 1000

def find_account(account_number):
    for acc in accounts:
        if acc.account_number == account_number:
            return acc
    return None


def main():
    global next_account_number

    while True:
        print("1. 계좌 생성")
        print("2. 계좌 목록")
        print("3. 입금")
        print("4. 출금")
        print("5. 이자 추가")
        print("0. 종료")

        choice = input("선택 :")

        try:
            if choice == "1":
                name = input("이름:")
                balance = int(input("초기 금액:"))
                account_type = input("1.일반 2.저축:")

                if account_type == "1":
                    acc = Account(name, next_account_number, balance)
                elif account_type == "2":
                    acc = SavingsAccount(name, next_account_number, balance)
                else:
                    print("잘못된 선택입니다.")
                    continue

                accounts.append(acc)
                next_account_number += 1

                print("계좌 생성 완료")
                print(acc)

            elif choice == "2":
                if not accounts:
                    print("등록된 계좌가 없습니다")
                else:
                    for acc in accounts:
                        print(acc)

            elif choice == "3":
                account_number = int(input("계좌 번호: "))
                acc = find_account(account_number)

                if acc:
                    amount = float(input("입금 금액: "))
                    acc.deposit(amount)
                    print("입금 완료")
                else:
                    print("해당 계좌가 없습니다")

            elif choice == "4":
                account_number = int(input("계좌 번호: "))
                acc = find_account(account_number)

                if acc:
                    amount = float(input("출금 금액: "))
                    acc.withdrwa(amount)
                    print("출금 완료")
                    print(acc)
                else:
                    print("해당 계좌가 없습니다")
                                
            elif choice == "5":
                account_number = int(input("계좌 번호 : "))
                acc = find_account(account_number)

                if not acc:
                    print("해당 계좌가 없습니다")
                elif isinstance(acc, SavingsAccount):
                    rate = float(input("이자율 입력 (예: 0.05): "))
                    acc.add_interest(rate)
                    print("이자 추가 완료")
                    print(acc)
                else:
                    print("저축계좌만 해당합니다")

            elif choice == "0":
                print("프로그램 종료")
                break

            else:
                print("올바른 번호를 입력해주세요")


        except ValueError as e:
            print("오류:", e)


if __name__ == "__main__":
    main()
