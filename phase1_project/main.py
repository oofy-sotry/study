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
                    acc = SavingsAccount(name, next_account_number_balance)
                else:
                    print("잘못된 선택입니다.")
                    continue

                accounts.append(acc)
                next_account_number += 1

                print("계좌 생성 완료")
                print(acc)

        except ValueError as e:
            print("오류:", e)
