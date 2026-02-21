from account import Account

if __name__ == "__main__":
    acc = Account("Kim", 1000)
    acc.deposit(500)
    acc.withdraw(300)
    print(acc)
