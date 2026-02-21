from account import Account, SavingsAccount

if __name__ == "__main__":
    acc = Account("Kim", 1000)
    acc.deposit(500)
    acc.withdraw(300)
    print(acc)

    s = SavingsAccount("Lee", 1000)
    s.add_interest(0.05)
    print(s)
