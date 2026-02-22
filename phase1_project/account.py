class Account:
    def __init__(self, owner, balance=0, account_number):
        self.owner = owner
        self.balance = balance
        self.account_number = account_number

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("0보다 커야 합니다")
        self.balance += amount

    def withdrwa(self, amount):
            if amount > self.balance:
                raise ValueError("잔액이 부족합니다")
            self.balance -= amount

    def get_balance(self):
        return self.balance

    def __str__(self):
        return f"{self.owner}님의 잔액: {self.balance}"

class SavingsAccount(Account):
    def add_interest(self, rate):
        interest = self.balance * rate
        self.deposit(interest)
