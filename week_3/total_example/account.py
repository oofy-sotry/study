class Account:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("0보다 커야 합니다.")
        self.balance += amount

    def withdraw(self, amount):
        if amount > self.balance:
            raise ValueError("잔액 부족")
        self.balance -= amount

    def __str__(self):
        return f"{self.owner}님의 잔액: {self.balance}"

class SavingsAccount(Account):
    def add_interest(self, rate):
        interest = self.balance * rate
        self.deposit(interest)
