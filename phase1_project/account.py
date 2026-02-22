class Account:
    def __init__(self, owner, account_number, balance = 0):
        self.owner = owner
        self.balance = balance
        self.account_number = account_number

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("0보다 커야 합니다")
        self.balance += amount

    def withdrwa(self, amount):
        if amount <= 0:
            raise ValueError("0보다 커야 합니다")
        if amount > self.balance:
            raise ValueError("잔액 부족")
        self.balance -= amount

    def get_balance(self):
        return self.balance

    def __str__(self):
        return f"[{self.account_number}] {self.owner}님의 잔액: {self.balance}"

class SavingsAccount(Account):
    def add_interest(self, rate):
        if rate <= 0:
            raise ValueError("이율은 0보다 커야 합니다")
        interest = self.balance * rate
        self.deposit(interest)
