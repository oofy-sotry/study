class Account:
    def __init__(self, owner, balance=0, account_number):
        self.owner = owner
        self.balance = balance
        self.account_number = account_number

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("0보다 커야 합니다")
        self.balance += amount
