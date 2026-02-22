from account import Account, SavingsAccount

accounts = []
next_account_number = 1000

def find_account(account_number):
    for acc in accounts:
        if acc.account_number == account_number:
            return acc
    return None

