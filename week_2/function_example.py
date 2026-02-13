def calculate(operation, *args):
    if operation == "sum":
        return sum(args)
    elif operation == "max":
        return max(args)
    else:
        return "지원하지 않는 연산"

def create_profile(**kwargs):
    profile = {}
    for key, value in kwargs.items():
        profile[key] = value
    return profile

if __name__ == "__main__":
    print(calculate("sum", 1, 2, 3, 4))
    print(calculate("max", 10, 5, 3))
    print(create_profile(name="Kim", age=30, city="Seoul"))
