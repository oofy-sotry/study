def dict():
    result = {x: x*x for x in range(5)}
    print("딕셔너리 컴프리헨션:", result)

if __name__ == "__main__":
    dict()
