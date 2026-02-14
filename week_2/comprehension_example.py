# 짝수만 제곱
def process_numbers(numbers):
    result = [x*x for x in numbers if x % 2 == 0]
    return result

# 60점 이상만 남기기
def filter_scores(scores):
    result = {k: v for k,v in scores.items() if v >= 60}
    return result

if __name__ == "__main__":
    nums = [1,2,3,4,5,6]
    print(process_numbers(nums))

    scores = {"a":80, "b":50, "c":90}
    print(filter_scores(scores))
