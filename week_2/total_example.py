from util_example.utils.math_utils import average
from util_example.utils.list_utils import filter_even, square_all
from util_example.utils.string_utils import reverse_string

def get_numbers():
    try:
        data = input("숫자들을 쉼표로 구분하여 입력하세요")
        return [float(x.strip()) for x in data.split(",")]
    except ValueError:
        print("올바른 숫자 형식이 아닙니다")
        return None

def get_string():
    while True:
        data = input("거꾸로 표시할 글자를 입력하세요: ")
        if data.strip() == "":
            print("문자를 입력해주세요!")
        else:
            return data

def main():
    numbers = get_numbers()

    if numbers is None:
        return
    data = get_string()

    evens = filter_even(numbers)
    squared = square_all(evens)
    avg = average(squared)
    reverse = reverse_string(data)

    print("짝수", evens)
    print("제곱", squared)
    print("평균", avg)
    print("문자열 테스트", reverse)

if __name__ == "__main__":
    main()
