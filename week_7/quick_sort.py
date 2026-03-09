def quick_sort(arr, depth=0):
    indent = "  " * depth
    print(f"{indent}[quick_sort 호출] arr = {arr}")

    # 종료 조건
    if len(arr) <= 1:
        print(f"{indent}길이가 1 이하이므로 그대로 반환: {arr}")
        return arr

    # pivot 선택
    pivot = arr[0]
    print(f"{indent}pivot 선택: {pivot}")

    left = []
    right = []

    # pivot 제외 나머지 검사
    for x in arr[1:]:
        print(f"{indent}  {x} 와 pivot({pivot}) 비교")

        if x < pivot:
            print(f"{indent}  -> {x} 는 pivot보다 작으므로 left에 추가")
            left.append(x)
        else:
            print(f"{indent}  -> {x} 는 pivot보다 크거나 같으므로 right에 추가")
            right.append(x)

    print(f"{indent}분할 결과 left = {left}, pivot = {pivot}, right = {right}")

    # 재귀 호출
    print(f"{indent}left 정렬 시작")
    sorted_left = quick_sort(left, depth + 1)

    print(f"{indent}right 정렬 시작")
    sorted_right = quick_sort(right, depth + 1)

    result = sorted_left + [pivot] + sorted_right

    print(f"{indent}[quick_sort 종료] {sorted_left} + [{pivot}] + {sorted_right} -> {result}")

    return result


data = [5,3,8,4,2]

print("초기 데이터:", data)
print()

sorted_data = quick_sort(data)

print()
print("최종 정렬 결과:", sorted_data)

