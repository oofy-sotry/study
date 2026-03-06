def merge_sort(arr, depth=0):
    indent = "  " * depth
    print(f"{indent}[merge_sort 호출] arr = {arr}")

    if len(arr) <= 1:
        print(f"{indent}길이가 1 이하이므로 그대로 반환: {arr}")
        return arr

    mid = len(arr) // 2
    print(f"{indent}mid = {mid}")
    print(f"{indent}왼쪽 분할: {arr[:mid]}")
    print(f"{indent}오른쪽 분할: {arr[mid:]}")

    left = merge_sort(arr[:mid], depth + 1)
    right = merge_sort(arr[mid:], depth + 1)

    merged = merge(left, right, depth)
    print(f"{indent}[merge_sort 종료] {left} + {right} -> {merged}")
    return merged


def merge(left, right, depth=0):
    indent = "  " * depth
    print(f"{indent}[merge 시작] left = {left}, right = {right}")

    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        print(f"{indent}  비교: left[{i}]={left[i]} , right[{j}]={right[j]}")

        if left[i] < right[j]:
            print(f"{indent}  -> {left[i]} 가 더 작으므로 result에 추가")
            result.append(left[i])
            i += 1
        else:
            print(f"{indent}  -> {right[j]} 가 더 작거나 같으므로 result에 추가")
            result.append(right[j])
            j += 1

        print(f"{indent}  현재 result = {result}")

    if i < len(left):
        print(f"{indent}left에 남은 값 {left[i:]}을 result에 추가")
    if j < len(right):
        print(f"{indent}right에 남은 값 {right[j:]}을 result에 추가")

    result.extend(left[i:])
    result.extend(right[j:])

    print(f"{indent}[merge 종료] result = {result}")
    return result


data = [5, 3, 8, 4, 2]
print("초기 데이터:", data)
print()

sorted_data = merge_sort(data)

print()
print("최종 정렬 결과:", sorted_data)

