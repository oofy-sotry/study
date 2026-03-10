def binary_search(arr, target):
    left = 0
    right = len(arr) - 1

    print(f"[이진 탐색 시작]  target={target}")

    while left <= right:
        mid = (left + right) // 2
        print(f" left={left}, right={right}, mid={mid}, arr[mid]={arr[mid]}")

        if arr[mid] == target:
            print(f"-> 찾음! index={mid}")
            return mid
        elif arr[mid] < target:
            print("-> target이 더 크므로 오른쪽 탐색")
            left = mid + 1
        else:
            print("-> target이 더 작으므로 왼쪽 탐색")
            right = mid -1
    print("-> 찾지 못함")
    return -1


arr = [1, 3, 5, 7, 9, 11, 13]
binary_search(arr, 9)

