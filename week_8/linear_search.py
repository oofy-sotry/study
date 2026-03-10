def linear_search(arr, target):
    print(f"[선형 탐색 시작] target={target}")

    for i in range(len(arr)):
        print(f" arr[{i}] = {arr[i]} 확인 중")
        if arr[i] == target:
            print(f"-> 찾음! index = {i}")
            return i
    print("-> 끝까지 찾이 못함")
    return -1

arr = [3, 8, 1, 10, 7]
linear_search(arr, 10)
