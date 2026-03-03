def bubble_sort(arr):
    a = arr[:]
    n = len(a)

    for i in range(n):
        print(f"--- {i + 1}회전 시작 ---")
        swapped = False
        for j in range(0, n - 1 - i):
            print(f"비교: {a[j]} vs {a[j + 1]}")
            if a[j] > a[j + 1]:
                print(" swap!")
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
                print(" 현재 배열:", a)
        if not swapped:
            print("이미 정렬됨 -> 종료")
            break
    return a


bubble_sort([5, 1, 4, 2, 8])

