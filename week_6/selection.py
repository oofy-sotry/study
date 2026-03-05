def selection_sort(arr):
    a = arr[:]
    n = len(a)

    for i in range(n - 1):
        print(f"\n--- {i+1}회전 시작 (i={i}) ---")
        min_idx = i
        print(f"시작 최소값 후보: a[{min_idx}] = {a[min_idx]}")

        for j in range(i + 1, n):
            print(f"비교: a[{j}]={a[j]} vs a[{min_idx}]={a[min_idx]}")
            if a[j] < a[min_idx]:
                min_idx = j
                print(f"  -> 최소값 후보 갱신! min_idx={min_idx}, 값={a[min_idx]}")

        # i회전이 끝났으니 i와 min_idx를 swap
        if min_idx != i:
            print(f"swap: a[{i}]={a[i]} <-> a[{min_idx}]={a[min_idx]}")
            a[i], a[min_idx] = a[min_idx], a[i]
        else:
            print("swap 없음 (이미 현재 위치가 최소값)")

        print("현재 배열:", a)

    print("\n정렬 완료!")
    return a


arr = [5,1,4,2,8]
result = selection_sort(arr)

print("원본 : ", arr)
print("결과:", result)
