def insertion_sort(arr):
    a = arr[:]
    n = len(a)

    for i in range(1, n):
        key = a[i]
        j = i - 1

        print(f"\n--- {i}번째 삽입 시작 ---")
        print(f"왼쪽(정렬된 구간): {a[:i]}")
        print(f"key = {key}")
        print(f"오른쪽(미정렬 구간): {a[i:]}")

        while j >= 0 and a[j] > key:
            print(f"  이동: a[{j+1}] = a[{j}] ({a[j]})")
            a[j + 1] = a[j]
            j -= 1
            print(f"  현재 배열: {a}")

        print(f"  삽입: a[{j+1}] = {key}")
        a[j + 1] = key
        print(f"  결과 배열: {a}")

    print("\n정렬 완료!")
    return a

result = insertion_sort([5, 1, 4, 2, 8])
print("최종 결과:", result)

