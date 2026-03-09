import random
import time

def bubble_sort(arr):
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        for j in range(n - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def selection_sort(arr):
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

def insertion_sort(arr):
    arr = arr.copy()
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = []
    right = []
    for x in arr[1:]:
        if x < pivot:
            left.append(x)
        else:
            right.append(x)
    return quick_sort(left) + [pivot] + quick_sort(right)

def measure_time(sort_func, data):
    start = time.time()
    sorted_data = sort_func(data)
    end = time.time()
    return sorted_data, end - start

if __name__ == "__main__":
    data = [random.randint(1, 10000) for _ in range(1000)]
    print("원본 데이터 일부(앞 20개):")
    print(data[:20])
    print()

    bubble_result, bubble_time = measure_time(bubble_sort, data)
    selection_result, selection_time = measure_time(selection_sort, data)
    insertion_result, insertion_time = measure_time(insertion_sort, data)
    merge_result, merge_time = measure_time(merge_sort, data)
    quick_result, quick_time = measure_time(quick_sort, data)

    print("===== 정렬 성능 비교 결과 =====")
    print(f"Bubble Sort    : {bubble_time:.6f} 초")
    print(f"Selection Sort : {selection_time:.6f} 초")
    print(f"Insertion Sort : {insertion_time:.6f} 초")
    print(f"Merge Sort     : {merge_time:.6f} 초")
    print(f"Quick Sort     : {quick_time:.6f} 초")
    print()

    expected = sorted(data)

    print("===== 정렬 결과 검증 =====")
    print("Bubble Sort    :", bubble_result == expected)
    print("Selection Sort :", selection_result == expected)
    print("Insertion Sort :", insertion_result == expected)
    print("Merge Sort     :", merge_result == expected)
    print("Quick Sort     :", quick_result == expected)
    print()

    print("정렬된 데이터 일부(앞 20개):")
    print(expected[:20])
