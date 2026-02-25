class DynamicArray:
    def __init__(self, initial_capacity=4):
        if initial_capacity <= 0:
            raise ValueError("initial_capacity는 1 이상이어야 합니다.")
        self._capacity = initial_capacity
        self._size = 0
        self._data = [None] * self._capacity

    def __len__(self):
        return self._size

    def __repr__(self):
        visible = [self._data[i] for i in range(self._size)]
        return f"DynamicArray(size={self._size}, capacity={self._capacity}, data={visible})"

    def _resize(self, new_capacity):
        new_data = [None] * new_capacity
        for i in range(self._size):
            new_data[i] = self._data[i]
        self._data = new_data
        self._capacity = new_capacity

    def get(self, index):
        self._check_index(index)
        return self._data[index]

    def set(self, index, value):
        self._check_index(index)
        self._data[index] = value

    def append(self, value):
        if self._size == self._capacity:
            self._resize(self._capacity * 2)

        self._data[self._size] = value
        self._size += 1

    def insert(self, index, value):
        if index < 0 or index > self._size:
            raise IndexError("insert index 범위를 벗어났습니다.")

        if self._size == self._capacity:
            self._resize(self._capacity * 2)

        for i in range(self._size, index, -1):
            self._data[i] = self._data[i - 1]

        self._data[index] = value
        self._size += 1

    def delete(self, index):
        self._check_index(index)
        removed = self._data[index]

        for i in range(index, self._size - 1):
            self._data[i] = self._data[i + 1]

        self._data[self._size - 1] = None
        self._size -= 1

        return removed

    def _check_index(self, index):
        if index < 0 or index >= self._size:
            raise IndexError("index 범위를 벗어났습니다.")

if __name__ == "__main__":
    arr = DynamicArray(initial_capacity=2)
    print("초기:", arr)

    print("--- append 테스트 ---")
    arr.append(10)
    print(arr)
    arr.append(20)
    print(arr)

    arr.append(30)
    print(arr)

    print("--- get(O(1)) 테스트 ---")
    print("arr.get(0) =", arr.get(0))
    print("arr.get(2) =", arr.get(2))

    print("--- insert(O(n)) 테스트: index=1에 15 삽입 ---")
    arr.insert(1, 15)
    print(arr)

    print("--- delete(O(n)) 테스트: index=2 삭제 ---")
    removed = arr.delete(2)
    print("삭제된 값:", removed)
    print(arr)


        
