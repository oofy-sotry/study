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

    def insert_debug(self, index, value):
        if index < 0 or index > self._size:
            raise IndexError("insert index 범위를 벗어났습니다.")

        if self._size == self._capacity:
            print(f"[insert_debug] capacity 꽉 참({self._capacity}) → resize({self._capacity * 2})")
            self._resize(self._capacity * 2)

        print("\n[insert_debug] 삽입 시작")
        print("  BEFORE:", self)

        print(f"  index={index} 위치에 {value} 삽입 → 뒤 요소들을 오른쪽으로 밀기")
        for i in range(self._size, index, -1):
            moved = self._data[i - 1]
            print(f"   - move: _data[{i}] = _data[{i-1}] ({moved})")
            self._data[i] = self._data[i - 1]

            self._print_internal_snapshot()

        self._data[index] = value
        self._size += 1

        print("  AFTER :", self)
        return self

    def delete_debug(self, index):
        self._check_index(index)

        print("\n[delete_debug] 삭제 시작")
        print("  BEFORE:", self)

        removed = self._data[index]
        print(f"  index={index} 값({removed}) 삭제 → 뒤 요소들을 왼쪽으로 당기기")

        for i in range(index, self._size - 1):
            pulled = self._data[i + 1]
            print(f"   - move: _data[{i}] = _data[{i+1}] ({pulled})")
            self._data[i] = self._data[i + 1]

            self._print_internal_snapshot()

        self._data[self._size - 1] = None
        self._size -= 1

        print("  AFTER :", self)
        print(f"  removed={removed}")
        return removed
       
    def _print_internal_snapshot(self):
        visible = [self._data[i] for i in range(self._size)]
        print(f"     snapshot(size={self._size}, cap={self._capacity}) visible={visible} raw={self._data}")        


if __name__ == "__main__":
    arr = DynamicArray(initial_capacity=8)

    print("초기 데이터 준비")
    for x in [10, 20, 30, 40, 50]:
        arr.append(x)
    print("초기 상태:", arr)

    print("insert 밀기 확인")
    arr.insert_debug(2, 25) 

    print("delete 당기기 확인")
    arr.delete_debug(3)

    print("\n최종 상태:", arr)
