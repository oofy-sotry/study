class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, value):
        new_node = Node(value)

        if not self.head:
            self.head = new_node
            return

        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def print_list(self):
        current = self.head
        while current:
            print(current.value, end=" -> ")
            current = current.next
        print("None")

    def get(self, index):
        if index < 0:
            raise IndexError("index는 0 이상이어야 합니다")
        current = self.head
        steps = 0

        while current and steps < index:
            print(f"[get] step={steps} 현재 노드={current.value} -> next로 이동")
            current = current.next
            steps += 1

        if not current:
            raise IndexError("index 범위를 벗어났습니다")

        print(f"[get] step={steps} 도착! 값={current.value}")
        return current.value

    def find_node(self, target_value):
        current = self.head
        while current:
            if current.value == target_value:
                return current
            current = current.next
        return None

    def insert_after(self, value, new_value):
        node = self.find_node(value)
        if not node:
            raise ValueError(f"{value}를 가진 노드를 찾을 수 없습니다")

        print(f"[insert_after] '{value}' 노드를 찾음. 이제 링크만 바꾸면 삽입 끝")

        new_node = Node(new_value)

        new_node.next = node.next
        node.next = new_node

        print(f"[insert_after] '{value}' 뒤에 '{new_value}' 삽입 완료")


if __name__ == "__main__":
    ll = LinkedList()

    for x in [10, 20, 30, 40, 50]:
        ll.append(x)

    print("초기 리스트")
    ll.print_list()

    print("get(index) 테스트: get(3)")
    ll.get(3)

    print("insert_after 테스트 : insert_after(30, 35)")
    ll.insert_after(30, 35)
    ll.print_list()
