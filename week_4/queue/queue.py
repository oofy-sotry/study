class Queue:
    def __init__(self):
        self.data = []

    def enqueue(self, x):
        self.data.append(x)

    def dequeue(self):
        if not self.is_empty():
            return self.data.pop(0)

    def front(self):
        if not self.is_empty():
            return self.data[0]

    def is_empty(self):
        return len(self.data) == 0

if __name__ == "__main__":
    q = Queue()

    q.enqueue(10)
    q.enqueue(20)
    q.enqueue(30)

    print("queue data", q.data)
    print("front", q.front())
    print("dequeue", q.dequeue())
    print("front", q.front())
    print("queue data", q.data)

