class Stack:
    def __init__(self, debug=False):
        self._data = []
        self.debug = debug

    def _log(self, msg):
        if self.debug:
            print(msg)

    def __repr__(self):
        return f"Stack(top->bottom): {list(reversed(self._data))} (size={self.size()})"

    def push(self, x):
        self._log(f"[push] x={x} | BEFORE {self}")
        self._data.append(x)
        self._log(f"[push] x={x} | AFTER  {self}")

    def pop(self):
        self._log(f"[pop ] BEFORE {self}")
        if self.is_empty():
            self._log("[pop ] ERROR: stack is empty")
            raise IndexError("pop from empty stack")
        value = self._data.pop()
        self._log(f"[pop ] value={value} | AFTER  {self}")
        return value

    def peek(self):
        self._log(f"[peek] BEFORE {self}")
        if self.is_empty():
            self._log("[peek] ERROR: stack is empty")
            raise IndexError("peek from empty stack")
        value = self._data[-1]
        self._log(f"[peek] value={value} | AFTER(unchanged) {self}")
        return value

    def is_empty(self):
        return len(self._data) == 0

    def size(self):
        return len(self._data)
       
       
if __name__ == "__main__":
    s = Stack(debug=True)

    s.push(10)
    s.push(20)
    s.peek()
    s.pop()
    s.pop()

    try:
        s.pop()
    except IndexError as e:
        print("예외 발생:", e)        
