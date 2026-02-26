history = []

history.append("page1")
history.append("page2")
history.append("page3")

#print(history)
#print(history.pop())
#print(history.pop())

current = history.pop()
print("삭제된 페이지", current)
print("현재 페이지", history[-1])
