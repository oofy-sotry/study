stack = []

stack.append(10)
stack.append(20)
stack.append(30)
print(stack)

print(stack[-1])

if stack:
    x = stack.pop()
    print(x)
    print(stack)
else:
    print("스택이 비어있음")
