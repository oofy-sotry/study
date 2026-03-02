import heapq
from collections import deque

class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
       
def preorder(node):
    if node is None:
        return
    print(node.value, end=" ")
    preorder(node.left)
    preorder(node.right)        

def inorder(node):
    if node is None:
        return
    inorder(node.left)
    print(node.value, end=" ")
    inorder(node.right)

def level_order(root):
    if root is None:
        return

    q = deque([root])
    while q:
        node = q.popleft()
        print(node.value, end=" ")

        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)

           
search_logs = [
    "python", "heap", "tree", "python", "dict",
    "heap", "python", "tree", "heap", "graph",
    "tree", "tree", "heap"
]            


freq = {}

for word in search_logs:
    if word in freq:
        freq[word] += 1
    else:
        freq[word] = 1

print("검색어 빈도수:", freq)

heap = []

for word, count in freq.items():
    heapq.heappush(heap, (-count, word))  

top_words = []
for _ in range(3):
    count, word = heapq.heappop(heap)
    top_words.append((word, -count))

print("상위 3개 검색어:", top_words)

root = Node(top_words[0])
root.left = Node(top_words[1])
root.right = Node(top_words[2])

print("전위 순회 (Preorder):")
preorder(root)

print("중위 순회 (Inorder):")
inorder(root)

print("레벨 순회 (BFS):")
level_order(root)
