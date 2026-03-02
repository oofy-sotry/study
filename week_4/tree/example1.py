from collections import deque

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def preorder_debug(node, depth=0):
    indent = "  " * depth
    if node is None:
        print(f"{indent}- None (stop)")
        return

    print(f"{indent}- visit {node.value} (PRE: Root)")

    print(f"{indent} go LEFT")
    preorder_debug(node.left, depth + 1)

    print(f"{indent} go RIGHT")
    preorder_debug(node.right, depth + 1)

def inorder_debug(node, depth=0):
    indent = " " * depth
    if node is None:
        print(f"{indent}- None (stop)")
        return

    print(f"{indent} go LEFT")
    inorder_debug(node.left, depth + 1)

    print(f"{indent}- visit {node.value} (IN:Root)")

    print(f"{indent} go RIGHT")
    inorder_debug(node.right, depth + 1)

def postorder_debug(node, depth=0):
    indent = " " * depth
    if node is None:
        print(f"{indent}- None (stop)")
        return

    print(f"{indent} go LEFT")
    postorder_debug(node.left, depth + 1)

    print(f"{indent}- visit {node.value} (IN:Root)")

    print(f"{indent} go RIGHT")
    postorder_debug(node.right, depth + 1)

def level_order_debug(root):
    if root is None:
        print("empty tree")       
        return

    q = deque([(root, 0)])
    print("BFS start")

    while q:
        node, depth = q.popleft()
        print(f"- pop {node.value} (depth={depth})")

        if node.left:
            print(f" push LEFT {node.left.value}")
            q.append((node.left, depth + 1))
        if node.right:
            print(f" push RIGHT {node.right.value}")
            q.append((node.right, depth + 1))
    
root = Node("A", 
            Node("B", Node("D"), Node("E")), 
            Node("C", None, Node("F")))

print("PREOEDER DEBUG")
preorder_debug(root)
print("==============================")

print("INORDER DEBUG")
inorder_debug(root)
print("==============================")

print("POSTORDER DEBUG")
postorder_debug(root)
print("==============================")

print("LEVEL ORDER DEBUG")
level_order_debug(root)
print("==============================")


