class ExprNode:
    def __init__(self, value, left=None, right=None):
        self.value = value  
        self.left = left
        self.right = right

def eval_expr_debug(node, depth=0):
    indent = "  " * depth

    if isinstance(node.value, (int, float)):
        print(f"{indent}return {node.value}")
        return node.value

    op = node.value
    print(f"{indent}op '{op}': evaluate LEFT")
    left_val = eval_expr_debug(node.left, depth + 1)

    print(f"{indent}op '{op}': evaluate RIGHT")
    right_val = eval_expr_debug(node.right, depth + 1)

    if op == "+":
        result = left_val + right_val
    elif op == "-":
        result = left_val - right_val
    elif op == "*":
        result = left_val * right_val
    elif op == "/":
        result = left_val / right_val
    else:
        raise ValueError(f"Unknown operator: {op}")

    print(f"{indent}compute {left_val} {op} {right_val} = {result}")
    return result

tree = ExprNode("*",
                ExprNode("+", ExprNode(3), ExprNode(5)),
                ExprNode(2))

print("=== EVAL DEBUG ===")
ans = eval_expr_debug(tree)
print("answer =", ans)


