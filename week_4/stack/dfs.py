def dfs_iterative_debug(graph, start):
    visited = set()
    stack = [start]

    step = 0
    print("===== DFS DEBUG START =====")
    print(f"start node: {start}")
    print("----------------------------------")

    while stack:
        print(f"[STEP {step}]")
        print(f"  stack (top->bottom): {list(reversed(stack))}")
        print(f"  visited: {visited}")

        node = stack.pop()
        print(f"  pop -> {node}")

        if node in visited:
            print(f"  {node} already visited -> continue")
            print("----------------------------------")
            step += 1
            continue

        visited.add(node)
        print(f"  visit {node}")
        print(f"  visited updated: {visited}")

        neighbors = graph[node]
        print(f"  neighbors of {node}: {neighbors}")

        for nxt in reversed(neighbors):
            if nxt not in visited:
                stack.append(nxt)
                print(f"    push {nxt}")

        print("----------------------------------")
        step += 1

    print("===== DFS END =====")
    print(f"final visited: {visited}")


graph = {
            1: [2, 3],
            2: [4],
            3: [5],
            4: [],
            5: []
        }

dfs_iterative_debug(graph, 1)
