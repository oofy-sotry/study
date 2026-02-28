from collections import deque

def bfs_debug(graph, start):
    visited = set()
    queue = deque([start])

    step = 0
    print(f"\n=== BFS 시작 (start={start}) ===\n")

    while queue:
        step += 1
        print(f"\n[STEP {step}]")
        print(f"  현재 queue 상태: {list(queue)}")
        print(f"  현재 visited 상태: {visited}")

        node = queue.popleft()
        print(f"  ▶ popleft() → {node}")

        if node in visited:
            print(f"  ⚠ {node} 는 이미 방문됨 → continue")
            continue

        visited.add(node)
        print(f"  ✓ {node} 방문 처리")
        print(f"  출력: {node}")

        neighbors = graph[node]
        print(f"  {node}의 이웃: {neighbors}")

        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append(neighbor)
                print(f"    → {neighbor} 를 queue에 추가")

        print(f"  queue 최종 상태: {list(queue)}")

    print("\n=== BFS 종료 ===")
   
   
if __name__ == "__main__":
    graph = {
        1: [2, 3],
        2: [4],
        3: [5],
        4: [],
        5: []
    }

    bfs_debug(graph, 1)    

