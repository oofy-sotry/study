def is_valid_parentheses_debug(s: str) -> bool:
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    openings = set("([{")
    closings = set(")]}")

    print(f"input: {s!r}")
    print(f"pairs: {pairs}")

    for i, ch in enumerate(s):
        print(f"[{i}] ch={ch!r} | BEFORE stack(top->bottom)={list(reversed(stack))}")

        if ch in openings:
            stack.append(ch)
            print(f"    action: push {ch!r}")
            print(f"    AFTER  stack(top->bottom)={list(reversed(stack))}")

        elif ch in closings:
            if not stack:
                print("    action: FAIL (closing bracket but stack is empty)")
                print("===== RESULT: False =====")
                return False

            top = stack.pop()
            expected = pairs[ch]
            print(f"    action: pop -> top={top!r}, expected opening for {ch!r} is {expected!r}")

            if top != expected:
                print("    action: FAIL (mismatch)")
                print(f"    reason: top {top!r} != expected {expected!r}")
                print("===== RESULT: False =====")
                return False

            print("    action: OK (matched)")
            print(f"    AFTER  stack(top->bottom)={list(reversed(stack))}")

        else:
            print("    action: ignore (not a bracket)")

        print("----------------------------------")

    if len(stack) == 0:
        print("end: stack is empty -> all matched")
        print("===== RESULT: True =====")
        return True
    else:
        print(f"end: stack not empty -> remaining openings: {stack}")
        print("===== RESULT: False =====")
        return False


print(is_valid_parentheses_debug("()[]{}"))
print()
print(is_valid_parentheses_debug("([)]"))
print()
print(is_valid_parentheses_debug("(((())))"))


