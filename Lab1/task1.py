def OR(a: int, b: int) -> int:
    return 1 if (a or b) else 0

def AND(a: int, b: int) -> int:
    return 1 if (a and b) else 0

def XOR(a: int, b: int) -> int:
    return AND(OR(a, b), 0 if AND(a, b) else 1)

table = [(0,0),(0,1),(1,0),(1,1)]
for x1, x2 in table:
    print((x1, x2), "->", XOR(x1, x2))