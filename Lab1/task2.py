def step_fn(z):
    return 1 if z >= 0 else 0


def node_or(x1, x2):
    w1, w2, bias = 1, 1, -0.5
    net = w1 * x1 + w2 * x2 + bias
    return step_fn(net)


def node_and(x1, x2):
    w1, w2, bias = 1, 1, -1.5
    net = w1 * x1 + w2 * x2 + bias
    return step_fn(net)


def xor_net(x1, x2):
    h1 = node_or(x1, x2)
    h2 = node_and(x1, x2)
    w1, w2, b_out = 1, -1, -0.5
    net_out = w1 * h1 + w2 * h2 + b_out
    return step_fn(net_out), h1, h2


samples = [(0, 0), (0, 1), (1, 0), (1, 1)]
expected = [0, 1, 1, 0]

print("Перевірка роботи XOR\n" + "=" * 40)
print("x1  x2  |  h1  h2  |  XOR")
print("-" * 40)

correct = True
for x1, x2 in samples:
    y, h1, h2 = xor_net(x1, x2)
    print(f"{x1}   {x2}   |   {h1}   {h2}   |   {y}")
    correct = correct and (y == expected[samples.index((x1, x2))])

print("=" * 40)

print("""
Структура мережі:
-----------------
Вхід: x1, x2
Схований рівень:
    h1 = step(x1 + x2 - 0.5)
    h2 = step(x1 + x2 - 1.5)
Вихід:
    XOR = step(h1 - h2 - 0.5)
""")
