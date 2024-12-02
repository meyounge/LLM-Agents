def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def power_mod(x, r, N):
    res = 1
    x = x % N
    while r > 0:
        if r % 2 == 1:
            res = (res * x) % N
        r = r // 2
        x = (x * x) % N
    return res

def find_order(x, N):
    for r in range(1, N + 1):
        if power_mod(x, r, N) == 1:
            return r
    return -1
