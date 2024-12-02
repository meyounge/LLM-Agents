import random
import math
import time

def shor(N):
    if N <= 1:
        return 1
    if N % 2 == 0:
        return 2
    for a in range(2, N):
        if math.gcd(a, N) != 1:
            return a
    x = random.randint(1, N - 1)
    if math.gcd(x, N) != 1:
        return x
    r = 0
    while pow(x, r, N) != 1:
        r += 1
    if r % 2 == 0:
        return math.gcd(x**(r//2) + 1, N), math.gcd(x**(r//2) - 1, N)
    return N

# Test case
N = 35
print(shor(N))