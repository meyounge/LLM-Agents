import random
import math

def shors_algorithm(N):
    # Step 0
    if N <= 1:
        return 1
    # Step 1
    if N % 2 == 0:
        return 2
    # Step 2
    for a in range(2, int(math.sqrt(N)) + 1):
        if N % a == 0:
            return a
    # Step 3
    x = random.randint(2, N - 1)
    gcd = math.gcd(x, N)
    if gcd > 1:
        return gcd
    # Step 4
    r = 1
    while (x ** r) % N != 1:
        r += 1
    # Step 5
    if r % 2 == 0:
        factor1 = math.gcd(x ** (r // 2) + 1, N)
        factor2 = math.gcd(x ** (r // 2) - 1, N)
        if factor1 > 1:
            return factor1
        if factor2 > 1:
            return factor2
    # Step 6
    return N
# Test Case
print(shors_algorithm(35))