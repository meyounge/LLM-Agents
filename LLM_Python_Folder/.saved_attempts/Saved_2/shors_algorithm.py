import random
import math
import sympy

def shors_algorithm(N):
    if not isinstance(N, int) or N <= 0:
        return 1
    if N % 2 == 0:
        return 2
    # Check if N is a perfect power of an integer
    for a in range(2, int(math.sqrt(N)) + 1):
        for b in range(2, int(math.sqrt(N)) + 1):
            if a**b == N:
                return a**b
    # Choose a random integer x
    x = random.randint(2, N - 1)
    # Check that x does not share a common factor with N
    if math.gcd(x, N) != 1:
        return math.gcd(x, N)
    # Find r such that x**r % N == 1
    r = 0
    while (x**r % N != 1):
        r += 1
    # If r is even, we've found a number with a common factor to N
    if r % 2 == 0:
        return x**(r // 2) + 1, x**(r // 2) - 1
    # Otherwise, return N
    return N

# Test case
N = 35
print(shors_algorithm(N))