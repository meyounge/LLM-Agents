# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:44:54 2024

@author: micha
"""

import math

def shors_algorithm(N):
    # Step 0: Ensure N is positive and non-trivial
    if N <= 1:
        return 1

    # Step 1: Ensure N is not an even number
    if N % 2 == 0:
        return 2

    # Step 2: Find a number 'a' such that a^(N-1) % N != 1
    for a in range(2, N):
        if pow(a, N-1, N) != 1:
            return a

    # Step 3: Choose a random positive non-trivial integer 'x' that is less than N
    x = random.randint(2, N-1)

    # Check if 'x' has a common factor with N
    if math.gcd(x, N) > 1:
        return x

    # Step 4: Find r such that x**r % N == 1
    r = 0
    while (pow(x, r, N) != 1):
        r += 1

    # Step 5: If r is even, we've found a number with a common factor to N
    if r % 2 == 0:
        gcd_value = math.gcd(x**(r//2) + 1, N)
        gcd_value2 = math.gcd(x**(r//2) - 1, N)
        return gcd_value if gcd_value > gcd_value2 else gcd_value2
    else:
        return N

# Test case
test_case = 35
print(shors_algorithm(test_case))