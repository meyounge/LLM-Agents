# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:42:14 2024

@author: micha
"""

import math

def classical_shor(N):
    """
    A simplified classical attempt at factoring N.  This is NOT a true Shor's algorithm.
    """

    # Step 0: Check for trivial cases
    if N <= 1 or isinstance(N, float):  #Added check for floats
        return 1
    if N % 2 == 0:
        return 2

    # Step 2: Check for perfect powers (simplified)
    for b in range(2, int(math.log2(N)) + 1):  #Check for perfect powers up to log2(N)
        a = round(N**(1/b))
        if a**b == N:
            return a

    # Step 3: Choose a random x and check for gcd
    x = 1
    while x == 1 or math.gcd(x, N) > 1: #Ensure x and N are coprime
        x = 1 + int(N * random.random())
    if math.gcd(x, N) > 1:
        return math.gcd(x, N)

    # Step 4: Find r (simplified - this is a major simplification!)
    r = 1
    while (x**r) % N != 1:
        r += 1
    #This loop will continue until it finds a suitable r which might take a long time or never finish, hence this simplified version is not ideal
        if r > N:
            return N

    # Step 5: Check for factors
    if r % 2 == 0:
        factor1 = math.gcd((x**(r//2)) + 1, N)
        factor2 = math.gcd((x**(r//2)) - 1, N)
        if factor1 > 1 and factor1 < N:
            return factor1
        if factor2 > 1 and factor2 < N:
            return factor2

    # Step 6: Failed or prime
    return N


import random
# Step 7: Test case
N = 35
result = classical_shor(N)
print(f"Attempting to factor {N}: Result = {result}")

N = 15
result = classical_shor(N)
print(f"Attempting to factor {N}: Result = {result}")

N = 21
result = classical_shor(N)
print(f"Attempting to factor {N}: Result = {result}")

N = 1024
result = classical_shor(N)
print(f"Attempting to factor {N}: Result = {result}") #This will likely fail because of the simplified r-finding