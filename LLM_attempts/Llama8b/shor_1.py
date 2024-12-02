# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:45:30 2024

@author: micha
"""

def find_r(x, N):
    r = 0
    while (pow(x, r, N) != 1):
        r += 1
    return r

def find_factors(x, N, r):
    factors = []
    for i in range(1, r+1):
        if (x**((2*i)-1) % N == 1):
            factors.append(N // (x**((2*i)-1)))
        if (x**((2*i)-2) % N == 1):
            factors.append(N // (x**((2*i)-2)))
    return factors

def shors_algorithm(N):
    if N <= 1:
        return 1
    if N % 2 == 0:
        return 2
    for a in range(2, N):
        if pow(a, N-1, N) != 1:
            return a
    for x in range(2, N):
        if gcd(x, N) > 1:
            return x
    r = find_r(x, N)
    if r % 2 == 0:
        gcd_value = gcd(x**(r//2) + 1, N)
        gcd_value2 = gcd(x**(r//2) - 1, N)
        return gcd_value if gcd_value > gcd_value2 else gcd_value2
    else:
        return N

def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

test_case = 35
print(shors_algorithm(test_case))