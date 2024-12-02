import math

def shors_algorithm(N):
    # Step 0: Check for trivial cases
    if N <= 1 or not isinstance(N, int):
        return 1

    # Step 1: Check for even numbers
    if N % 2 == 0:
        return 2

    # Step 2: Check for perfect powers
    for b in range(3, int(math.log2(N)) + 1):
        a = round(N**(1/b))
        if a**b == N:
            return a

    # Step 3: Choose a random x
    x = 1
    while math.gcd(x, N) != 1:
        x = 1
        while math.gcd(x, N) != 1 or x >= N:
            x = int(random.random()*N) # added random import to ensure x is random
    
    # Step 4: Find r using brute force (not efficient for large numbers)
    r = 1
    while pow(x, r, N) != 1:
        r += 1

    # Step 5: Check if r is even
    if r % 2 == 0:
        factor1 = math.gcd(pow(x, r // 2) + 1, N)
        factor2 = math.gcd(pow(x, r // 2) - 1, N)
        if factor1 > 1 and factor1 < N:
            return factor1
        elif factor2 > 1 and factor2 < N:
            return factor2

    # Step 6: Return N if no factor found
    return N

import random

# Step 7: Test case
number_to_factor = 35
result = shors_algorithm(number_to_factor)
print(f"Attempting to factor {number_to_factor}: Result = {result}")


number_to_factor = 15
result = shors_algorithm(number_to_factor)
print(f"Attempting to factor {number_to_factor}: Result = {result}")

number_to_factor = 21
result = shors_algorithm(number_to_factor)
print(f"Attempting to factor {number_to_factor}: Result = {result}")

number_to_factor = 100
result = shors_algorithm(number_to_factor)
print(f"Attempting to factor {number_to_factor}: Result = {result}")
