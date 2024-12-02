import random
from math import gcd, isqrt

def shors_algorithm(N):
    # Step 0: Ensure N is positive and non-trivial
    if N <= 1:
        return 1
    
    # Step 1: Ensure N is not an even number
    if N % 2 == 0:
        return 2
    
    # Step 2: Ensure there is no 'a' such that a**b == N
    for a in range(2, isqrt(N) + 1):
        b = 2
        while (result := a ** b) <= N:
            if result == N:
                return a
            b += 1
    
    # Step 3: Choose a random x and ensure gcd(x, N) == 1
    x = random.randint(2, N - 1)
    g = gcd(x, N)
    if g > 1:
        return g
    
    # Step 4: Find r such that x**r % N == 1
    r = None
    for i in range(1, N):
        if pow(x, i, N) == 1:
            r = i
            break
    
    if r is None:
        return N  # Failed to find such an r
    
    # Step 5: Check if r is even and find gcd from x**(r/2) ± 1
    if r % 2 == 0:
        xr2 = pow(x, r // 2, N)
        factors = [gcd(xr2 + 1, N), gcd(xr2 - 1, N)]
        for factor in factors:
            if 1 < factor < N:
                return factor
    
    # Step 6: Return N if no factors are found
    return N

# Step 7: Write a test case
if __name__ == "__main__":
    test_number = 35
    print(f"Testing Shor's Algorithm on {test_number}")
    result = shors_algorithm(test_number)
    print(f"Output: {result}")