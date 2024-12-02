import math
import random
from math import gcd

def classical_shor(N):
    # Step 0: Ensure N is positive and non-trivial
    if N <= 1:
        return 1

    # Step 1: Ensure N is not an even number
    if N % 2 == 0:
        return 2

    # Step 2: Check if N is a perfect power (a^b == N)
    for a in range(2, int(math.sqrt(N)) + 1):
        b = 2
        while (a ** b) <= N:
            if a ** b == N:
                return a
            b += 1

    # Step 3: Choose a random integer 'x' such that 1 < x < N and gcd(x, N) == 1
    x = random.randint(2, N - 1)
    if gcd(x, N) != 1:
        return gcd(x, N)

    # Step 4: Find the order 'r' such that x^r % N == 1
    r = None
    for candidate_r in range(1, N):
        if pow(x, candidate_r, N) == 1:
            r = candidate_r
            break

    # If no 'r' is found, return N (possible failure or N is prime)
    if r is None:
        return N

    # Step 5: If 'r' is even, find factors using x^(r/2) ± 1
    if r % 2 == 0:
        xr_2 = pow(x, r // 2, N)
        factor1 = gcd(xr_2 + 1, N)
        factor2 = gcd(xr_2 - 1, N)
        if 1 < factor1 < N:
            return factor1
        if 1 < factor2 < N:
            return factor2

    # Step 6: If no factors are found, return N (possible failure or N is prime)
    return N

# Step 7: Write a test case
def test_classical_shor():
    test_number = 35
    print(f"Testing classical_shor with N = {test_number}")
    result = classical_shor(test_number)
    print(f"Result: {result}")
    return result

# Step 8: Run the file and verify it works as intended
if __name__ == "__main__":
    # Run the test case multiple times to observe different results
    for _ in range(5):
        factor = test_classical_shor()
        if factor not in {1, test_classical_shor}:
            print(f"Found non-trivial factor: {factor}")
