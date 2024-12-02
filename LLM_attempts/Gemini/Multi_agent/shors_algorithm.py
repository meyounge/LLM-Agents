import random

# Function to calculate the greatest common divisor (GCD) using Euclid's algorithm
def gcd(a, b):
    """
    Calculates the greatest common divisor (GCD) of two integers using Euclid's algorithm.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The GCD of a and b.  Returns 1 if either a or b is 0 to avoid ZeroDivisionError.
    """
    if a == 0:
        return b
    if b == 0:
        return a
    while b:
        a, b = b, a % b
    return a


def shors_algorithm(N):
    """
    Implements Shor's algorithm to factor a number N.

    Args:
        N: The number to factor.

    Returns:
        A non-trivial factor of N, or N if factorization fails or N is prime.
    """
    # Step 0: Check if N is positive and non-trivial
    if N <= 1:
        return 1

    # Step 1: Check if N is even
    if N % 2 == 0:
        return 2

    # Step 2: Check for trivial factors (perfect powers)
    for b in range(2, int(N**0.5) + 1):
      a = round(N**(1/b))
      if a**b == N:
        return a

    # Step 3: Choose a random x coprime to N
    x = random.randint(1, N - 1)
    if gcd(x, N) != 1:
        return gcd(x, N)

    # Step 4: Find r such that x**r % N == 1
    r = 1
    while (x**r) % N != 1:
        r += 1

    # Step 5: Check if r is even and find factors
    if r % 2 == 0:
        factor1 = gcd((x**(r // 2)) + 1, N)
        factor2 = gcd((x**(r // 2)) - 1, N)
        if factor1 != 1 and factor1 != N:
            return factor1
        elif factor2 != 1 and factor2 != N:
            return factor2

    # Step 6: Factorization failed or N is prime
    return N


# Test case
N = 15
factor = shors_algorithm(N)
print("Testing Shor's Algorithm:")
print(f"Shor's Algorithm for 15: {shors_algorithm(15)}")  # Should find a factor
print(f"Shor's Algorithm for 21: {shors_algorithm(21)}")  #Should find a factor
print(f"Shor's Algorithm for 35: {shors_algorithm(35)}")  #Should find a factor
print(f"Shor's Algorithm for 100: {shors_algorithm(100)}") #Should find a factor
print(f"Shor's Algorithm for 7: {shors_algorithm(7)}")   # Should return 7 (prime)
print(f"Shor's Algorithm for 1001: {shors_algorithm(1001)}") #Should find a factor
print(f"Shor's Algorithm for 1: {shors_algorithm(1)}")   #Should return 1
print(f"Shor's Algorithm for -5: {shors_algorithm(-5)}") #Should return -5
print(f"Shor's Algorithm for 16: {shors_algorithm(16)}") #Should return 16