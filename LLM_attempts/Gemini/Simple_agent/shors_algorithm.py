import random

def gcd(a, b):
    """Euclidean algorithm for greatest common divisor."""
    while b:
        a, b = b, a % b
    return a

def modular_exponentiation(base, exponent, modulus):
    """Efficient modular exponentiation."""
    result = 1
    base %= modulus
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent >>= 1
        base = (base * base) % modulus
    return result

def shors_algorithm(N):
    """Shor's algorithm for factoring."""

    # Step 0: Check for trivial cases
    if N <= 1 or not isinstance(N, int):  # Added type check
        return 1
    if N % 2 == 0:
        return 2

    # Step 2: Check for perfect powers (brute force for simplicity)
    for b in range(2, int(N**0.5) + 1):
        a = round(N**(1/b))
        if a**b == N:
            return a

    # Step 3: Choose random x
    while True:
        x = random.randint(2, N - 1)
        if gcd(x, N) == 1:
            break
        else:
            return gcd(x,N) #return the gcd found

    # Step 4: Find r (order finding - simplified for demonstration)
    # This is a simplification. A proper implementation would use quantum computation here.
    r = 1
    while modular_exponentiation(x, r, N) != 1:
        r += 1
    
    #Added to handle cases where r is odd
    if r % 2 != 0:
        return N


    # Step 5: Factor if r is even
    factor1 = gcd(modular_exponentiation(x, r // 2) + 1, N)
    factor2 = gcd(modular_exponentiation(x, r // 2) - 1, N)

    if 1 < factor1 < N:
        return factor1
    elif 1 < factor2 < N:
        return factor2
    else:
        return N #Failure


# Step 7: Test case
if __name__ == "__main__":
    number_to_factor = 35
    result = shors_algorithm(number_to_factor)
    print(f"The factor found for {number_to_factor} is: {result}")

