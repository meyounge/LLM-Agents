import random
import math

# Function to find the greatest common divisor (GCD) using Euclid's algorithm
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Function to find the order r such that x**r % N == 1
def find_order(x, N):
    r = 1
    while True:
        if pow(x, r, N) == 1:
            return r
        r += 1

# Simulates the quantum part of Shor's algorithm classically (probabilistic)
def quantum_part(N):
    x = random.randint(2, N - 1)
    if gcd(x, N) > 1:
        return gcd(x, N)  # Found a factor immediately
    try:
        r = find_order(x, N)
        if r % 2 != 0:
            return N #Order is odd, algorithm fails in this simplified simulation
        a = pow(x, r // 2, N)
        if a == 1 or a == N -1:
            return N #Algorithm fails in this simplified simulation
        return gcd(a - 1, N)
    except Exception as e:
        print(f"Error in quantum_part: {e}")
        return N


def shors_algorithm(N):
    if N <= 1 or N % 2 == 0 or int(round(N**0.5))**2 == N:
        return N  # Handle trivial cases
    return quantum_part(N)

# Test cases
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
