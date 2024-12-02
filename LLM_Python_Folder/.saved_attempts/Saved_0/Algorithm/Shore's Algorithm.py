import random
import math

def Shore_Algorithm(N):
    if N <= 1:
        return 1
    if N % 2 == 0:
        return 2
    for a in range(2, N):
        if a**N % N == 1:
            return a
    for x in range(2, N):
        if math.gcd(x, N) != 1:
            return math.gcd(x, N)
    r = 0
    while True:
        x = random.randint(2, N-1)
        print(x)
        if math.gcd(x, N) != 1:
            return math.gcd(x, N)
        for i in range(1, N):
            if (x**i % N) == 1:
                print("cal")
                r = i
                break
    print("I was called")
    if r % 2 == 0:
        return (x**(r//2) + 1) // math.gcd(x**(r//2) + 1, N), (x**(r//2) - 1) // math.gcd(x**(r//2) - 1, N)
    return N
