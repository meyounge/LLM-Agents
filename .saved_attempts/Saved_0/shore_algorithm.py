import random
import math
def shore_algorithm(N):
    if N <= 1:
        return 1
    
    if N % 2 == 0:
        return 2
    
    for a in range(2, int(math.sqrt(N))+1):
        if a**2 % N == 0:
            return a
        
    x = random.randint(2, N-1)
    while math.gcd(x, N) != 1:
        x = random.randint(2, N-1)
        
    r = 0
    while x**(r+1) % N != 1:
        r += 1
        
    if r % 2 == 0:
        return math.gcd(x**(r//2), N)
    
    return N
