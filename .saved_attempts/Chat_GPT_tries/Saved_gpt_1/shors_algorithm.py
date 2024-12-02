import random
import math

def shors_algorithm(N):
    #Step 0
    if N <= 1:
        return 1
    #Step 1
    if N % 2 == 0:
        return 2
    #Step 2
    for a in range(2, int(math.sqrt(N)) + 1):
        if N % a == 0:
            return a
    #Step 3
    x = random.randint(2, N-1)
    gcd = math.gcd(x, N)
    if gcd > 1:
        return gcd
    #Step 4
    r = 1
    while (x ** r) % N != 1:
        r += 1
    #Step 5
    if r % 2 == 0:
        y = (x ** (r // 2)) % N
        factor = math.gcd(y + 1, N)
        if factor > 1:
            return factor
        factor = math.gcd(y - 1, N)
        if factor > 1:
            return factor
    #Step 6
    return N

#Step 7
for i in range(5):
    print(shors_algorithm(125))