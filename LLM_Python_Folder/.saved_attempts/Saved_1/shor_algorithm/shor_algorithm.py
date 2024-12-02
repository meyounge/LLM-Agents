import random; def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b);

def mod_exp(b, e, n):
    if e == 0:
        return 1
    if e % 2 == 0:
        return mod_exp(b * b % n, e // 2, n)
    return b * mod_exp(b, e - 1, n) % n;

def shor(N):
    if N <= 1:
        return 1
    if N % 2 == 0:
        return 2
    for a in range(2, N):
        if pow(a, N, N) == 1:
            break
    r = 0
    while pow(a, 2**r, N) != 1:
        r += 1
    if r % 2 == 0:
        return math.gcd(a**(r//2)+1, a**(r//2)-1, N)
    else:
        return N
