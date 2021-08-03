import math


def arrangements(n, k):
    return math.factorial(n) // math.factorial(n - k)


def permutations(n):
    return math.factorial(n)


def combinations(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
