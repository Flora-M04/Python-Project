'''def is_prime(n):
    if n <= 1:
        return False  # 0 and 1 are not prime
    for i in range(2, int(n**0.5) + 1): 
        if n % i == 0:
            return False
    return True

num = int(input("Enter a number: "))

if is_prime(num):
    print(num, "is a prime number.")
else:
    print(num, "is not a prime number.")

def factorial_below_50():
    num = int(input("Enter a number below 50: "))

    if num < 0:
        print("Factorial is not defined for negative numbers.")
    elif num >= 50:
        print("Number is too large! Please enter a number below 50.")
    else:
        factorial = 1
        for i in range(1, num + 1):
            factorial *= i
        print(f"The factorial of {num} is {factorial}")


factorial_below_50()'''

import numpy as np
arr = np.array([1,2,3])
a = arr.ndim

print(a)
