arr = [2, 3, 1, 5, 4]

for i in range(len(arr) - 1):
    for j in range(len(arr) - 1 - i):
        if arr[j] < arr[j + 1]:
            arr[j], arr[j + 1] = arr[j + 1], arr[j]

print(arr)
print(arr[4])