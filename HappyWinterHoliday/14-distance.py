import math

n = int(input())
list1 = input().split(' ')
students = 0
for i in range(n):
    students += int(list1[i])
result = math.ceil(students / 4)
print(result)
