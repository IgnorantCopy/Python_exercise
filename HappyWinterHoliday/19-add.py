list1 = input().split(' ')
target = int(input())
for i in range(len(list1)):
    for j in range(i + 1, len(list1)):
        if int(list1[i]) + int(list1[j]) == target:
            print(i, j)
