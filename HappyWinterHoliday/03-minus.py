list1 = input().split(' ')
result = 0
list2 = []
for i in range(len(list1)):
    list2.append(int(list1[i]))
for i in range(len(list2)):
    for j in range(i + 1, len(list2)):
        result = list2[j] - list2[i] if list2[j] - list2[i] > result else result
print(result if result != 0 else -1)
