def minus(str1, str2):
    list1 = str1.split(':')
    list2 = str2.split(':')
    num1 = int(list1[0]) * 3600 + int(list1[1]) * 60 + int(list1[2])
    list3 = list2[2].split(' ')
    num2 = 0
    if len(list3) == 1:
        num2 = int(list2[0]) * 3600 + int(list2[1]) * 60 + int(list3[0])
    elif len(list3) == 2:
        num2 = (int(list2[0]) + int(list3[1][2]) * 24) * 3600 + int(list2[1]) * 60 + int(list3[0])
    return num2 - num1


time1, time2 = map(str, input().split(' ', 1))
time3, time4 = map(str, input().split(' ', 1))
answer1 = minus(time1, time2)
answer2 = minus(time3, time4)
result = (answer1 + answer2) // 2
hours = result // 3600
result -= hours * 3600
minutes = result // 60
result -= minutes * 60
seconds = result
print("%02d:%02d:%02d" % (hours, minutes, seconds))
