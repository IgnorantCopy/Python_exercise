n = int(input())
scores = input().split(' ')
num_of_students = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
strings = [" 0 -  9:", "10 - 19:", "20 - 29:", "30 - 39:", "40 - 49:",
           "50 - 59:", "60 - 69:", "70 - 79:", "80 - 89:", "90 -100:"]
for i in scores:
    score = int(i)
    if score == 100:
        score -= 1
    num_of_students[score // 10] += 1
for i in range(len(num_of_students)):
    if i != len(num_of_students) - 1:
        print(num_of_students[i], end=',')
    else:
        print(num_of_students[i])
for i in range(len(num_of_students)):
    print(strings[i] + num_of_students[i] * '*')
