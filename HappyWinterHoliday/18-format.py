s = input()
k = int(input())
result = ""
counter = 0
for i in s[::-1]:
    if i == '-':
        continue
    counter += 1
    result += i.upper() if i.isalpha() else i
    if counter == k:
        counter = 0
        result += '-'
result = result[::-1]
print(result[1:] if result[0] == '-' else result)
