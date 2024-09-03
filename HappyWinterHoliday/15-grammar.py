def grammar(strings):
    list2 = []
    string = ""
    for i in strings:
        if i.endswith("lios"):
            list2.append(True)
            string += 'a'
        elif i.endswith("liala"):
            list2.append(False)
            string += 'a'
        elif i.endswith("etr"):
            list2.append(True)
            string += 'n'
        elif i.endswith("etra"):
            list2.append(False)
            string += 'n'
        elif i.endswith("initis"):
            list2.append(True)
            string += 'v'
        elif i.endswith("inites"):
            list2.append(False)
            string += 'v'
        else:
            return 0
    gender = list2[0]
    for i in list2:
        if i != gender:
            return 0
    num_of_nouns = string.count('n')
    index1 = string.rfind('a')
    index2 = string.rfind('n')
    index3 = string.rfind('v')
    if num_of_nouns != 1 or not ((index1 != -1 and index1 < index2) and
        (index1 != -1 and index3 != -1 and index1 < index3) and (index3 != -1 and index2 < index3)):
        return 0
    return 1


list1 = input().split(' ')
print("YES" if grammar(list1) else "NO")
