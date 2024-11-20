import random
import time
from typing import Callable

answer = "123456780"


class Node:
    def __init__(self, num: str, score: int, depth: int = 0):
        self.num = num
        self.score = score
        self.depth = depth

    def get_children(self, eval_func: Callable):
        nodes = []
        vector = [-3, -1, 1, 3]
        index = self.num.find('0')
        for offset in vector:
            new_index = index + offset
            if 0 <= new_index < 9:
                new_num = [i for i in self.num]
                new_num[index] = new_num[new_index]
                new_num[new_index] = '0'
                new_num = ''.join(new_num)
                nodes.append(Node(new_num, eval_func(new_num) + self.depth + 1, self.depth + 1))
        return nodes


# distance evaluation function
def get_distance(num: str) -> int:
    result = 0
    for i, c in enumerate(num):
        if c == '0':
            continue
        index = answer.find(c)
        temp = abs(index - i)
        result += temp - temp // 3 * 2
    return result


# position evaluation function
def get_position_deviation(num: str) -> int:
    result = 0
    for i in range(len(num)):
        if num[i] != answer[i]:
            result += 1
    return result


# my evaluation function
def get_reverse_order(num: str) -> int:
    result = 0
    for i, c in enumerate(num):
        if c == '0':
            continue
        for j in range(0, i):
            if num[j] != '0' and int(num[j]) > int(num[i]):
                result += 1
    return result


def problem_generator() -> str:
    num = ''.join(random.sample(answer, 9))
    while get_reverse_order(num) % 2 != 0:
        num = ''.join(random.sample(num, 9))
    return num


def search(eval_func: Callable, patterns: list):
    visited = []
    count = 0
    while len(patterns) > 0:
        count += 1
        patterns_copy = [i.num for i in patterns]
        visited_copy = [i.num for i in visited]
        if count > 15000:
            return False
        node = patterns.pop(0)
        visited.append(node)
        if node.num == answer:
            return True
        nodes = node.get_children(eval_func)
        for i in nodes:
            if i.num not in visited_copy and i not in patterns_copy:
                patterns.append(i)
        patterns.sort(key=lambda x: x.score)
    return True


def main(epoch=100):
    total_time = [0, 0, 0]
    total_success = [0, 0, 0]
    eval_funcs = [get_distance, get_position_deviation, get_reverse_order]
    for _ in range(epoch):
        num = problem_generator()
        print(num)
        for i, eval_func in enumerate(eval_funcs):
            start_time = time.time()
            patterns = [Node(num, eval_func(num), 0)]
            is_success = search(eval_func, patterns)
            print(is_success)
            total_time[i] += time.time() - start_time
            if is_success:
                total_success[i] += 1
    return total_time, total_success


if __name__ == '__main__':
    time, success = main()
    print(f'''
                distance                position            reverse_order
    time:    {time[0] / 100}    {time[1] / 100}    {time[2] / 100}
    success:    {success[0]}%                       {success[1]}%                   {success[2]}%
    ''')