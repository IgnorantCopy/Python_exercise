import random
from utils import TreeNode

score_lookup_table = [
    [1, [-1, 0]],
    [1, [0, -1]],
    [10, [0, -1, 0]],
    [10, [-1, -1, 0]],
    [10, [0, -1, -1]],
    [25, [0, -1, -1, 0]],
    [25, [-1, -1, -1, 0]],
    [25, [-1, -1, 0, -1, 0]],
    [25, [-1, 0, -1, -1, 0]],
    [25, [0, -1, -1, -1]],
    [25, [0, -1, 0, -1, -1]],
    [25, [0, -1, -1, 0, -1]],
    [100, [0, -1, -1, 0, -1, 0]],
    [100, [0, -1, 0, -1, -1, 0]],
    [500, [0, -1, -1, -1, 0]],
    [500, [-1, -1, -1, -1, 0]],
    [500, [-1, -1, -1, 0, -1]],
    [500, [-1, -1, 0, -1, -1]],
    [500, [-1, 0, -1, -1, -1]],
    [500, [0, -1, -1, -1, -1]],
    [5000, [0, -1, -1, -1, -1, 0]],
    [50000, [-1, -1, -1, -1, -1]],
]

N = 15
WHITE = -1
BLACK = 1
weight = 20
win = False
win_pos = []

score_lookup_table_negative = []
for index, rule in enumerate(score_lookup_table):
    score_lookup_table_negative.append([rule[0], [-p for p in rule[1]]])

def distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def sort_coordinates(coordinates, x, y, x1, y1, x2, y2):
    if x1 == -1 and y1 == -1 and x2 == -1 and y2 == -1:
        coordinates.sort(key=lambda pos: distance(pos[0], pos[1], x, y))
    elif x1 == -1 and y1 == -1:
        coordinates.sort(key=lambda pos: distance(pos[0], pos[1], x2, y2))
    elif x2 == -1 and y2 == -1:
        coordinates.sort(key=lambda pos: distance(pos[0], pos[1], x1, y1))
    else:
        coordinates.sort(key=lambda pos: min(distance(pos[0], pos[1], x1, y1), distance(pos[0], pos[1], x2, y2)))


def update_score(flag, score_board, board, i, j, row):
    if flag == WHITE:
        lookup_table = score_lookup_table
    else:
        lookup_table = score_lookup_table_negative
    for k, rule in enumerate(lookup_table):
        pattern = rule[1]
        # 横向
        for l, p in enumerate(pattern):
            if (j + l < N and row[j + l] != p) or j + l >= N:
                break
        else:
            score_board[i][j][0] = max(score_board[i][j][0], rule[0])
        for l, p in enumerate(pattern):
            if (j - l >= 0 and row[j - l] != p) or j - l < 0:
                break
        else:
            score_board[i][j][0] = max(score_board[i][j][0], rule[0])
        # 纵向
        for l, p in enumerate(pattern):
            if (i + l < N and board[i + l][j] != p) or i + l >= N:
                break
        else:
            score_board[i][j][1] = max(score_board[i][j][1], rule[0])
        for l, p in enumerate(pattern):
            if (i - l >= 0 and board[i - l][j] != p) or i - l < 0:
                break
        else:
            score_board[i][j][1] = max(score_board[i][j][1], rule[0])
        # 主对角线
        for l, p in enumerate(pattern):
            if (i + l < N and j + l < N and board[i + l][j + l] != p) or i + l >= N or j + l >= N:
                break
        else:
            score_board[i][j][2] = max(score_board[i][j][2], rule[0])
        for l, p in enumerate(pattern):
            if (i - l >= 0 and j - l >= 0 and board[i - l][j - l] != p) or i - l < 0 or j - l < 0:
                break
        else:
            score_board[i][j][2] = max(score_board[i][j][2], rule[0])
        # 副对角线
        for l, p in enumerate(pattern):
            if (i - l >= 0 and j + l < N and board[i - l][j + l] != p) or i - l < 0 or j + l >= N:
                break
        else:
            score_board[i][j][3] = max(score_board[i][j][3], rule[0])
        for l, p in enumerate(pattern):
            if (i + l < N and j - l >= 0 and board[i + l][j - l] != p) or i + l >= N or j - l < 0:
                break
        else:
            score_board[i][j][3] = max(score_board[i][j][3], rule[0])


def get_single_score(score: list):
    result1 = 0
    result2 = 0
    for i in score:
        if i <= 10:
            result1 += i
        else:
            if result2 == 0:
                result2 += i
            else:
                result2 *= i * 10
    return result1 + result2


def get_total_score(board):
    score_board_white = [[[0 for _ in range(4)] for _ in range(N)] for _ in range(N)]
    score_board_black = [[[0 for _ in range(4)] for _ in range(N)] for _ in range(N)]
    for i, row in enumerate(board):
        for j, elem in enumerate(row):
            update_score(WHITE, score_board_white, board, i, j, row)
            update_score(BLACK, score_board_black, board, i, j, row)
    # 计算总分
    score_white = 0
    for i in score_board_white:
        for j in i:
            score = get_single_score(j)
            score_white = max(score_white, score)
    score_black = 0
    for i in score_board_black:
        for j in i:
            score = get_single_score(j)
            score_black = max(score_black, score)
    total_score = score_white - score_black * weight
    return total_score


def print_board(board):
    for row in board:
        print(row)


def judge(board, depth, x, y):
    global win, win_pos
    if win and board[win_pos[0][1]][win_pos[0][0]] == 0:
        return win_pos[0]
    else:
        win = False
        win_pos = []

    score = get_total_score(board)
    print(score)
    print_board(board)
    root = TreeNode(score, x, y, BLACK, 0)
    coordinates, x1, y1, x2, y2, count = generator(board, x, y)
    if count >= 4 and ((x1 != -1 and y1 != -1) or (x2 != -1 and y2 != -1)) and not win:
        return coordinates[0]
    if len(coordinates) == 1:
        return coordinates[0]
    search(root, board, coordinates, depth)
    select = []
    for child in root.children:
        if root.score == child.score:
            select.append(child)
    if len(select) == 1:
        pos_x, pos_y = select[0].pos_x, select[0].pos_y
    else:
        temp = [[child.pos_x, child.pos_y] for child in select]
        sort_coordinates(temp, x, y, x1, y1, x2, y2)
        pos_x, pos_y = temp[0][0], temp[0][1]
    print(select[0].score)
    board[pos_y][pos_x] = WHITE
    win_pos, x1, y1, x2, y2, count = generator(board, pos_x, pos_y, WHITE)
    if count >= 4 and ((x1 != -1 and y1 != -1) or (x2 != -1 and y2 != -1)) and not win:
        win = True
    return pos_x, pos_y


def generator(board, x, y, flag=BLACK):
    max_count = 0
    max_level = 0
    vectors = [
        [1, 0],
        [0, 1],
        [1, -1],
        [1, 1],
    ]
    x1 = -1
    y1 = -1
    x2 = -1
    y2 = -1
    for direction in vectors:
        new_x = x
        new_y = y
        count = 0
        level = 0
        temp_x1 = -1
        temp_y1 = -1
        temp_x2 = -1
        temp_y2 = -1
        while 0 <= new_x < N and 0 <= new_y < N and board[new_y][new_x] == flag:
            new_x += direction[0]
            new_y += direction[1]
            count += 1
        if 0 <= new_x < N and 0 <= new_y < N and not board[new_y][new_x]:
            temp_x1, temp_y1 = new_x, new_y
            level += 1
        new_x = x - direction[0]
        new_y = y - direction[1]
        while 0 <= new_x < N and 0 <= new_y < N and board[new_y][new_x] == flag:
            count += 1
            new_x -= direction[0]
            new_y -= direction[1]
        if 0 <= new_x < N and 0 <= new_y < N and not board[new_y][new_x]:
            temp_x2, temp_y2 = new_x, new_y
            level += 1
        if count > max_count or (count == max_count and level > max_level):
            max_count = count
            max_level = level
            x1, y1, x2, y2 = temp_x1, temp_y1, temp_x2, temp_y2
    result = [[i, j] for i in range(N) for j in range(N) if board[j][i] == 0]
    if max_count < 2:
        x1, y1, x2, y2 = -1, -1, -1, -1
    sort_coordinates(result, x, y, x1, y1, x2, y2)
    return result, x1, y1, x2, y2, max_count


def get_new_board(child: TreeNode, board):
    new_board = [[i for i in r] for r in board]
    while child.parent:
        new_board[child.pos_y][child.pos_x] = child.piece_type
        child = child.parent
    return new_board


def is_visited(root: TreeNode, x, y):
    while root.parent:
        if root.pos_x == x and root.pos_y == y:
            return True
        root = root.parent
    return False


def search(root: TreeNode, board, coordinates, depth):
    piece_type = -root.piece_type
    for x, y in coordinates:
        if is_visited(root, x, y):
            continue
        new_node = TreeNode(None, x, y, piece_type, root.depth + 1, root.left, root.right)
        root.add_child(new_node)
        if new_node.depth == depth:
            new_board = get_new_board(new_node, board)
            new_node.score = get_total_score(new_board)
            new_node.left = new_node.score
            new_node.right = new_node.score
            if root.update(new_node.score):
                return False
        elif search(new_node, board, coordinates, depth):
            return False
    if root.score and root.depth != 0:
        return root.parent.update(root.score)
    if root.piece_type == BLACK:
        root.score = max([child.score for child in root.children])
    else:
        root.score = min([child.score for child in root.children])
    if root.parent:
        return root.parent.update(root.score)


if __name__ == '__main__':
    board = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 1, 1, -1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 1, -1, 1, -1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, -1, -1, 1, 1, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 1, 1, 1, 1, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, -1, -1, -1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    print(get_total_score(board))