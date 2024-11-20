import random
from utils import TreeNode

score_lookup_table = [
    [1, [-1, 0]],
    [1, [0, -1]],
    [10, [0, -1, 0]],
    [10, [-1, -1, 0]],
    [10, [0, -1, -1]],
    # [20, [0, -1, 0, -1]],
    # [20, [-1, 0, -1, 0]],
    [50, [0, -1, -1, 0]],
    # [50, [0, -1, 0, -1, 0]],
    [50, [-1, -1, -1, 0]],
    [50, [-1, -1, 0, -1, 0]],
    [50, [-1, 0, -1, -1, 0]],
    [50, [0, -1, -1, -1]],
    [50, [0, -1, 0, -1, -1]],
    [50, [0, -1, -1, 0, -1]],
    [100, [0, -1, -1, 0, -1, 0]],
    [100, [0, -1, 0, -1, -1, 0]],
    [200, [0, -1, -1, -1, 0]],
    [200, [-1, -1, -1, -1, 0]],
    [200, [-1, -1, -1, 0, -1]],
    [200, [-1, -1, 0, -1, -1]],
    [200, [-1, 0, -1, -1, -1]],
    [200, [0, -1, -1, -1, -1]],
    [1000, [0, -1, -1, -1, -1, 0]],
    [1000000, [-1, -1, -1, -1, -1]],
]

score_lookup_table_negative = []
for rule in score_lookup_table:
    score_lookup_table_negative.append([rule[0], [-p for p in rule[1]]])

N = 15
WHITE = -1
BLACK = 1
weight = 10


def distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def get_total_score(board):
    score_board_white = [[[0 for _ in range(4)] for _ in range(N)] for _ in range(N)]
    score_board_black = [[[0 for _ in range(4)] for _ in range(N)] for _ in range(N)]
    for i, row in enumerate(board):
        for j, elem in enumerate(row):
            for k, rule in enumerate(score_lookup_table):
                pattern = rule[1]
                # 横向
                for l, p in enumerate(pattern):
                    if (j + l < N and row[j + l] != p) or j + l >= N:
                        break
                else:
                    score_board_white[i][j][0] = max(score_board_white[i][j][0], rule[0])
                # 纵向
                for l, p in enumerate(pattern):
                    if (i + l < N and board[i + l][j] != p) or i + l >= N:
                        break
                else:
                    score_board_white[i][j][1] = max(score_board_white[i][j][1], rule[0])
                # 主对角线
                for l, p in enumerate(pattern):
                    if (i + l < N and j + l < N and board[i + l][j + l] != p) or i + l >= N or j + l >= N:
                        break
                else:
                    score_board_white[i][j][2] = max(score_board_white[i][j][2], rule[0])
                # 副对角线
                for l, p in enumerate(pattern):
                    if (i - l >= 0 and j + l < N and board[i - l][j + l] != p) or i - l < 0 or j + l >= N:
                        break
                else:
                    score_board_white[i][j][3] = max(score_board_white[i][j][3], rule[0])
            for k, rule in enumerate(score_lookup_table_negative):
                pattern = rule[1]
                # 横向
                for l, p in enumerate(pattern):
                    if (j + l < N and row[j + l] != p) or j + l >= N:
                        break
                else:
                    score_board_black[i][j][0] = max(score_board_black[i][j][0], rule[0])
                # 纵向
                for l, p in enumerate(pattern):
                    if (i + l < N and board[i + l][j] != p) or i + l >= N:
                        break
                else:
                    score_board_black[i][j][1] = max(score_board_black[i][j][1], rule[0])
                # 主对角线
                for l, p in enumerate(pattern):
                    if (i + l < N and j + l < N and board[i + l][j + l] != p) or i + l >= N or j + l >= N:
                        break
                else:
                    score_board_black[i][j][2] = max(score_board_black[i][j][2], rule[0])
                # 副对角线
                for l, p in enumerate(pattern):
                    if (i - l >= 0 and j + l < N and board[i - l][j + l] != p) or i - l < 0 or j + l >= N:
                        break
                else:
                    score_board_black[i][j][3] = max(score_board_black[i][j][3], rule[0])
    # 计算总分
    score_white = 0
    for i in score_board_white:
        for j in i:
            score_white = max(score_white, max(j))
    score_black = 0
    for i in score_board_black:
        for j in i:
            score_black = max(score_black, max(j))
    total_score = score_white - score_black * weight
    return total_score


def judge(board, depth, x, y):
    score = get_total_score(board)
    root = TreeNode(score, x, y, BLACK, 0)
    coordinates = generator(board, x, y)
    if len(coordinates) == 1:
        return coordinates[0][0], coordinates[0][1]
    search(root, board, coordinates, depth)
    pos_x, pos_y = x, y
    select = []
    for child in root.children:
        if root.score == child.score:
            select.append(child)
    if len(select) == 1:
        pos_x, pos_y = select[0].pos_x, select[0].pos_y
    else:
        select.sort(key=lambda node: distance(node.pos_x, node.pos_y, x, y))
        if distance(select[0].pos_x, select[0].pos_y, x, y) < distance(select[1].pos_x, select[1].pos_y, x, y):
            pos_x, pos_y = select[0].pos_x, select[0].pos_y
        else:
            new_select = []
            d = distance(select[0].pos_x, select[0].pos_y, x, y)
            for i in range(len(select)):
                if distance(select[i].pos_x, select[i].pos_y, x, y) == d:
                    new_select.append(select[i])
            index = random.randint(0, len(new_select) - 1)
            pos_x, pos_y = new_select[index].pos_x, new_select[index].pos_y
    return pos_x, pos_y


def generator(board, x, y):
    count = 0
    vectors = [
        [1, 0],
        [0, 1],
        [1, -1],
        [1, 1],
    ]
    new_x = x
    new_y = y
    result = []
    for direction in vectors:
        while 0 <= new_x < N and 0 <= new_y < N and board[new_y][new_x] == BLACK:
            new_x += direction[0]
            new_y += direction[1]
            count += 1
        if 0 <= new_x < N and 0 <= new_y < N and board[new_y][new_x] == 0:
            result.append([new_y, new_x])
        new_x = x
        new_y = y
        while 0 <= new_x < N and 0 <= new_y < N and board[new_y][new_x] == BLACK:
            new_x -= direction[0]
            new_y -= direction[1]
            count += 1
        count -= 1
        if 0 <= new_x < N and 0 <= new_y < N and board[new_y][new_x] == 0:
            result.append([new_y, new_x])
        if (count == 3 and len(result) == 2) or (count == 4 and len(result) >= 1):
            return result
        else:
            result = []
    result = [[i, j] for i in range(N) for j in range(N) if board[i][j] == 0]
    result.sort(key=lambda pos: distance(pos[0], pos[1], x, y))
    return result


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
    for i, j in coordinates:
        if is_visited(root, j, i):
            continue
        new_node = TreeNode(None, j, i, piece_type, root.depth + 1)
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
