import pygame as p
import utils
import ai

WIDTH = 900
HEIGHT = 1000
WHITE = -1
BLACK = 1
N = 15
DEPTH = 2
p.init()
window = p.display.set_mode((WIDTH, HEIGHT))
p.display.set_caption("Gomoku")

font1 = p.font.SysFont('Curlz', 80)
clock = p.time.Clock()
background = utils.Image("./images/background.jpg", WIDTH, HEIGHT)
board_image = utils.Image("./images/board.png", 600, 600)
board_image.center(WIDTH / 2, HEIGHT / 2 + 100)
title = utils.Text(font1, "Gomoku")
title.center(WIDTH / 2, 80)

p.mixer.music.load("./audio/bgm.WAV")
p.mixer.music.play(-1)

board = [[0 for _ in range(N)] for _ in range(N)]
offset_x = 173
offset_y = 323
current_player_x = 800
current_player_y = 200
board_width = 39.5
epsilon = 15
is_black = 1
is_finish = False
step = 0

font2 = p.font.SysFont('Curlz', 50)
black_image = utils.Image("./images/black.png", epsilon * 2, epsilon * 2)
white_image = utils.Image("./images/white.png", epsilon * 2, epsilon * 2)
restart_text = utils.Text(font2, "Restart")
restart_button = utils.Button(160, 50, (0, 255, 0), text=restart_text)
restart_button.center(WIDTH / 4, 200)


def init():
    global is_black, is_finish, step
    is_black = 1
    is_finish = False
    step = 0
    for i in range(N):
        for j in range(N):
            board[i][j] = 0
    background.blit(window)
    board_image.blit(window)
    title.blit(window)
    black_image.set_position(current_player_x, current_player_y)
    black_image.blit(window)
    restart_button.blit(window)


def draw_piece(i, j):
    global is_black
    board[i][j] = BLACK if is_black else WHITE
    piece_x = offset_x + j * board_width
    piece_y = offset_y + i * board_width
    piece_image = black_image if is_black else white_image
    piece_image.center(piece_x, piece_y)
    piece_image.blit(window)
    is_black = 1 - is_black
    current_player_image = black_image if is_black else white_image
    current_player_image.set_position(current_player_x, current_player_y)
    current_player_image.blit(window)
    p.mixer.Sound("./audio/down.WAV").play()


def check_win(i, j):
    vectors = [
        [1, 0],
        [0, 1],
        [1, -1],
        [1, 1],
    ]
    for d in vectors:
        count = 1
        new_i = i
        new_j = j
        while True:
            new_i += d[0]
            new_j += d[1]
            if 0 <= new_i < N and 0 <= new_j < N and board[new_i][new_j] == board[i][j]:
                count += 1
            else:
                break
        new_i = i
        new_j = j
        while True:
            new_i -= d[0]
            new_j -= d[1]
            if 0 <= new_i < N and 0 <= new_j < N and board[new_i][new_j] == board[i][j]:
                count += 1
            else:
                break
        if count >= 5:
            return True
    return False


if __name__ == '__main__':
    init()
    while True:
        clock.tick(60)
        p.display.update()
        for event in p.event.get():
            restart_button.onclick(event, window, init)
            if event.type == p.QUIT:
                p.quit()
                exit()
            if event.type == p.MOUSEBUTTONUP:
                x, y = p.mouse.get_pos()
                if (offset_x - epsilon <= x <= offset_x + (N - 1) * board_width + epsilon and
                        offset_y - epsilon <= y <= offset_y + (N - 1) * board_width + epsilon) and not is_finish:
                    board_y = int((y - offset_y + epsilon) // board_width)
                    board_x = int((x - offset_x + epsilon) // board_width)
                    if board[board_y][board_x] == 0:
                        draw_piece(board_y, board_x)
                        step += 1
                        p.display.update()
                        is_finish = check_win(board_y, board_x)
                        if is_finish:
                            result = utils.Text(font2, "Black wins!" if not is_black else "White wins!")
                            result.center(3 * WIDTH / 5, 200)
                            result.blit(window)
                        elif not is_black:
                            coordinate = ai.judge(board, DEPTH, board_x, board_y)
                            draw_piece(coordinate[1], coordinate[0])
                            step += 1
                            p.display.update()
                            is_finish = check_win(coordinate[1], coordinate[0])
                            if is_finish:
                                result = utils.Text(font2, "Black wins!" if not is_black else "White wins!")
                                result.center(3 * WIDTH / 5, 200)
                                result.blit(window)