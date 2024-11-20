import random

N = 9


class Generator:
    def __init__(self):
        self.board = [[0 for _ in range(N)] for _ in range(N)]

    def _is_valid(self, row, col, num) -> bool:
        for i in range(N):
            if self.board[row][i] == num:
                return False
            if self.board[i][col] == num:
                return False
        box_row = row // 3 * 3
        box_col = col // 3 * 3
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if self.board[i][j] == num:
                    return False
        return True

    def _solve(self) -> bool:
        for row in range(N):
            for col in range(N):
                if self.board[row][col] == 0:
                    for num in range(1, N + 1):
                        if self._is_valid(row, col, num):
                            self.board[row][col] = num
                            if self._solve():
                                return True
                            self.board[row][col] = 0
                    return False
        return True

    def _generate_board(self) -> bool:
        for i in range(N):
            for j in range(N):
                numbers = [n for n in range(1, N + 1)]
                random.shuffle(numbers)
                for num in numbers:
                    if self._is_valid(i, j, num):
                        self.board[i][j] = num
                        if self._solve():
                            return True
                        self.board[i][j] = 0
        return False

    def print_board(self):
        result = "["
        for i in range(N):
            result += '['
            for j in range(N):
                if self.board[i][j] == 0:
                    result += '_'
                else:
                    result += str(self.board[i][j])
                if j != N - 1:
                    result += ','
            if i != N - 1:
                result += '],'
        result += ']]'
        print(result)

    def generate_question(self):
        self._generate_board()
        mask_rate = random.randint(5, 10)
        for _ in range(mask_rate):
            row = random.randint(0, N - 1)
            col = random.randint(0, N - 1)
            while self.board[row][col] == 0:
                row = random.randint(0, N - 1)
                col = random.randint(0, N - 1)
            self.board[row][col] = 0


def main():
    g = Generator()
    g.generate_question()
    g.print_board()


if __name__ == '__main__':
    main()