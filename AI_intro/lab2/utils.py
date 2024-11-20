import pygame as p


class Image:
    def __init__(self, filename: str, width: int, height: int, x: int = 0, y: int = 0):
        self.filename = filename
        self.width = width
        self.height = height
        self.image = p.image.load(filename)
        self.image = p.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def blit(self, window):
        window.blit(self.image, self.rect)

    def set_position(self, x, y):
        self.rect.x = x
        self.rect.y = y

    def center(self, x, y):
        self.rect.center = (x, y)

    def scale(self, width, height):
        self.width = width
        self.height = height
        self.image = p.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect()


class Text:
    def __init__(self, font, content: str, x: int = 0, y: int = 0, color=(0, 0, 0)):
        self.font = font
        self.content = content
        self.text = font.render(self.content, True, color)
        self.rect = self.text.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.color = color

    def blit(self, window):
        window.blit(self.text, self.rect)

    def set_position(self, x, y):
        self.rect.x = x
        self.rect.y = y

    def center(self, x, y):
        self.rect.center = (x, y)


class Button:
    def __init__(self, width: int, height: int, color=(255, 255, 255), x: int = 0, y: int = 0,
                 background_image: Image = None, text: Text = None):
        self.width = width
        self.height = height
        self.color = color
        background_image.scale(background_image, (width, height)) if background_image else None
        self.background_image = background_image
        self.text = text
        self.rect = p.Rect(x, y, width, height)
        self.image = p.Surface((width, height))
        self.image.fill(color)

    def blit(self, window):
        window.blit(self.image, self.rect)
        if self.background_image:
            window.blit(self.background_image, self.rect)
        if self.text:
            self.text.blit(window)

    def set_position(self, x, y):
        self.rect.x = x
        self.rect.y = y

    def center(self, x, y):
        self.rect.center = (x, y)
        if self.text:
            self.text.center(x, y)
        if self.background_image:
            self.background_image.center(x, y)

    def onclick(self, event, window, callback):
        mouse_x, mouse_y = p.mouse.get_pos()
        if (self.rect.x <= mouse_x <= self.rect.x + self.rect.width and
                self.rect.y <= mouse_y <= self.rect.y + self.rect.height):
            if event.type == p.MOUSEBUTTONDOWN:
                if self.text:
                    text = Text(self.text.font, self.text.content, self.text.rect.x, self.text.rect.y, color=(0, 125, 0))
                    text.blit(window)
            elif event.type == p.MOUSEBUTTONUP:
                self.text.blit(window)
                callback()


class TreeNode:
    def __init__(self, score, pos_x, pos_y, piece_type, depth, left=float('-inf'), right=float('inf')):
        self.score = score
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.piece_type = piece_type
        self.depth = depth
        self.left = left
        self.right = right
        self.children = []
        self.parent = None

    def add_child(self, child):
        self.children.append(child)
        child.parent = self
        child.left = self.left
        child.right = self.right

    def update(self, score):
        if self.piece_type == -1:
            self.right = min(self.right, score)
            if self.parent and self.right <= self.parent.left:
                self.score = self.right
                return True
        elif self.piece_type == 1:
            self.left = max(self.left, score)
            if self.parent and self.left >= self.parent.right:
                self.score = self.left
                return True
        return False
