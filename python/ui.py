import pygame
import numpy as np

class UI:
    COLOR_SCREEN = (255, 255, 255)  # 白色
    COLOR_LINE = (0, 0, 0)  # 黑色
    COLOR_RECT = 'Red'  # 蓝色

    def __init__(self, width, height, n, title):
        self.width = width
        self.height = height
        self.n = n
        self.title = title
        self.clock = pygame.time.Clock()
        self.screen = None
        self.init_pygame()
        self.set_screen()
        self.font = pygame.font.SysFont(None, 18)

    def init_pygame(self):
        pygame.init()  # 必须最先 init

    def update(self):
        pygame.display.update()

    def tick(self, frame_rate):
        self.clock.tick(frame_rate)

    def check_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # 退出按钮
                pygame.quit()  # pygame.quit() 和 pygame.init() 相对
                return True  # 成功退出
        return False  # 尚未退出

    def set_screen(self):
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption(self.title)

    def draw_screen(self):
        self.screen.fill(self.COLOR_SCREEN)

    def draw_grid(self):
        w, h = self.screen.get_size()  # 获取窗口大小
        interval = int(min(w, h) / (self.n + 1))  # 计算间隔
        for i in range(0, self.n):  # 绘制横线和竖线
            pygame.draw.line(self.screen, self.COLOR_LINE, ((i + 1) * interval, interval),
                             ((i + 1) * interval, self.n * interval), 2)
            pygame.draw.line(self.screen, self.COLOR_LINE, (interval, (i + 1) * interval),
                             (self.n * interval, (i + 1) * interval), 2)

    def get_mouse_pos(self):           # 获取鼠标在哪一格
        x, y = pygame.mouse.get_pos()  # 获取鼠标位置
        w, h = self.screen.get_size()  # 获取窗口大小
        interval = int(min(w, h) / (self.n + 1))  # 计算间隔
        for i in range(0, self.n):
            for j in range(0, self.n):
                x_goal = (i + 1) * interval
                y_goal = (j + 1) * interval
                if abs(x - x_goal) < interval // 2 and abs(y - y_goal) < interval // 2:
                    return i, j
        return -1, -1

    def draw_mouse(self):
        i, j = self.get_mouse_pos()
        w, h = self.screen.get_size()  # 获取窗口大小
        interval = int(min(w, h) / (self.n + 1))  # 计算间隔
        x = (i + 1) * interval
        y = (j + 1) * interval
        if i != -1 and j != -1:
            pygame.draw.rect(self.screen, self.COLOR_RECT,(x - interval // 2, y - interval // 2, interval, interval), 2)

    def is_pressed(self):
        return pygame.mouse.get_pressed()[0]

    def draw_board(self, board):
        w, h = self.screen.get_size()  # 获取窗口大小
        interval = int(min(w, h) / (self.n + 1))  # 计算间隔
        for i in range(self.n):
            for j in range(self.n):
                y = (i + 1) * interval
                x = (j + 1) * interval
                if board[i, j] == 1:        # 这里是黑子
                    pygame.draw.circle(self.screen, 'black', (x, y), interval // 3)
                elif board[i, j] == 0:      # 这里是白子
                    pygame.draw.circle(self.screen, 'gray', (x, y), interval // 3)
                else:
                    pass

    def draw_search_info(self, search_info):
        w, h = self.screen.get_size()  # 获取窗口大小
        interval = int(min(w, h) / (self.n + 1))  # 计算间隔

        for i, j, win_rate, vis in search_info:
            y = (i + 1) * interval
            x = (j + 1) * interval
            # pygame.draw.circle(self.screen, 'green', (x, y), interval // 4)
            text_surface = self.font.render(format(win_rate, '.1f') + "/" + str(vis), True, 'Red')
            text_w, text_h = text_surface.get_size()
            # self.screen.blit(text_surface, (x - text_w // 2, y - text_h // 2))
            self.screen.blit(text_surface, (x, y))
