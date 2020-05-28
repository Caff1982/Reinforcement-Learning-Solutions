import pygame
import sys

WHITE = (255,255,255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
CYAN = (0, 255, 255)
DARKGRAY = (70, 70, 70)

pygame.init()

from racetracks import *

class GUI:
    def __init__(self, track):
        self.track = track

        self.rows = len(track[0])
        self.cols = len(track)
        self.tile_size = 20
        self.height = self.cols * self.tile_size
        self.width = self.rows * self.tile_size

        self.screen = pygame.display.set_mode((self.width, self.height))
        self.surface = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()

        self.running = True

    def draw_square(self, x, y, cell_type):
        pygame.draw.rect(self.screen, WHITE, (x, y, self.tile_size,
                                              self.tile_size))

    def update(self, sequence):
        self.screen.fill(BLACK)
        for x in range(len(self.track[0])):
            for y in range(len(self.track)):
                cell_type = self.track[y][x]
                if cell_type == 0:
                    pygame.draw.rect(self.screen, WHITE,
                                    (x*self.tile_size, y*self.tile_size,
                                    self.tile_size, self.tile_size))
                elif cell_type == 1:
                    pygame.draw.rect(self.screen, RED,
                                    (x*self.tile_size, y*self.tile_size,
                                    self.tile_size, self.tile_size))
                elif cell_type == 2:
                    pygame.draw.rect(self.screen, GREEN,
                                    (x*self.tile_size, y*self.tile_size,
                                    self.tile_size, self.tile_size))
                elif cell_type == 3:
                    pygame.draw.rect(self.screen, DARKGRAY,
                                    (x*self.tile_size, y*self.tile_size,
                                    self.tile_size, self.tile_size))
        for cell in sequence:
            pygame.draw.rect(self.screen, CYAN,
                            (cell[0]*self.tile_size, cell[1]*self.tile_size,
                            self.tile_size, self.tile_size))
        pygame.display.update()

    def plot_sequence(self, sequence):
        path = [(step[0][1], step[0][0]) for step in sequence]
        idx = 1
        while self.running: 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False

            self.update(path[:idx])
            if idx < len(path):
                idx += 1
            self.clock.tick(2)

        pygame.quit()
        sys.exit()


