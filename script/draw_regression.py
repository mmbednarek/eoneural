#!/usr/bin/env python3

import pandas as pd
import pygame
import sys
from pygame.locals import *

RED = (200, 0, 0, 255)
GREEN = (20, 200, 20, 255)
BLUE = (20, 20, 200, 255)
YELLOW = (200, 20, 200, 255)


def add_point(surf, color, x, y):
    pygame.draw.ellipse(surf, color, (x, y, 6, 6))

def draw_board(surf, points):
    for (_, point) in points.iterrows():
        add_point(surf, GREEN, (point.x + 5) * 70, (point.y)*800)
        add_point(surf, BLUE, (point.x + 5) * 70, (point.ey)*800)


def main():
    pygame.init()
    surface = pygame.display.set_mode((800, 800))

    input = sys.argv[1]
    points = pd.read_csv(input)
    draw_board(surface, points)

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.update()

if __name__ == '__main__':
    main()
