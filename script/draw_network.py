#!/usr/bin/env python3

import pandas as pd
import pygame
import sys
import math
import time
from pygame.locals import *

BLUE = (63, 159, 199, 255)

def draw_neuron(surf, x, y):
    pygame.draw.ellipse(surf, BLUE, (x, y, 60, 60))

def clamp(num, mi, ma):
    return max(min(num, ma), mi)

def error_color(e):
    evalue = clamp(math.log10(1e4 * abs(e)) * 51, 0, 255)
    if e < 0:
        return (evalue, 0, 0, 255)
    return (0, evalue, 0, 255)

def draw_network(surf, row, layers, weight_count):
    x = 32
    pl = -1
    wi = 2
    for l in layers:
        offset = 200
        y = 400 - offset * (l/2)
        for _ in range(l):
            if pl != -1:
                for pn in range(pl):
                    py = 400 - offset * (pl/2) + pn*offset
                    value = clamp((10 + row[wi]) * 12.5, 0, 255)
                    pygame.draw.line(surf, (value, value, value, 255), (x - 160 + 30, py + 30 - 4), (x + 30, y + 30 - 4))
                    pygame.draw.line(surf, error_color(row[weight_count + wi]), (x - 160 + 30, py + 30 + 4), (x + 30, y + 30 + 4))
                    wi += 1
            y += offset
        pl = l
        x += 160

    x = 32
    for l in layers:
        offset = 200
        y = 400 - offset * (l/2)
        for _ in range(l):
            draw_neuron(surf, x, y)
            y += offset
        x += 160


def main():
    pygame.init()
    surface = pygame.display.set_mode((800, 800))

    input = sys.argv[1]
    points = pd.read_csv(input)
    it = points.iterrows()

    font = pygame.font.SysFont(None, 24)
    stopped = False

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
                

        if not stopped:
            try:
                surface.fill((0,0,0,255))
                row = next(it)
                img = font.render('epoch: {} iteration: {}'.format(row[1][0], row[1][1]), True, BLUE)
                surface.blit(img, (20, 20))

                draw_network(surface, row[1], [2, 2, 1], 12)
            except StopIteration:
                stopped = True
            
        pygame.display.update()
        time.sleep(0.01)

if __name__ == '__main__':
    main()
