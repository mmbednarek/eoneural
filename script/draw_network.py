#!/usr/bin/env python3

import pandas as pd
import pygame
import sys
import math
import time
from pygame.locals import *

BLUE = (63, 159, 199, 255)

WIDTH=1200
HEIGHT=1200
OFFSET=100
LEFT=120

def draw_neuron(surf, x, y):
    pygame.draw.ellipse(surf, BLUE, (x, y, 60, 60))


def clamp(num, mi, ma):
    return max(min(num, ma), mi)


def weight_color(w):
    value = clamp(abs(w) * 20, 0, 255)
    if w < 0:
        return value, 0, 0, 255
    return value, value, value, 255


def error_color(e):
    evalue = clamp(math.log10(1e4 * abs(e)) * 51, 0, 255)
    if e < 0:
        return 0, 0, evalue, 255
    return 0, evalue, 0, 255


def draw_network(surf, row, layers, weight_count):
    x = LEFT
    pl = -1
    wi = 2
    for l in layers:
        y = HEIGHT/2 - OFFSET * (l / 2)
        for _ in range(l):
            if pl != -1:
                pygame.draw.line(surf, weight_color(row[wi]), (x + 30 - 4, y - 80), (x + 30 - 4, y + 30), 3)
                pygame.draw.line(surf, error_color(row[weight_count + wi]), (x + 30 + 4, y - 80), (x + 30 + 4, y + 30))
                wi += 1
                for pn in range(pl):
                    py = HEIGHT/2 - OFFSET * (pl / 2) + pn * OFFSET
                    pygame.draw.line(surf, weight_color(row[wi]), (x - 160 + 30, py + 30 - 4), (x + 30, y + 30 - 4), 3)
                    pygame.draw.line(surf, error_color(row[weight_count + wi]), (x - 160 + 30, py + 30 + 4),
                                     (x + 30, y + 30 + 4))
                    wi += 1
            y += OFFSET
        pl = l
        x += 160

    x = LEFT
    for l in layers:
        y = HEIGHT/2 - OFFSET * (l / 2)
        for _ in range(l):
            draw_neuron(surf, x, y)
            y += OFFSET
        x += 160

def calculate_weight_count(weights):
    result = 0
    prev = weights[0]
    for count in weights[1:]:
        result += (prev+1) * count
        prev = count
    return result

def main():
    pygame.init()
    surface = pygame.display.set_mode((WIDTH, HEIGHT))

    if len(sys.argv) < 4:
        print("invalid number of arguments")
        return

    input = sys.argv[1]
    layers = []
    for w in sys.argv[2:]:
        layers.append(int(w))

    weight_count = calculate_weight_count(layers)

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
                row = next(it)

                surface.fill((0, 0, 0, 255))
                img = font.render('epoch: {} iteration: {}'.format(row[1][0], row[1][1]), True, BLUE)
                surface.blit(img, (20, 20))
                draw_network(surface, row[1], layers, weight_count)
            except StopIteration:
                stopped = True

        pygame.display.update()
        # time.sleep(0.0001)


if __name__ == '__main__':
    main()
