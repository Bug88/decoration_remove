__author__ = 'liuzhen'



import os

import pygame


background = [(0, 0, 0), (128, 128, 128), (255, 255, 255)]
foregrund = [(255, 255, 255), (128, 128, 128), (0, 0, 0)]

chinese_dir = '../chinese'

if not os.path.exists(chinese_dir):

    os.mkdir(chinese_dir)



pygame.init()

start, end = (0x4E00, 0x9FA5)

for codepoint in range(int(start), int(end)):

    word = unichr(codepoint)

    font = pygame.font.Font("yahei.ttf", 300)

    for i in range(len(background)):
        rtext = font.render(word, True, background[i], foregrund[i])

        pygame.image.save(rtext, os.path.join(chinese_dir, word + ".png"))