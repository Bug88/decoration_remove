__author__ = 'liuzhen'

# -*- coding: utf-8 -*-

import os
# select the images to construct the database
dir_base = '../chinese/'
dir = '../valuable_chinese/'

para = open(dir+'0.rtf', 'r').read().decode("utf-8")

filenames = [ ls for ls in os.listdir(dir) if ls.split('.')[1] == 'txt' ]
character = []
for file in filenames:
    fp = open(dir+file, 'r')
    charac = fp.read().decode("utf-8")
    fp.close()

    charac = charac.split(para)

    character.extend(charac)

# for i in range(len(character)):
#     q = character[i] + '.png'
#     os.system('mv' + '../chinese/'+q + ' ' + '../chinese_sub')
    #os.system('mv '+ dir_base + character[i] + '.png' + ' ' + '../chinese_sub/')


import pygame

background = [(0, 0, 0), (128, 128, 128), (128, 128, 128), (255, 255, 255)]
foreground = [(255, 255, 255), (255, 255, 255), (0, 0, 0), (0, 0, 0)]

chinese_dir = '../chinese'

if not os.path.exists(chinese_dir):

    os.mkdir(chinese_dir)

pygame.init()

start, end = (0x4E00, 0x9FA5)

for codepoint in range(int(start), int(end)):

    word = unichr(codepoint)

    if word not in character:
        continue

    font = pygame.font.Font("../font_image/yahei.ttf", 300)

    for i in range(len(background)):
        rtext = font.render(word, True, foreground[i], background[i])

        pygame.image.save(rtext, os.path.join(chinese_dir, word + '_' + str(i) + ".png"))
