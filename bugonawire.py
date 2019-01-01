import csv
import random
import numpy as np
import pygame,sys
from pygame.locals import *
import time
from pynput.keyboard import Key, Controller

from sklearn.externals import joblib

'''
Left lane value, middle, right, cur_pos, key_pressed[0], [1](one-hot encoded)
'''
view, lanes = 500, 3
world = np.zeros((view, lanes), dtype=int)
dataset = []


def main():
    # load model
    clf = joblib.load('model')
    keyboard = Controller()
    t2 = 0
    pygame.init()
    vel = 3
    FPS = 30
    fpsClock = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((lanes*100, view))
    pygame.display.set_caption('Bug on a wire!')
    curr_lane = 1
    score = 0
    objects = [[0,0], [2,0], [1, int(view/2)], [0, int(view/2)]] #[lane, y]

    keys = [Key.left, Key.right]

    t = time.time() * 1000.0
    while True:  # main game loop
        DISPLAYSURF.fill((0,0,0))
        #curr_lane = random.randint(0,3)
        pygame.draw.circle(DISPLAYSURF, (255, 255, 255), (curr_lane*100 + 50, view-50), 50, 10)
        #min0 = min(a for a in objects[1] if objects[0] ==0 )

        for object in objects:
            object[1] = (object[1]+ int(vel))
            pygame.draw.circle(DISPLAYSURF, (255, 0,0), (object[0]*100+50, object[1]-50), 50, 10)
            if(object[1] > view+100):
                objects.remove(object)
                objects.append([random.randint(0,3), 0])
            if(object[1]> view-100):
                if object[0] == curr_lane:
                    print("Score:", score)
                    with open('new_dataset.csv', 'a') as myfile:
                        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                        for i in range(0, len(dataset) - 10):
                            wr.writerow(dataset[i])


                    pygame.quit()
                    sys.exit()

        t1 = time.time() * 1000.0

        left_obs = [obj[1] for obj in objects if obj[0] == 0]
        mid_obs = [obj[1] for obj in objects if obj[0] == 1]
        right_obs = [obj[1] for obj in objects if obj[0] == 2]

        left_obs.append(0)
        right_obs.append(0)
        mid_obs.append(0)

        left = max(left_obs)
        mid = max(mid_obs)
        right = max(right_obs)

        cur_list = [0, 0, 0]
        cur_list[curr_lane] = 1

        if (t1 - t) > 400:
            t = time.time() * 1000.0
            # dataset.append([left, mid, right, cur_list[0], cur_list[1], cur_list[2], 0, 0, 1])

        #######################Classifier steps

        pred = np.around(clf.predict([[left, mid, right, cur_list[0], cur_list[1], cur_list[2]]]))

        idx = cur_list.index(1)
        t3 = time.time() * 1000.0
        if pred[0][1] == 1 and t3 - t2 > 60:
            print('pressed key right')
            keyboard.press(Key.right)
            keyboard.release(Key.right)
            t2 = time.time() * 1000.0
        elif pred[0][0] == 1 and t3 - t2 > 60:
            print('pressed key left')
            keyboard.press(Key.left)
            keyboard.release(Key.left)
            t2 = time.time() * 1000.0
        #############################

        for event in pygame.event.get():
            if event.type == KEYDOWN:
                left_obs = [obj[1] for obj in objects if obj[0] == 0]
                mid_obs = [obj[1] for obj in objects if obj[0] == 1]
                right_obs = [obj[1] for obj in objects if obj[0] == 2]

                left_obs.append(0)
                right_obs.append(0)
                mid_obs.append(0)

                left = max(left_obs)
                mid = max(mid_obs)
                right = max(right_obs)

                cur_list = [0, 0, 0]
                cur_list[curr_lane] = 1

                if event.key == K_LEFT:
                    dataset.append([left, mid, right, cur_list[0], cur_list[1], cur_list[2], 1, 0, 0])
                    curr_lane = (curr_lane - 1)
                    if curr_lane < 0:
                        curr_lane = 0

                if event.key == K_RIGHT:
                    dataset.append([left, mid, right, cur_list[0], cur_list[1], cur_list[2], 0, 1, 0])
                    curr_lane = (curr_lane + 1)
                    if curr_lane > 2:
                        curr_lane = 2
            elif event.type == QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.update()
        fpsClock.tick(FPS)
        vel *= 1.001
        score += 1



main()