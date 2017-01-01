import logging

logging.basicConfig(format='%(asctime)s %(message)s')

from PLEEnvironment import PLEEnvironment
import random
import argparse
import sys
import cv2
import pygame, sys

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Environment')
envarg.add_argument("--environment", choices=["ale", "gym"], default="ale",
                    help="Whether to train agent using ALE or OpenAI Gym.")
envarg.add_argument("--display_screen", type=str2bool, default=False,
                    help="Display game screen during training and testing.")
# envarg.add_argument("--sound", type=str2bool, default=False, help="Play (or record) sound.")
envarg.add_argument("--frame_skip", type=int, default=4, help="How many times to repeat each chosen action.")
envarg.add_argument("--repeat_action_probability", type=float, default=0,
                    help="Probability, that chosen action will be repeated. Otherwise random action is chosen during repeating.")
envarg.add_argument("--minimal_action_set", dest="minimal_action_set", type=str2bool, default=True,
                    help="Use minimal action set.")
envarg.add_argument("--color_averaging", type=str2bool, default=True,
                    help="Perform color averaging with previous frame.")
envarg.add_argument("--screen_width", type=int, default=84, help="Screen width after resize.")
envarg.add_argument("--screen_height", type=int, default=84, help="Screen height after resize.")
envarg.add_argument("--record_screen_path",
                    help="Record game screens under this path. Subfolder for each game is created.")
envarg.add_argument("--record_sound_filename", help="Record game sound in this file.")


comarg = parser.add_argument_group('Common')
comarg.add_argument("--random_seed", default=9090, type=int, help="Random seed for repeatable experiments.")
args = parser.parse_args()

if args.random_seed:
    random.seed(args.random_seed)

import ple

game = ple.games.snake.Snake()
env = PLEEnvironment(game, args)
moves = env.numActions()

# Set env mode test so that loss of life is not considered as terminal
env.setMode('test')

while True:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                print env.act(0)
            elif event.key == pygame.K_w:
                print env.act(1)
            elif event.key == pygame.K_e:
                print env.act(2)
            elif event.key == pygame.K_r:
                print env.act(3)
            elif event.key == pygame.K_t:
                print env.act(4)
            print event.key
            cv2.imwrite("ple_capture.jpg", env.getScreen())

            if env.isTerminal():
                print "GAME OVER... restart!"
                env.restart()
        elif event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()




