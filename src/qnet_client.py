import logging

logging.basicConfig(format='%(asctime)s %(message)s')

from environment import ALEEnvironment
from PLEEnvironment import PLEEnvironment

import random
import argparse
import requests
import cv2
import ple

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Environment')
envarg.add_argument("--environment", choices=["ale", "gym"], default="ale",
                    help="Whether to train agent using ALE or OpenAI Gym.")
envarg.add_argument("--display_screen", type=str2bool, default=True,
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
comarg.add_argument("--random_seed", type=int, help="Random seed for repeatable experiments.")
comarg.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO",
                    help="Log level.")
args = parser.parse_args()

logger = logging.getLogger()
logger.setLevel(args.log_level)

bunch_of_env = [
    {
        'env' : 'ale',
        'gameName' : 'kangaroo',
        'rom' : "/home/danny/Documents/simple_dqn/roms/kangaroo.bin"
    },
    {
        'env' : 'ale',
        'gameName' : 'space_invaders',
        'rom' : "/home/danny/Documents/simple_dqn/roms/space_invaders.bin"
    },
    {
        'env' : 'ale',
        'gameName' : 'demon_attack',
        'rom' : "/home/danny/Documents/simple_dqn/roms/demon_attack.bin"
    },
    {
        'env' : 'ale',
        'gameName' : 'star_gunner',
        'rom' : "/home/danny/Documents/simple_dqn/roms/star_gunner.bin"
    },
    {
        'env': 'ale',
        'gameName': 'breakout',
        'rom': "/home/danny/Documents/simple_dqn/roms/breakout.bin"
    },
    {
        'env': 'ple',
        'gameName': 'catcher'
    },
    {
        'env': 'ple',
        'gameName': 'snake'
    }
]

# if args.random_seed:
#     random.seed(args.random_seed)

while True:
    rand_game = bunch_of_env[random.randint(0, len(bunch_of_env) - 1)]

    if rand_game['env'] == 'ale':
        env = ALEEnvironment(rand_game['rom'], args)
        env.setMode('test')
    elif rand_game['env'] == 'ple':
        if rand_game['gameName'] == 'snake':
            game = ple.games.snake.Snake()
        elif rand_game['gameName'] == 'catcher':
            game = ple.games.catcher.Catcher()
        env = PLEEnvironment(game, args)

    payload = {
        'inputCount': env.numActions(),
        'gameName': rand_game['gameName'],
    }

    address = 'http://172.17.0.1:5000/'
    r = requests.post(address, data=payload)
    assert r.json()['reply'] == 'ok'

    reward = 0
    zero_counter = 0

    while True:
        cv2.imwrite("capture.jpg", env.getScreen())
        files = {'screen': open("capture.jpg", 'rb')}

        payload = {
            'reward':int(reward), # PLE casts the reward to a float, so need to change to int
            'isTerminal':env.isTerminal()
        }

        r = requests.post(address, files=files, data=payload)

        j = r.json()
        action = j['action']
        reward = env.act(action)

        if reward == 0:
            zero_counter += 1
            if zero_counter >= 500:
                print 'You have been stuck WAY too long. DIE!!'
                r = requests.post(address, files=files, data=payload)
                break
        else:
            zero_counter = 0

        if env.isTerminal():
            print 'terminal state reached!'
            r = requests.post(address, files=files, data=payload)
            break


