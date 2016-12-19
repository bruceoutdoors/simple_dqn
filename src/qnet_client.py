import logging

logging.basicConfig(format='%(asctime)s %(message)s')

from environment import ALEEnvironment
import random
import argparse
import requests
import cv2

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

env = ALEEnvironment("/home/bruceoutdoors/Documents/simple_dqn/roms/breakout.bin", args)
env.setMode('test')

if args.random_seed:
    random.seed(args.random_seed)

payload = {
    'inputCount' : env.numActions(),
    'gameName' : 'space_invaders',
}

address = 'http://127.0.0.1:5000/'
r = requests.post(address, data=payload)
assert r.json()['reply'] == 'ok'

reward = 0

while True:
    cv2.imwrite("capture.jpg", env.getScreen())
    files = {'screen': open("capture.jpg", 'rb')}

    payload = {
        'reward':reward,
        'isTerminal':env.isTerminal()
    }

    r = requests.post(address, files=files, data=payload)

    if env.isTerminal():
        break

    j = r.json()
    action = j['action']
    reward = env.act(action)


raw_input()