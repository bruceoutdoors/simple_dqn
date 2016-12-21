from environment import Environment
import sys
import os
import logging
import cv2

logger = logging.getLogger(__name__)


class PLEEnvironment(Environment):
    def __init__(self, game, args):

        from ple import PLE

        self.ale = PLE(game, display_screen=False, frame_skip=4)

        #if args.display_screen:
        self.ale.display_screen = True

        self.ale.frame_skip = args.frame_skip

        self.actions = self.ale.getActionSet()

        self.screen_width = args.screen_width
        self.screen_height = args.screen_height

        self.life_lost = False
        self.ale.init()

    def numActions(self):
        return len(self.actions)

    def restart(self):
        # In test mode, the game is simply initialized. In train mode, if the game
        # is in terminal state due to a life loss but not yet game over, then only
        # life loss flag is reset so that the next game starts from the current
        # state. Otherwise, the game is simply initialized.
        if (
            self.mode == 'test' or
            not self.life_lost or  # `reset` called in a middle of episode
            self.ale.game_over()  # all lives are lost
        ):
            self.ale.reset_game()
        self.life_lost = False

    def act(self, action):
        lives = self.ale.lives()
        reward = self.ale.act(self.actions[action])
        self.life_lost = (not lives == self.ale.lives())
        return reward

    def getScreen(self):
        screen = self.ale.getScreenGrayscale()
        resized = cv2.resize(screen, (self.screen_width, self.screen_height))
        return resized

    def isTerminal(self):
        if self.mode == 'train':
            return self.ale.game_over() or self.life_lost
        return self.ale.game_over()
