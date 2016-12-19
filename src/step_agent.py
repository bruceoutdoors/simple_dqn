import random
import logging
import numpy as np

logger = logging.getLogger(__name__)
from state_buffer import StateBuffer


class StepAgent:
    def __init__(self, num_actions, replay_memory, deep_q_network, args):
        self.mem = replay_memory
        self.net = deep_q_network
        self.buf = StateBuffer(args)
        self.random_starts = args.random_starts
        self.history_length = args.history_length

        self.exploration_rate_start = args.exploration_rate_start
        self.exploration_rate_end = args.exploration_rate_end
        self.exploration_decay_steps = args.exploration_decay_steps
        self.exploration_rate_test = args.exploration_rate_test
        self.total_train_steps = args.start_epoch * args.train_steps

        self.train_frequency = args.train_frequency
        self.train_repeat = args.train_repeat
        self.target_steps = args.target_steps

        self.num_actions = num_actions

        self.callback = None

        self.current_action = 0

    def _restartRandom(self):
        # self.env.restart()

        # perform random number of dummy actions to produce more stochastic games
        for i in xrange(random.randint(self.history_length, self.random_starts) + 1):
            reward = self.env.act(0)
            screen = self.env.getScreen()
            terminal = self.env.isTerminal()
            assert not terminal, "terminal state occurred during random initialization"
            # add dummy states to buffer
            self.buf.add(screen)

    def _explorationRate(self):
        # calculate decaying exploration rate
        if self.total_train_steps < self.exploration_decay_steps:
            return self.exploration_rate_start - self.total_train_steps * (
            self.exploration_rate_start - self.exploration_rate_end) / self.exploration_decay_steps
        else:
            return self.exploration_rate_end

    def step_play_receive(self, screen, reward, terminal):
        # print reward
        if reward <> 0:
            logger.debug("Reward: %d" % reward)

        # add screen to buffer
        self.buf.add(screen)

        # restart the game if over
        if terminal:
            logger.debug("Terminal state, restarting")
            return

        self.mem.add(self.current_action, reward, screen, terminal)

    def step_play_send(self):
        # exploration rate determines the probability of random moves
        if random.random() < self.exploration_rate_test:
            action = random.randrange(self.num_actions)
            logger.debug("Random action = %d" % action)
        else:
            # otherwise choose action with highest Q-value
            state = self.buf.getStateMinibatch()
            # for convenience getStateMinibatch() returns minibatch
            # where first item is the current state
            qvalues = self.net.predict(state)
            assert len(qvalues[0]) == self.num_actions
            # choose highest Q-value of first state
            action = np.argmax(qvalues[0])
            logger.debug("Predicted action = %d" % action)

        return action

    def play_init(self, num_games):
        # just make sure there is history_length screens to form a state
        self._restartRandom()
