from network import NeuralNetwork, INPUT_NODES_NUM
import numpy as np
from game_settings import *


def calculate_distance_from_goal(position, target_goal):
    return np.sqrt(np.power(position[0] - target_goal[0], 2) + np.power(position[1] - target_goal[1], 2))


def get_fitness(play):
    return play.fitness


class Player:
    def __init__(self):
        self.max_steps = MAX_STEPS
        self.step = 0
        self.is_dead = False
        self.nn = NeuralNetwork()
        self.position = np.array([PLAYER_START_X_POSITION, PLAYER_START_Y_POSITION], dtype=np.float32)
        self.size = np.array([PLAYER_WIDTH, PLAYER_HEIGHT], dtype=np.float32)
        self.color = PLAYER_COLOR
        self.update_input_neurons()
        self.previous_direction = ''
        self.is_arrived = False
        self.fitness = 0.0


    def update_input_neurons(self):
        self.nn.update_input_neurons(self.position[0], self.position[1], TARGET_DOT[0], TARGET_DOT[1])


    def calculate_fitness(self):
        if calculate_distance_from_goal(self.position, TARGET_DOT) < 5:
            self.is_arrived = True
            self.fitness = 2
            return
        self.fitness = 1 / np.power(calculate_distance_from_goal(self.position, TARGET_DOT), 2)


    def move(self):
        self.step += 1
        if self.step == self.max_steps:
            self.is_dead = True

        if self.is_dead or self.is_arrived:
            return

        direction_to_go = self.nn.calculate_move(self.previous_direction)
        x, y = self.position
        if direction_to_go == 'left':
            x -= 5
        elif direction_to_go == 'right':
            x += 5
        elif direction_to_go == 'up':
            y -= 5
        elif direction_to_go == 'down':
            y += 5


        if calculate_distance_from_goal(self.position, TARGET_DOT) < 5:
            self.is_arrived = True

        self.position = x, y
        self.update_input_neurons()

