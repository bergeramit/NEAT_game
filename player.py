from network import NeuralNetwork, OUTPUT_NODES_NUMBER, INPUT_NODES_NUM
import numpy as np
from game_settings import *

NORMALIZE = 1000
DIRECTIONS_TO_PLAY = {0: 'down', 1: 'up', 2: 'left', 3: 'right'}

def calculate_distance_from_goal(position, target_goal):
    return np.sqrt(np.power(position[0] - target_goal[0], 2) + np.power(position[1] - target_goal[1], 2))


def get_fitness(play):
    return play.fitness


def calculate_direction(output):
    '''
    calculate the direction to got based on the max element in the array
    '''
    maxx = -100
    direction = -1
    for i in range(OUTPUT_NODES_NUMBER):
        if maxx < output[i, 0]:
            maxx = output[i, 0]
            direction = i
    return DIRECTIONS_TO_PLAY[direction]


class Player:
    def __init__(self, target_dot):
        self.max_steps = MAX_STEPS
        self.step = 0
        self.is_dead = False
        self.nn = NeuralNetwork()
        self.position = np.array([PLAYER_START_X_POSITION, PLAYER_START_Y_POSITION], dtype=np.float32)
        self.size = np.array([PLAYER_WIDTH, PLAYER_HEIGHT], dtype=np.float32)
        # How badly this player got stuck
        self.stuck = 0
        self.prev_move = ''
        self.color = PLAYER_COLOR
        self.update_input_neurons(target_dot)

        self.is_arrived = False
        self.fitness = 0.0


    def normalize(self, *inputs):
        return np.array(list(inputs)).reshape(1,INPUT_NODES_NUM) / NORMALIZE


    def update_input_neurons(self, target_dot):
        ins = self.normalize(self.position[0], self.position[1], target_dot[0], target_dot[1], calculate_distance_from_goal(self.position, target_dot))
        self.nn.update_input_neurons(ins)


    def calculate_fitness(self, target_dot):
        if calculate_distance_from_goal(self.position, target_dot) < 5:
            self.is_arrived = True
            self.fitness = 2
            return
        self.fitness = (1 / np.power(calculate_distance_from_goal(self.position, target_dot), 3))
        self.fitness -= self.stuck / 60
        #if self.stuck != 0:
            #print("Stuck: {}".format(str(self.stuck)))


    def move(self, target_dot):
        self.step += 1
        if self.step == self.max_steps:
            self.is_dead = True

        if self.is_dead or self.is_arrived:
            return

        direction_to_go = calculate_direction(self.nn.calculate_move())
        x, y = self.position
        if direction_to_go == 'left':
            if self.prev_move == 'right':
                self.stuck += 1
            else:
                self.stuck = 0
            x -= 5
        elif direction_to_go == 'right':
            if self.prev_move == 'left':
                self.stuck += 1
            else:
                self.stuck = 0
            x += 5
        elif direction_to_go == 'up':
            if self.prev_move == 'down':
                self.stuck += 1
            else:
                self.stuck = 0
            y -= 5
        elif direction_to_go == 'down':
            if self.prev_move == 'up':
                self.stuck += 1
            else:
                self.stuck = 0
            y += 5

        self.prev_move = direction_to_go

        if calculate_distance_from_goal(self.position, target_dot) < 5:
            self.is_arrived = True

        self.position = x, y
        self.update_input_neurons(target_dot)

