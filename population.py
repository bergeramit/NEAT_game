from game_settings import *
import numpy as np
from network import OUTPUT_NODES_NUMBER, INPUT_NODES_NUM
import pygame
from player import Player, get_fitness
import random


class Generation:
    def __init__(self):
        self.old_population = []
        self.population = []
        self.is_extinct = False
        for _ in range(SIZE_OF_POPULATION):
            self.population.append(Player())

        self.generation_counter = 1


    def move(self):
        self.is_extinct = True
        for play in self.population:
            if not play.is_dead:
                self.is_extinct = False
                play.move()


    def show(self, screen):
        c = 0
        for play in self.population:
            if c > 2:
                break
            if not play.is_dead:
                pygame.draw.rect(screen, PLAYER_COLOR, [play.position[0], play.position[1], play.size[0], play.size[1]])
                c += 1


    def calculate_fitness(self):
        for play in self.population:
            play.calculate_fitness()
        self.population.sort(key=get_fitness, reverse=True)


    def clone_elite(self):
        self.population = self.old_population[:ELITE_NUMBER]


    def revive(self):
        self.is_extinct = False
        self.old_population = []
        for play in self.population:
            play.is_dead =- False
            play.step = 0
            play.fitness = 0
            play.is_arrived = False
            play.position = np.array([random.uniform(0, BACKGROUND_SIZE[0]), random.uniform(0, BACKGROUND_SIZE[1])], dtype=np.float32)


    def generate_new_population(self):
        self.generation_counter += 1
        self.is_extinct = False
        self.calculate_fitness()

        self.old_population = self.population[:MATE_POPULATION]
        self.population = []

        self.clone_elite()
        self.mate()
        self.revive()


    def generate_parents(self):
        sum_of_fitness = 0
        parents = np.zeros((SIZE_OF_POPULATION - ELITE_NUMBER, 2), dtype=np.float32)
        for i in range(MATE_POPULATION):
            sum_of_fitness += self.old_population[i].fitness

        place = sum_of_fitness * np.random.random(SIZE_OF_POPULATION - ELITE_NUMBER)
        for j in range(SIZE_OF_POPULATION - ELITE_NUMBER):
            first_index = 0
            for i in range(MATE_POPULATION):
                if place[j] <= self.old_population[i].fitness:
                    first_index = i
                    break
                else:
                    place[j] -= self.old_population[i].fitness

            # I chose the first index random with respect to the fitness and the second is its predecessor
            second_index = first_index - 1
            if first_index == 0:
                second_index = 1

            np.append(parents, [first_index, second_index])

        return parents


    def mate_parents(self, first, second):
        parent1 = self.old_population[int(first)]
        parent2 = self.old_population[int(second)]
        son = Player()

        # Chose bias
        son.nn.bias = parent1.nn.bias
        if random.uniform(0, 1) > 0.5:
            son.nn.bias = parent2.nn.bias

        # Pick Weights
        for i in range(OUTPUT_NODES_NUMBER):
            for j in range(INPUT_NODES_NUM):
                coin = random.uniform(0,1)
                if coin < 0.05:
                    new_value = 0
                elif coin > 0.6:
                    new_value = parent1.nn.weights_out[i, j]
                else:
                    new_value = parent2.nn.weights_out[i, j]

                son.nn.weights_out[i, j] = new_value

        # In case of hidden neurons
        for i in range(parent1.nn.hidden_neurons):
            if random.uniform(0, 1) > 0.5:
                son.nn.add_node()
                son.nn.hidden_neurons_weights_in[son.nn.hidden_neurons - 1] = parent1.nn.hidden_neurons_weights_in[i]
                son.nn.hidden_neurons_weights_out[:,-1] = parent1.nn.hidden_neurons_weights_out[:,i]

        # In case of hidden neurons
        for i in range(parent2.nn.hidden_neurons):
            if random.uniform(0, 1) > 0.5:
                son.nn.add_node()
                son.nn.hidden_neurons_weights_in[son.nn.hidden_neurons - 1] = parent2.nn.hidden_neurons_weights_in[i]
                son.nn.hidden_neurons_weights_out[:,-1] = parent2.nn.hidden_neurons_weights_out[:,i]

        # Adding new neuron
        if random.uniform(0,1) < MUTATE_RATE:
            son.nn.add_node()

        return son


    def mate(self):
        parents = self.generate_parents()

        for first,second in parents:
            self.population.append(self.mate_parents(first, second))

