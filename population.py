from game_settings import *
import pygame
from player import Player



class Generation:
    def __init__(self):
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
        for play in self.population:
            pygame.draw.rect(screen, PLAYER_COLOR, [play.position[0], play.position[1], play.size[0], play.size[1]])
