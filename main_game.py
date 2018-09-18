import pygame
from game_settings import *
from network import NeuralNetwork


def test():
    a = NeuralNetwork()
    a.add_node()
    a.add_node()
    a.calculate_move()
    a.calculate_move()
    a.add_node()
    a.calculate_move()


def start():
    stop = False
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(BACKGROUND_SIZE)

    while not stop:

        screen.fill(BACKGROUND_COLOR)
        pygame.display.update()
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop = True


    pygame.quit()

if __name__ == "__main__":
    test()
    #start()
