import pygame
from population import Generation, TARGET_DOT
from game_settings import *
from network import NeuralNetwork
from player import Player


def test():
    a = NeuralNetwork()
    a.add_node()
    a.add_node()
    a.calculate_move()
    a.calculate_move()
    a.add_node()
    a.calculate_move()


    #pygame.draw.rect(screen, terrain.color, [terrain.position.x, terrain.position.y, terrain.height, terrain.width])

def start():
    stop = False
    pygame.init()
    pop = Generation()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(BACKGROUND_SIZE)

    while not stop:

        screen.fill(BACKGROUND_COLOR)
        pygame.draw.rect(screen, TARGET_COLOR, [pop.target_dot[0], pop.target_dot[1], TARGET_WIDTH, TARGET_HEIGHT])
        pop.move()
        pop.show(screen)
        pygame.display.update()
        clock.tick(60)

        #print(str(first_man.nn))
        if pop.is_extinct:
            pop.generate_new_population()
            print("Generation: {}".format(pop.generation_counter))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop = True


    pygame.quit()

if __name__ == "__main__":
    #test()
    start()
