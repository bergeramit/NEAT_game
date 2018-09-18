import pygame
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
    first_man = Player()
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(BACKGROUND_SIZE)
    count = 0

    while not stop:

        screen.fill(BACKGROUND_COLOR)
        pygame.draw.rect(screen, PLAYER_COLOR, [first_man.position[0], first_man.position[1], first_man.size[0], first_man.size[1]])
        pygame.draw.rect(screen, TARGET_COLOR, [TARGET_DOT[0], TARGET_DOT[1], TARGET_WIDTH, TARGET_HEIGHT])
        pygame.display.update()
        clock.tick(60)

        first_man.move()
        #print(str(first_man.nn))

        count += 1
        if count  % 5 == 0:
            first_man.nn.add_node()

        if first_man.is_dead:
            stop = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop = True


    pygame.quit()

if __name__ == "__main__":
    #test()
    start()
