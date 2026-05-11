
from game import Game
import pygame

g = Game()

while g.running:
    g.curr_menu.display_menu()

    if g.playing:
        g.run()              # correr nivel seleccionado
        g.playing = False    # por si acaso
        g.curr_menu = g.main_menu

pygame.quit()

