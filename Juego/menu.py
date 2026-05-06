import pygame

class Menu:
    def __init__(self,game):
        self.game = game
        self.mid_w, self.mid_h = self.game.screen.get_rect().center
        self.run_display = True  # Variable para controlar la ejecución del menú
        self.cursor_rect = pygame.Rect(0, 0, 20, 20)
        self.offset = -100 # Offset para posicionar el cursor a la izquierda del texto


    def draw_cursor(self):
        cursor = self.game.render_bitmap_text_surface("*", self.bitmap_font, spacing=self.bitmap_spacing)
        self.game.screen.blit(cursor, (self.cursor_rect.x, self.cursor_rect.y))

    def blit_screen(self):
        self.game.screen.blit(self.game.display, (0, 0))
        pygame.display.update()
        self.game.reset_keys()

class MainMenu(Menu):
    def __init__(self, game):
        Menu.__init__(self,game)
        self.state = "Start"
        self.startx, self.starty = self.mid_w, self.mid_h + 30
        self.optionsx, self.optionsy = self.mid_w, self.mid_h + 60
        self.cursor_rect.midtop = (self.startx + self.offset, self.starty)

    def display_menu(self):
        self.run_display = True
        while self.run_display:
            self.game.display.fill((0, 0, 0))  # Limpiar la pantalla
            mm=self.game.render_bitmap_text("Main Menu", self.mid_w, self.mid_h - 30, spacing=10)
            sg=self.game.render_bitmap_text("Start Game", self.startx, self.starty, spacing=10)
            op=self.game.render_bitmap_text("Options", self.optionsx, self.optionsy, spacing=10)
            self.game.screen.blit(mm, (self.mid_w, self.mid_h - 30))
            self.game.screen.blit(sg, (self.startx, self.starty))
            self.game.screen.blit(op, (self.optionsx, self.optionsy))
            self.draw_cursor()
            self.game.check_events()
            self.check_input()
            self.blit_screen()
            
    def move_cursor(self):
        if self.game.DOWN_KEY:
            if self.state == "Start":
                self.cursor_rect.midtop = (self.optionsx + self.offset, self.optionsy)
                self.state = "Options"
            elif self.state == "Options":
                self.cursor_rect.midtop = (self.startx + self.offset, self.starty)
                self.state = "Start"
        elif self.game.UP_KEY:
            if self.state == "Start":
                self.cursor_rect.midtop = (self.optionsx + self.offset, self.optionsy)
                self.state = "Options"
            elif self.state == "Options":
                self.cursor_rect.midtop = (self.startx + self.offset, self.starty)
                self.state = "Start"

    def check_input(self):
        self.move_cursor()
        if self.game.START_KEY:
            if self.state == "Start":
                print("Start Game selected")
                # Aquí puedes agregar la lógica para iniciar el juego
            elif self.state == "Options":
                print("Options selected")
                # Aquí puedes agregar la lógica para mostrar las opciones del juego 