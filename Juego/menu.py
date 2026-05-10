
import pygame

class Menu:
    def __init__(self, game):
        self.game = game
        self.run_display = True
        self.state = None  # lo define cada menú

    def draw_cursor(self, option_rect):
        cursor = self.game.render_bitmap_text_surface(
            ".", self.game.bitmap_font_big_white, spacing=self.game.bitmap_spacing
        )
        # cursor a la izquierda de la opción seleccionada
        cursor_rect = cursor.get_rect(midright=(option_rect.left - 10, option_rect.centery))
        self.game.display.blit(cursor, cursor_rect)

    def blit_screen(self):
        self.game.screen.fill((0, 0, 0))
        scaled = pygame.transform.scale(self.game.display, self.game.blit_size)
        self.game.screen.blit(scaled, self.game.blit_offset)
        pygame.display.update()
        self.game.reset_keys()
        self.game.clock.tick(60)


class MainMenu(Menu):
    def __init__(self, game):
        super().__init__(game)
        self.state = "Start"

    def display_menu(self):
        self.run_display = True

        while self.run_display:
            self.game.check_events()
            self.check_input()

            # --- dibujar en DISPLAY (480x270) ---
            self.game.display.fill((0, 0, 0))
            mid_w, mid_h = self.game.display.get_rect().center

            mm = self.game.render_bitmap_text_surface(
                "MAIN MENU", self.game.bitmap_font_white, spacing=self.game.bitmap_spacing
            )
            sg = self.game.render_bitmap_text_surface(
                "Start Game", self.game.bitmap_font_white, spacing=self.game.bitmap_spacing
            )
            op = self.game.render_bitmap_text_surface(
                "Options", self.game.bitmap_font_white, spacing=self.game.bitmap_spacing
            )

            mm_rect = mm.get_rect(center=(mid_w, mid_h - 60))
            sg_rect = sg.get_rect(center=(mid_w, mid_h + 10))
            op_rect = op.get_rect(center=(mid_w, mid_h + 40))

            self.game.display.blit(mm, mm_rect)
            self.game.display.blit(sg, sg_rect)
            self.game.display.blit(op, op_rect)

            # --- cursor según estado ---
            if self.state == "Start":
                self.draw_cursor(sg_rect)
            else:
                self.draw_cursor(op_rect)

            # --- mostrar en pantalla ---
            self.blit_screen()

    def move_cursor(self):
        if self.game.DOWN_KEY:
            self.state = "Options" if self.state == "Start" else "Start"
        if self.game.UP_KEY:
            self.state = "Options" if self.state == "Start" else "Start"

    def check_input(self):
        self.move_cursor()

        if self.game.START_KEY:
            if self.state == "Start":
                self.game.playing = True
                self.run_display = False  # salir del menú
            elif self.state == "Options":
                print("Options selected (pendiente)")
