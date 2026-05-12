
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
        self.options = ["Niveles", "Configuracion"]
        self.index = 0

    def display_menu(self):
        self.run_display = True
        while self.run_display:
            self.game.check_events()
            self.check_input()

            self.game.display.fill((0, 0, 0))
            mid_w, mid_h = self.game.display.get_rect().center

            title = self.game.render_bitmap_text_surface(
                "Main Menu", self.game.bitmap_font_white, spacing=self.game.bitmap_spacing
            )
            title_rect = title.get_rect(center=(mid_w, mid_h - 70))
            self.game.display.blit(title, title_rect)

            rects = []
            for i, text in enumerate(self.options):
                surf = self.game.render_bitmap_text_surface(
                    text, self.game.bitmap_font_white, spacing=self.game.bitmap_spacing
                )
                rect = surf.get_rect(center=(mid_w, mid_h + i * 30))
                self.game.display.blit(surf, rect)
                rects.append(rect)

            self.draw_cursor(rects[self.index])
            self.blit_screen()

    def check_input(self):
        if self.game.DOWN_KEY:
            self.index = (self.index + 1) % len(self.options)
        if self.game.UP_KEY:
            self.index = (self.index - 1) % len(self.options)

        if self.game.START_KEY:
            choice = self.options[self.index]

            if choice == "Niveles":
                self.game.curr_menu = self.game.levels_menu
                self.run_display = False

            elif choice == "Configuracion":
                self.game.curr_menu = self.game.config_menu
                self.run_display = False



class ConfigMenu(Menu):
    def __init__(self, game):
        super().__init__(game)
        self.state = "Volver"   # o lo que quieras como primera opción

    def display_menu(self):
        self.run_display = True
        while self.run_display:
            self.game.check_events()
            self.check_input()

            self.game.display.fill((0, 0, 0))
            mid_w, mid_h = self.game.display.get_rect().center

            title = self.game.render_bitmap_text_surface(
                "Configuracion", self.game.bitmap_font_white, spacing=self.game.bitmap_spacing
            )
            volver = self.game.render_bitmap_text_surface(
                "Volver", self.game.bitmap_font_white, spacing=self.game.bitmap_spacing
            )

            title_rect = title.get_rect(center=(mid_w, mid_h - 60))
            volver_rect = volver.get_rect(center=(mid_w, mid_h + 20))

            self.game.display.blit(title, title_rect)
            self.game.display.blit(volver, volver_rect)

            # cursor siempre en volver (por ahora)
            self.draw_cursor(volver_rect)

            self.blit_screen()

    def check_input(self):
        # ESC vuelve
        if self.game.BACK_KEY:
            self.game.curr_menu = self.game.main_menu
            self.run_display = False

        # ENTER en "Volver" vuelve
        if self.game.START_KEY and self.state == "Volver":
            self.game.curr_menu = self.game.main_menu
            self.run_display = False


class LevelsMenu(Menu):
    def __init__(self, game):
        super().__init__(game)
        self.options = ["Nivel 1", "Nivel 2", "Volver"]
        self.index = 0  # opción seleccionada

    def display_menu(self):
        self.run_display = True
        while self.run_display:
            self.game.check_events()
            self.check_input()

            self.game.display.fill((0, 0, 0))
            mid_w, mid_h = self.game.display.get_rect().center

            title = self.game.render_bitmap_text_surface(
                "Niveles", self.game.bitmap_font_white, spacing=self.game.bitmap_spacing
            )
            title_rect = title.get_rect(center=(mid_w, mid_h - 70))
            self.game.display.blit(title, title_rect)

            # dibuja opciones
            option_rects = []
            y = mid_h - 10
            for i, opt in enumerate(self.options):
                surf = self.game.render_bitmap_text_surface(
                    opt, self.game.bitmap_font_white, spacing=self.game.bitmap_spacing
                )
                rect = surf.get_rect(center=(mid_w, y + i * 25))
                self.game.display.blit(surf, rect)
                option_rects.append(rect)

            # cursor en la opción actual
            self.draw_cursor(option_rects[self.index])

            self.blit_screen()

    def check_input(self):
        if self.game.DOWN_KEY:
            self.index = (self.index + 1) % len(self.options)
        if self.game.UP_KEY:
            self.index = (self.index - 1) % len(self.options)

        # ESC vuelve
        if self.game.BACK_KEY:
            self.game.curr_menu = self.game.main_menu
            self.run_display = False

        # ENTER selecciona
        if self.game.START_KEY:
            choice = self.options[self.index]

            if choice == "Volver":
                self.game.curr_menu = self.game.main_menu
                self.run_display = False

            elif choice == "Nivel 1":
                # aquí puedes setear variables y arrancar juego
                self.game.selected_level = 1
                self.game.playing = True
                self.run_display = False

            elif choice == "Nivel 2":
                self.game.selected_level = 2
                self.game.playing = True
                self.run_display = False