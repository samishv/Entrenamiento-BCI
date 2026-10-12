import pygame

class Menu:
    def __init__(self, game):
        self.game = game
        self.run_display = True
        self.state = None
        
        # ===== CONSTANTE DE DESPLAZAMIENTO VERTICAL (SIN ESTO NO CUADRAN LOS PX) =====
        self.offset_y = 5

    def draw_cursor(self, option_rect):
        cursor = self.game.render_bitmap_text_surface(
            ">", self.game.bitmap_font_white, spacing=self.game.bitmap_spacing
        )
        cursor_rect = cursor.get_rect(midright=(option_rect.left - 10, option_rect.centery))
        self.game.display.blit(cursor, cursor_rect)

    def blit_screen(self):
        self.game.screen.fill((0, 0, 0))
        scaled = pygame.transform.scale(self.game.display, self.game.blit_size)
        self.game.screen.blit(scaled, self.game.blit_offset)
        pygame.display.update()
        self.game.reset_keys()
        self.game.clock.tick(60)
        
        
# ===== MENU PRINCIPAL =====
class MainMenu(Menu):
    def __init__(self, game):
        super().__init__(game)
        self.options = ["NIVELES", "CONFIGURACION"]
        self.index = 0

    def display_menu(self):
        self.run_display = True
        while self.run_display:
            self.game.check_events()
            self.check_input()

            self.game.display.blit(self.game.menu_bg, (0, 0))
            mid_w, _ = self.game.display.get_rect().center

            # ===== TITULO DE MAIN MENU =====
            title = self.game.render_bitmap_text_surface(
                "MAIN MENU", self.game.bitmap_font_white, spacing=self.game.bitmap_spacing
            )
            title_rect = title.get_rect(center=(mid_w, 50 + self.offset_y))
            self.game.display.blit(title, title_rect)

            # ===== ALTURAS DE NIVELES Y CONFIG =====
            posiciones_y = [140, 170]
            rects = []
            for i, text in enumerate(self.options):
                surf = self.game.render_bitmap_text_surface(
                    text, self.game.bitmap_font_white, spacing=self.game.bitmap_spacing
                )
                rect = surf.get_rect(center=(mid_w, posiciones_y[i] + self.offset_y))
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

            if choice == "NIVELES":
                self.game.curr_menu = self.game.levels_menu
                self.run_display = False

            elif choice == "CONFIGURACION":
                self.game.curr_menu = self.game.config_menu
                self.run_display = False


# ===== VENTANA MENU DE CONFIG =====
class ConfigMenu(Menu):
    def __init__(self, game):
        super().__init__(game)
        self.state = "MODO"  # opciones: "BT", "MODO", "COLOR" o "Volver"
        
        self.nickname_buffer = ""
        self.max_nickname_len = 20
        self.cursor_blink_ms = 500

    def display_menu(self):
        self.run_display = True
        pygame.key.start_text_input()
        pygame.key.set_repeat(300, 35)
        self.nickname_buffer = self.game.gamer_tag

        while self.run_display:
            self.game.check_events()

            # ── Nickname input ──────────────────────────────
            if self.game.TEXT_INPUT:
                self.nickname_buffer += self.game.TEXT_INPUT
            if self.game.BACKSPACE and len(self.nickname_buffer) > 0:
                self.nickname_buffer = self.nickname_buffer[:-1]
            allowed = set(self.game.bitmap_charset)
            self.nickname_buffer = "".join(
                ch for ch in self.nickname_buffer if ch in allowed
            )
            self.nickname_buffer = self.nickname_buffer[:self.max_nickname_len]
            self.check_input()

            self.game.display.blit(self.game.menu_bg, (0, 0))
            mid_w, _ = self.game.display.get_rect().center

            # ── Título ──────────────────────────────────────
            title = self.game.render_bitmap_text_surface(
                "CONFIGURACION", self.game.bitmap_font_white,
                spacing=self.game.bitmap_spacing
            )
            self.game.display.blit(
                title, title.get_rect(center=(mid_w, 50 + self.offset_y))
            )

            # ── Nickname label ──────────────────────────────
            nick_label = self.game.render_bitmap_text_surface(
                "NICKNAME", self.game.bitmap_font_white,
                spacing=self.game.bitmap_spacing
            )
            self.game.display.blit(
                nick_label,
                nick_label.get_rect(center=(mid_w, 110 + self.offset_y))
            )

            # ── Nickname typing + caret ─────────────────────
            typed_y   = 140 + self.offset_y
            base_text = self.nickname_buffer

            if base_text:
                base_surf = self.game.render_bitmap_text_surface(
                    base_text, self.game.bitmap_font_white,
                    spacing=self.game.bitmap_spacing
                )
                base_rect = base_surf.get_rect(center=(mid_w, typed_y))
                self.game.display.blit(base_surf, base_rect)
            else:
                base_rect = None

            if (pygame.time.get_ticks() // self.cursor_blink_ms) % 2 == 0:
                caret_surf = self.game.render_bitmap_text_surface(
                    "|", self.game.bitmap_font_white,
                    spacing=self.game.bitmap_spacing
                )
                if base_rect:
                    caret_rect = caret_surf.get_rect(
                        midleft=(base_rect.right + 2, base_rect.centery)
                    )
                else:
                    caret_rect = caret_surf.get_rect(center=(mid_w, typed_y))
                self.game.display.blit(caret_surf, caret_rect)

            # ── Fila horizontal de configuracion ─────────────

            options_y = 190 + self.offset_y

            # Posiciones horizontales provisionales.
            bt_x = 100
            modo_x = 250
            color_x = 400

            # Dibujar las tres opciones.
            bt_rect = self._draw_bt_switch(bt_x, options_y)
            modo_rect = self._draw_hand_mode(modo_x, options_y)
            color_rect = self._draw_player_color(color_x, options_y)

            # ── Opcion Volver ────────────────────────────────

            volver = self.game.render_bitmap_text_surface(
                "VOLVER",
                self.game.bitmap_font_white,
                spacing=self.game.bitmap_spacing
            )

            volver_rect = volver.get_rect(
                center=(mid_w, 235 + self.offset_y)
            )

            self.game.display.blit(volver, volver_rect)

            # ── Cursor de la opcion seleccionada ─────────────

            if self.state == "BT":
                self.draw_cursor(bt_rect)

            elif self.state == "MODO":
                self.draw_cursor(modo_rect)

            elif self.state == "COLOR":
                self.draw_cursor(color_rect)

            elif self.state == "VOLVER":
                self.draw_cursor(volver_rect)

            self.blit_screen()

        pygame.key.stop_text_input()
        pygame.key.set_repeat()

    # ── Dibuja el switch ────────────────────────────────────
    def _draw_bt_switch(self, x, y):
        """
        Dibuja la opcion ARIADNE: OFF o ARIADNE: ON
        centrada en la posicion recibida.
        """
        enabled = self.game.bluetooth_enabled

        text_str = "ON" if enabled else "OFF"
        box_color = (0, 200, 80) if enabled else (0, 0, 0)

        # Texto ARIADNE:
        label = self.game.render_bitmap_text_surface(
            "ARIADNE",
            self.game.bitmap_font_white,
            spacing=self.game.bitmap_spacing
        )

        # Texto ON/OFF
        switch_text = self.game.render_bitmap_text_surface(
            text_str,
            self.game.bitmap_font_white,
            spacing=self.game.bitmap_spacing
        )

        spacing_between = 4
        box_padding_x = 4
        box_padding_y = 2

        box_w = switch_text.get_width() + (box_padding_x * 2)
        box_h = switch_text.get_height() + (box_padding_y * 2)

        total_width = (
            label.get_width()
            + spacing_between
            + box_w
        )

        start_x = x - (total_width // 2)

        label_rect = label.get_rect(
            midleft=(start_x, y)
        )

        box_rect = pygame.Rect(
            label_rect.right + spacing_between,
            y - (box_h // 2),
            box_w,
            box_h
        )

        self.game.display.blit(label, label_rect)

        pygame.draw.rect(
            self.game.display,
            box_color,
            box_rect,
            border_radius=3
        )

        self.game.display.blit(
            switch_text,
            switch_text.get_rect(center=box_rect.center)
        )

        option_rect = label_rect.union(box_rect)

        self._bt_rect = box_rect
        self._bt_option_rect = option_rect

        return option_rect
        
    def _draw_hand_mode(self, x, y):
        """
        Dibuja unicamente DIESTRO o ZURDO.
        """
        text = self.game.hand_mode
    
        mode_surface = self.game.render_bitmap_text_surface(
            text,
            self.game.bitmap_font_white,
            spacing=self.game.bitmap_spacing
        )
    
        mode_rect = mode_surface.get_rect(
            center=(x, y)
        )
    
        self.game.display.blit(
            mode_surface,
            mode_rect
        )
    
        return mode_rect

    def _draw_player_color(self, x, y):
        """
        Dibuja unicamente el nombre del color seleccionado.
        El cuadro de color se conserva debajo de las letras.
        """
    
        # Crear el texto con el nombre del color.
        color_surface = self.game.render_bitmap_text_surface(
            self.game.player_color,
            self.game.bitmap_font_white,
            spacing=self.game.bitmap_spacing
        )
    
        # Espacio adicional alrededor de las letras.
        padding_x = 4
        padding_y = 2
    
        # El cuadro se adapta al tamaño del nombre del color.
        color_box_width = color_surface.get_width() + (padding_x * 2)
        color_box_height = color_surface.get_height() + (padding_y * 2)
    
        # Centrar el cuadro en la posicion recibida.
        color_box_rect = pygame.Rect(
            x - (color_box_width // 2),
            y - (color_box_height // 2),
            color_box_width,
            color_box_height
        )
    
        # Obtener el color correspondiente a la gama seleccionada.
        background_color = self.game.get_player_menu_color()
    
        # Dibujar primero el cuadro de color.
        pygame.draw.rect(
            self.game.display,
            background_color,
            color_box_rect,
            border_radius=3
        )
    
        # Dibujar el nombre del color encima del cuadro.
        color_text_rect = color_surface.get_rect(
            center=color_box_rect.center
        )
    
        self.game.display.blit(
            color_surface,
            color_text_rect
        )
    
        # Este rectangulo se usa para colocar el cursor de seleccion.
        return color_box_rect

    def check_input(self):
        """
        Navegacion visual del menu:

        BT  <->  MODO  <->  COLOR
                  |
                VOLVER
        """

        top_options = ["BT", "MODO", "COLOR"]

        # ── Navegacion horizontal ────────────────────────────

        if self.state in top_options:
            current_index = top_options.index(self.state)

            if self.game.RIGHT_KEY:
                self.state = top_options[
                    (current_index + 1) % len(top_options)
                ]

            elif self.game.LEFT_KEY:
                self.state = top_options[
                    (current_index - 1) % len(top_options)
                ]

            # Desde cualquier opcion superior, abajo va a Volver.
            elif self.game.DOWN_KEY:
                self.state = "VOLVER"

        # ── Navegacion desde Volver ──────────────────────────

        elif self.state == "VOLVER":

            # Arriba siempre regresa a la opcion central MODO.
            if self.game.UP_KEY:
                self.state = "MODO"

        # ── Ejecutar la opcion seleccionada ──────────────────

        if self.game.START_KEY:

            if self.state == "BT":
                self.game.bluetooth_enabled = (
                    not self.game.bluetooth_enabled
                )

            elif self.state == "MODO":
                self.game.toggle_hand_mode()

            elif self.state == "COLOR":
                self.game.next_player_color()

            elif self.state == "VOLVER":
                self._save_and_return()

        # Escape regresa al menu principal.
        if self.game.BACK_KEY:
            self._save_and_return()

    def _save_and_return(self):
        cleaned = self.nickname_buffer.strip()
        if cleaned:
            self.game.gamer_tag = cleaned
        self.game.curr_menu = self.game.main_menu
        self.run_display = False


# ===== VENTANA NIVELES =====
class LevelsMenu(Menu):
    def __init__(self, game):
        super().__init__(game)
        self.options = ["NIVEL 1", "NIVEL 2", "VOLVER"]
        self.index = 0

    def display_menu(self):
        self.run_display = True
        while self.run_display:
            self.game.check_events()
            self.check_input()

            self.game.display.blit(self.game.menu_bg, (0, 0))
            mid_w, _ = self.game.display.get_rect().center

            # ===== TITULO DE NIVELES =====
            title = self.game.render_bitmap_text_surface(
                "NIVELES", self.game.bitmap_font_white, spacing=self.game.bitmap_spacing
            )
            title_rect = title.get_rect(center=(mid_w, 50 + self.offset_y))
            self.game.display.blit(title, title_rect)

            # ===== ALTURAS PARA LVL 1, LVL 2 Y VOLVER =====
            posiciones_y = [110, 140, 200]
            option_rects = []
            for i, opt in enumerate(self.options):
                surf = self.game.render_bitmap_text_surface(
                    opt, self.game.bitmap_font_white, spacing=self.game.bitmap_spacing
                )
                rect = surf.get_rect(center=(mid_w, posiciones_y[i] + self.offset_y))
                self.game.display.blit(surf, rect)
                option_rects.append(rect)

            self.draw_cursor(option_rects[self.index])
            self.blit_screen()

    def check_input(self):
        if self.game.DOWN_KEY:
            self.index = (self.index + 1) % len(self.options)
        if self.game.UP_KEY:
            self.index = (self.index - 1) % len(self.options)

        if self.game.BACK_KEY:
            self.game.curr_menu = self.game.main_menu
            self.run_display = False

        if self.game.START_KEY:
            choice = self.options[self.index]

            if choice == "VOLVER":
                self.game.curr_menu = self.game.main_menu
                self.run_display = False

            elif choice == "NIVEL 1":
                self.game.selected_level = 1
                self.game.playing = True
                self.run_display = False

            elif choice == "NIVEL 2":
                self.game.selected_level = 2
                self.game.playing = True
                self.run_display = False
