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
    # ===== DISTRIBUCION VISUAL DEL MENU DE CONFIGURACION =====
    TITLE_Y = 20
    NICKNAME_Y = 70
    NICKNAME_LINE = (280, 440)

    OPTIONS_X = 120
    
    BT_Y = 120
    MODE_Y = 165
    COLOR_Y = 210
    
    BACK_Y = 250

    PREVIEW_RECT = pygame.Rect(300, 100, 120, 120)
    PREVIEW_FRAME_MS = 300

    def __init__(self, game):
        super().__init__(game)
        self.state = "MODO"  # opciones: "BT", "MODO", "COLOR" o "VOLVER"
        self.nickname_buffer = ""
        self.max_nickname_len = 20
        self.cursor_blink_ms = 500

        # Cache independiente para la previsualizacion del personaje.
        # Se crea al entrar al menu, cuando los assets del juego ya existen.
        self.preview_color = None
        self.preview_idle_frames = []

    def display_menu(self):
        self.run_display = True
        pygame.key.start_text_input()
        pygame.key.set_repeat(300, 35)
        self.nickname_buffer = self.game.gamer_tag

        while self.run_display:
            self.game.check_events()

            # ── Entrada del nickname ──────────────────────────
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

            # ── Titulo ────────────────────────────────────────
            title = self.game.render_bitmap_text_surface(
                "CONFIGURACION",
                self.game.bitmap_font_white,
                spacing=self.game.bitmap_spacing,
            )
            self.game.display.blit(
                title,
                title.get_rect(center=(mid_w, self.TITLE_Y + self.offset_y)),
            )

            # ── Nickname en una sola fila ─────────────────────
            nick_label = self.game.render_bitmap_text_surface(
                "NICKNAME",
                self.game.bitmap_font_white,
                spacing=self.game.bitmap_spacing,
            )
            self.game.display.blit(
                nick_label,
                nick_label.get_rect(center=(self.OPTIONS_X, self.NICKNAME_Y)),
            )
            self._draw_nickname_field()

            # ── Opciones verticales a la izquierda ────────────
            bt_rect = self._draw_bt_switch(self.OPTIONS_X, self.BT_Y)
            mode_rect = self._draw_hand_mode(self.OPTIONS_X, self.MODE_Y)
            color_rect = self._draw_player_color(self.OPTIONS_X, self.COLOR_Y)

            # ── Previsualizacion animada a la derecha ─────────
            self._draw_player_preview()

            # ── Opcion Volver ─────────────────────────────────
            volver = self.game.render_bitmap_text_surface(
                "VOLVER",
                self.game.bitmap_font_white,
                spacing=self.game.bitmap_spacing,
            )
            volver_rect = volver.get_rect(center=(mid_w, self.BACK_Y))
            self.game.display.blit(volver, volver_rect)

            # ── Cursor de la opcion seleccionada ──────────────
            option_rects = {
                "BT": bt_rect,
                "MODO": mode_rect,
                "COLOR": color_rect,
                "VOLVER": volver_rect,
            }
            self.draw_cursor(option_rects[self.state])
            self.blit_screen()

        pygame.key.stop_text_input()
        pygame.key.set_repeat()

    def _draw_nickname_field(self):
        """Dibuja el nickname sobre una linea, como en el boceto."""
        line_left, line_right = self.NICKNAME_LINE
        y = self.NICKNAME_Y
        pygame.draw.line(
            self.game.display,
            (255, 255, 255),
            (line_left, y + 12),
            (line_right, y + 12),
            1,
        )

        if self.nickname_buffer:
            nickname = self.game.render_bitmap_text_surface(
                self.nickname_buffer,
                self.game.bitmap_font_white,
                spacing=self.game.bitmap_spacing,
            )
            nickname_rect = nickname.get_rect(
                midbottom=((line_left + line_right) // 2, y + 5)
            )
            self.game.display.blit(nickname, nickname_rect)
        else:
            nickname_rect = None

        if (pygame.time.get_ticks() // self.cursor_blink_ms) % 2 == 0:
            caret = self.game.render_bitmap_text_surface(
                "|",
                self.game.bitmap_font_white,
                spacing=self.game.bitmap_spacing,
            )
            if nickname_rect:
                caret_rect = caret.get_rect(
                    midleft=(nickname_rect.right + 2, nickname_rect.centery)
                )
            else:
                caret_rect = caret.get_rect(
                    center=((line_left + line_right) // 2, y)
                )
            self.game.display.blit(caret, caret_rect)

    def _refresh_preview_frames(self):
        """Recolorea el idle solo cuando cambia el color seleccionado."""
        if self.preview_color == self.game.player_color and self.preview_idle_frames:
            return

        self.preview_idle_frames = self.game._recolor_player_images(
            self.game.original_player_images["idle"],
            self.game.player_color,
        )
        self.preview_color = self.game.player_color

    def _draw_player_preview(self):
        """Dibuja un marco y reproduce dentro la animacion idle actual."""
        preview_rect = self.PREVIEW_RECT

        # Fondo semitransparente para separar al personaje del fondo del menu.
        preview_bg = pygame.Surface(preview_rect.size, pygame.SRCALPHA)
        preview_bg.fill((73, 103, 141, 255))
        self.game.display.blit(preview_bg, preview_rect.topleft)
        pygame.draw.rect(self.game.display, (255, 255, 255), preview_rect, 2)

        self._refresh_preview_frames()
        if not self.preview_idle_frames:
            return

        frame_index = (
            pygame.time.get_ticks() // self.PREVIEW_FRAME_MS
        ) % len(self.preview_idle_frames)
        frame = self.preview_idle_frames[frame_index]

        # Escalado entero para conservar nitido el pixel art.
        max_w = preview_rect.width - 20
        max_h = preview_rect.height - 16
        scale = max(
            1,
            min(max_w // frame.get_width(), max_h // frame.get_height()),
        )
        preview_frame = pygame.transform.scale(
            frame,
            (frame.get_width() * scale, frame.get_height() * scale),
        )
        frame_rect = preview_frame.get_rect(center=preview_rect.center)
        self.game.display.blit(preview_frame, frame_rect)

    def _draw_bt_switch(self, x, y):
        """Dibuja ARIADNE junto con su estado ON/OFF."""
        enabled = self.game.bluetooth_enabled
        text_str = "ON" if enabled else "OFF"
        box_color = (0, 200, 80) if enabled else (0, 0, 0)

        label = self.game.render_bitmap_text_surface(
            "ARIADNE",
            self.game.bitmap_font_white,
            spacing=self.game.bitmap_spacing,
        )
        switch_text = self.game.render_bitmap_text_surface(
            text_str,
            self.game.bitmap_font_white,
            spacing=self.game.bitmap_spacing,
        )

        spacing_between = 4
        padding_x, padding_y = 4, 2
        box_w = switch_text.get_width() + padding_x * 2
        box_h = switch_text.get_height() + padding_y * 2
        total_width = label.get_width() + spacing_between + box_w
        start_x = x - total_width // 2

        label_rect = label.get_rect(midleft=(start_x, y))
        box_rect = pygame.Rect(
            label_rect.right + spacing_between,
            y - box_h // 2,
            box_w,
            box_h,
        )
        self.game.display.blit(label, label_rect)
        pygame.draw.rect(self.game.display, box_color, box_rect, border_radius=3)
        self.game.display.blit(
            switch_text,
            switch_text.get_rect(center=box_rect.center),
        )
        return label_rect.union(box_rect)

    def _draw_hand_mode(self, x, y):
        """Dibuja unicamente DIESTRO o ZURDO."""
        mode_surface = self.game.render_bitmap_text_surface(
            self.game.hand_mode,
            self.game.bitmap_font_white,
            spacing=self.game.bitmap_spacing,
        )
        mode_rect = mode_surface.get_rect(center=(x, y))
        self.game.display.blit(mode_surface, mode_rect)
        return mode_rect

    def _draw_player_color(self, x, y):
        """Dibuja el nombre del color sobre su cuadro de color."""
        color_surface = self.game.render_bitmap_text_surface(
            self.game.player_color,
            self.game.bitmap_font_white,
            spacing=self.game.bitmap_spacing,
        )
        padding_x, padding_y = 4, 2
        color_box_rect = pygame.Rect(
            x - (color_surface.get_width() + padding_x * 2) // 2,
            y - (color_surface.get_height() + padding_y * 2) // 2,
            color_surface.get_width() + padding_x * 2,
            color_surface.get_height() + padding_y * 2,
        )
        pygame.draw.rect(
            self.game.display,
            self.game.get_player_menu_color(),
            color_box_rect,
            border_radius=3,
        )
        self.game.display.blit(
            color_surface,
            color_surface.get_rect(center=color_box_rect.center),
        )
        return color_box_rect

    def check_input(self):
        """
        Navegacion vertical del nuevo acomodo:

        ARIADNE
           |
          MODO
           |
          COLOR
           |
         VOLVER

        ENTER ejecuta la opcion. En BT, MODO y COLOR tambien se puede usar
        IZQUIERDA/DERECHA para cambiar el valor sin alterar la seleccion.
        """
        options = ["BT", "MODO", "COLOR", "VOLVER"]
        current_index = options.index(self.state)

        if self.game.DOWN_KEY:
            self.state = options[(current_index + 1) % len(options)]
        elif self.game.UP_KEY:
            self.state = options[(current_index - 1) % len(options)]

        change_value = self.game.LEFT_KEY or self.game.RIGHT_KEY
        if self.game.START_KEY or change_value:
            if self.state == "BT":
                self.game.bluetooth_enabled = not self.game.bluetooth_enabled
            elif self.state == "MODO":
                self.game.toggle_hand_mode()
            elif self.state == "COLOR":
                self.game.next_player_color()
            elif self.state == "VOLVER" and self.game.START_KEY:
                self._save_and_return()

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
