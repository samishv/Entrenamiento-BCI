import os
import sys
import pygame

ENTORNO = "vscode"   # "vscode" | "spyder"

if ENTORNO == "vscode":
    try:
        _CARPETA_JUEGO = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _CARPETA_JUEGO = os.path.abspath(os.path.join(os.getcwd(), "Juego"))
    os.chdir(_CARPETA_JUEGO)            
    if _CARPETA_JUEGO not in sys.path:      
        sys.path.insert(0, _CARPETA_JUEGO)
        

class Menu:
    def __init__(self, game):
        self.game = game
        self.run_display = True
        self.state = None

        # ===== CONSTANTE DE DESPLAZAMIENTO VERTICAL (SIN ESTO NO CUADRAN LOS PX) =====
        self.offset_y = 5

    def _scale_pixel_surface(self, surface, scale):
        """Escala texto o imagen sin suavizado para conservar el pixel art."""
        scale = max(0.01, float(scale))
        width = max(1, round(surface.get_width() * scale))
        height = max(1, round(surface.get_height() * scale))
        return pygame.transform.scale(surface, (width, height))

    def _load_menu_image(self, image_path, image_size):
        """Carga una imagen del menu y evita cerrar el juego si no existe."""
        try:
            image = pygame.image.load(image_path).convert_alpha()
            return pygame.transform.scale(image, image_size)
        except (FileNotFoundError, pygame.error) as error:
            print(f"[MENU] No se pudo cargar la imagen {image_path}: {error}")
            return pygame.Surface(image_size, pygame.SRCALPHA)

    def _draw_secondary_title(self, text):
        """Dibuja un titulo secundario, dos lineas y un copo de nieve."""
        mid_w, _ = self.game.display.get_rect().center
        title = self.game.render_bitmap_text_surface(text, self.game.bitmap_font_white, spacing=self.game.bitmap_spacing,)
        title = self._scale_pixel_surface(title, self.TITLE_SCALE)
        self.game.display.blit(title, title.get_rect(center=(mid_w, self.TITLE_Y + self.offset_y)),)
        line_y = self.DECORATION_LINE_Y + self.offset_y
        pygame.draw.line(self.game.display, (255, 255, 255), (self.LEFT_LINE_START_X, line_y), (self.LEFT_LINE_END_X, line_y), self.DECORATION_LINE_WIDTH,)
        pygame.draw.line(self.game.display, (255, 255, 255), (self.RIGHT_LINE_START_X, line_y), (self.RIGHT_LINE_END_X, line_y), self.DECORATION_LINE_WIDTH,)
        snowflake_rect = self.snowflake_image.get_rect(center=(mid_w, line_y))
        self.game.display.blit(self.snowflake_image, snowflake_rect)

    def _draw_scaled_back_option(self, center_x):
        """Dibuja VOLVER con la escala propia del menu."""
        volver = self.game.render_bitmap_text_surface(
            "VOLVER", self.game.bitmap_font_white, spacing=self.game.bitmap_spacing,
        )
        volver = self._scale_pixel_surface(volver, self.BACK_TEXT_SCALE)
        volver_rect = volver.get_rect(center=(center_x, self.BACK_Y))
        self.game.display.blit(volver, volver_rect)
        return volver_rect

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

    # =========================================================
    # DISTRIBUCION VISUAL DEL MENU PRINCIPAL
    # =========================================================
    # --- TITULO
    TITLE_X = 240
    TITLE_Y = 28
    TITLE_SCALE = 2 

    # --- DECORACION DEBAJO DEL TITULO
    MOUNTAIN_X = 240
    MOUNTAIN_Y = 50
    MOUNTAIN_WIDTH = 20
    MOUNTAIN_HEIGHT = 20
    LEFT_LINE_START_X = 140
    LEFT_LINE_END_X = 225
    RIGHT_LINE_START_X = 255
    RIGHT_LINE_END_X = 340
    DECORATION_LINE_Y = 50
    DECORATION_LINE_WIDTH = 2

    # --- TARJETAS
    LEVELS_CARD_X = 140
    CONFIG_CARD_X = 340
    CARDS_Y = 85
    CARD_WIDTH = 130
    CARD_HEIGHT = 130
    CARD_BACKGROUND_COLOR = (50, 90, 120)
    CARD_BORDER_COLOR = (255, 255, 255)
    CARD_BORDER_WIDTH = 2

    # --- IMAGENES DENTRO DE LAS TARJETAS
    CARD_IMAGE_Y = 137
    CARD_IMAGE_WIDTH = 80
    CARD_IMAGE_HEIGHT = 80

    # --- NOMBRES DE LAS OPCIONES
    CARD_LABEL_Y = 195

    # --- INDICACIONES INFERIORES
    CONTROLS_TEXT_SCALE = 0.7
    CONTROLS_Y = 245
    MOVE_CONTROL_X = 200
    ACCEPT_CONTROL_X = 280

    # =========================================================
    # RUTAS DE LAS IMAGENES
    # =========================================================
    MAP_IMAGE_PATH = os.path.join("data", "images", "menu", "map.png")
    GEAR_IMAGE_PATH = os.path.join("data", "images", "menu", "gear.png")
    MOUNTAIN_IMAGE_PATH = os.path.join("data", "images", "menu", "snowflake.png")

    def __init__(self, game):
        super().__init__(game)
        self.options = ["NIVELES","CONFIG.",]
        self.index = 0

        self.map_image = self._load_menu_image(self.MAP_IMAGE_PATH,(self.CARD_IMAGE_WIDTH, self.CARD_IMAGE_HEIGHT),)
        self.gear_image = self._load_menu_image(self.GEAR_IMAGE_PATH,(self.CARD_IMAGE_WIDTH, self.CARD_IMAGE_HEIGHT),)
        self.mountain_image = self._load_menu_image(self.MOUNTAIN_IMAGE_PATH,(self.MOUNTAIN_WIDTH, self.MOUNTAIN_HEIGHT),)

    def _load_menu_image(self, image_path, image_size):
        """Carga y escala una imagen del menu."""
        try:
            image = pygame.image.load(image_path).convert_alpha()
            image = pygame.transform.scale(image, image_size,)
            return image

        except (FileNotFoundError, pygame.error) as error:
            print(f"[MENU] No se pudo cargar la imagen {image_path}: {error}")
            return pygame.Surface(image_size, pygame.SRCALPHA,)

    def _scale_pixel_surface(self, surface, scale):
        """Escala una superficie sin utilizar suavizado."""
        scale = max(0.01, float(scale))
        scaled_width = max(1, round(surface.get_width() * scale))
        scaled_height = max(1, round(surface.get_height() * scale))
        return pygame.transform.scale(surface, (scaled_width, scaled_height))

    def _get_card_rect(self, center_x):
        """Devuelve el rectángulo de una tarjeta a partir de su centro horizontal."""
        return pygame.Rect(center_x - self.CARD_WIDTH // 2, self.CARDS_Y, self.CARD_WIDTH, self.CARD_HEIGHT)

    def _draw_title(self):
        """Dibuja RAPPINGU al doble sin aplicar suavizado."""
        title = self.game.render_bitmap_text_surface(
            "RAPPINGU", self.game.bitmap_font_white, spacing=self.game.bitmap_spacing
        )
        title = self._scale_pixel_surface(title, self.TITLE_SCALE)
        title_rect = title.get_rect(center=(self.TITLE_X, self.TITLE_Y + self.offset_y))
        self.game.display.blit(title, title_rect)

    def _draw_title_decoration(self):
        """Dibuja: -------- [montaña] --------"""
        line_y = self.DECORATION_LINE_Y + self.offset_y
        pygame.draw.line(self.game.display, (255, 255, 255), (self.LEFT_LINE_START_X, line_y), (self.LEFT_LINE_END_X, line_y), self.DECORATION_LINE_WIDTH)
        pygame.draw.line(self.game.display, (255, 255, 255), (self.RIGHT_LINE_START_X, line_y), (self.RIGHT_LINE_END_X, line_y), self.DECORATION_LINE_WIDTH)
        mountain_rect = self.mountain_image.get_rect(center=(self.MOUNTAIN_X, self.MOUNTAIN_Y + self.offset_y))
        self.game.display.blit(self.mountain_image, mountain_rect)

    def _draw_card(self, center_x, image, label_text):
        """Dibuja una tarjeta completa (Fondo, Icono, Nombre, Borde)."""
        card_rect = self._get_card_rect(center_x)
        pygame.draw.rect(self.game.display, self.CARD_BACKGROUND_COLOR, card_rect)
        image_rect = image.get_rect(center=(center_x, self.CARD_IMAGE_Y))
        self.game.display.blit(image, image_rect)
        label = self.game.render_bitmap_text_surface(
            label_text, self.game.bitmap_font_white, spacing=self.game.bitmap_spacing
        )
        label_rect = label.get_rect(center=(center_x, self.CARD_LABEL_Y))
        self.game.display.blit(label, label_rect)
        pygame.draw.rect(self.game.display, self.CARD_BORDER_COLOR, card_rect, self.CARD_BORDER_WIDTH)
        return card_rect.union(label_rect)

    def _draw_controls(self):
        """Dibuja las indicaciones inferiores."""
        move_text = self.game.render_bitmap_text_surface("<> MOVER", self.game.bitmap_font_white, spacing=self.game.bitmap_spacing)
        accept_text = self.game.render_bitmap_text_surface("ENTER", self.game.bitmap_font_white, spacing=self.game.bitmap_spacing)

        move_text = self._scale_pixel_surface(move_text, self.CONTROLS_TEXT_SCALE)
        accept_text = self._scale_pixel_surface(accept_text, self.CONTROLS_TEXT_SCALE)

        move_rect = move_text.get_rect(center=(self.MOVE_CONTROL_X, self.CONTROLS_Y))
        accept_rect = accept_text.get_rect(center=(self.ACCEPT_CONTROL_X, self.CONTROLS_Y))

        self.game.display.blit(move_text, move_rect)
        self.game.display.blit(accept_text, accept_rect)

    def display_menu(self):
        self.run_display = True
        while self.run_display:
            self.game.check_events()
            self.check_input()

            self.game.display.blit(self.game.menu_bg, (0, 0))
            self._draw_title()
            self._draw_title_decoration()

            levels_option_rect = self._draw_card(center_x=self.LEVELS_CARD_X, image=self.map_image, label_text="NIVELES")
            config_option_rect = self._draw_card(center_x=self.CONFIG_CARD_X, image=self.gear_image, label_text="CONFIG.")

            option_rects = [levels_option_rect, config_option_rect]
            self.draw_cursor(option_rects[self.index])
            self._draw_controls()
            self.blit_screen()

    def check_input(self):
        """ Distribución horizontal: NIVELES  <---->  CONFIGURACION """
        if self.game.RIGHT_KEY:
            self.index = 1
        elif self.game.LEFT_KEY:
            self.index = 0
        elif self.game.DOWN_KEY or self.game.UP_KEY:
            self.index = 1 - self.index

        if self.game.START_KEY:
            choice = self.options[self.index]

            if choice == "NIVELES":
                self.game.curr_menu = self.game.levels_menu
                self.run_display = False
            elif choice == "CONFIG.":
                self.game.curr_menu = self.game.config_menu
                self.run_display = False


# ===== VENTANA MENU DE CONFIG =====
class ConfigMenu(Menu):
    # ===== DISTRIBUCION VISUAL DEL MENU DE CONFIGURACION =====
    TITLE_Y = 20
    TITLE_SCALE = 1.5
    DECORATION_LINE_Y = 40
    DECORATION_LINE_WIDTH = 2
    LEFT_LINE_START_X = 115
    LEFT_LINE_END_X = 225
    RIGHT_LINE_START_X = 255
    RIGHT_LINE_END_X = 365
    SNOWFLAKE_SIZE = 16
    SNOWFLAKE_IMAGE_PATH = os.path.join("data", "images", "menu", "snowflake.png")
    NICKNAME_Y = 82
    NICKNAME_LINE = (280, 440)
    OPTIONS_X = 120
    BT_Y = 132
    MODE_Y = 172
    COLOR_Y = 212
    BACK_TEXT_SCALE = 0.7
    BACK_Y = 250

    # =========================================================
    # TARJETA DE PREVISUALIZACION
    # =========================================================
    PREVIEW_RECT = pygame.Rect(300, 107, 120, 120)
    COLOR_PREVIEW_FRAME_MS = 300
    HAND_PREVIEW_FRAME_MS = 180
    ARIADNE_PREVIEW_FRAME_MS = 300

    # =========================================================
    # ESCALAS DE LAS ANIMACIONES
    # =========================================================
    COLOR_PREVIEW_SCALE = 2.0
    HAND_PREVIEW_SCALE = 1.0
    ARIADNE_PREVIEW_SCALE = 1.0
    HAND_RIGHT_PREVIEW_FOLDER = os.path.join("data", "images", "entities", "hand")
    HAND_LEFT_PREVIEW_FOLDER = os.path.join("data", "images", "entities", "hand_left")
    ARIADNE_PREVIEW_FOLDER = os.path.join("data", "images", "entities", "disp_vib")

    def __init__(self, game):
        super().__init__(game)
        self.state = "VOLVER"
        self.nickname_buffer = ""
        self.max_nickname_len = 20
        self.cursor_blink_ms = 500
        self.snowflake_image = self._load_menu_image(self.SNOWFLAKE_IMAGE_PATH,(self.SNOWFLAKE_SIZE, self.SNOWFLAKE_SIZE),)

        # =========================================================
        # ESTADO DE LA PREVISUALIZACION
        # =========================================================
        self.preview_state = "COLOR"
        self.preview_animation_key = None
        self.preview_animation_start_ms = pygame.time.get_ticks()
        self.preview_color = None
        self.preview_idle_frames = []
        self.preview_external_frames_loaded = False
        self.preview_hand_right_frames = []
        self.preview_hand_left_frames = []
        self.preview_ariadne_frames = []

    def display_menu(self):
        self.run_display = True
        pygame.key.start_text_input()
        pygame.key.set_repeat(300, 35)
        self.nickname_buffer = self.game.gamer_tag

        while self.run_display:
            self.game.check_events()
            if self.game.TEXT_INPUT:
                self.nickname_buffer += self.game.TEXT_INPUT
            if self.game.BACKSPACE and len(self.nickname_buffer) > 0:
                self.nickname_buffer = self.nickname_buffer[:-1]
            allowed = set(self.game.bitmap_charset)
            self.nickname_buffer = "".join(ch for ch in self.nickname_buffer if ch in allowed)
            self.nickname_buffer = self.nickname_buffer[:self.max_nickname_len]
            self.check_input()
            self._update_preview_state()
            self.game.display.blit(self.game.menu_bg, (0, 0))
            mid_w, _ = self.game.display.get_rect().center

            self._draw_secondary_title("CONFIGURACION")

            nick_label = self.game.render_bitmap_text_surface("NICKNAME", self.game.bitmap_font_white, spacing=self.game.bitmap_spacing,)
            self.game.display.blit(nick_label, nick_label.get_rect(center=(self.OPTIONS_X, self.NICKNAME_Y-1)),)
            self._draw_nickname_field()

            bt_rect = self._draw_bt_switch(self.OPTIONS_X, self.BT_Y)
            mode_rect = self._draw_hand_mode(self.OPTIONS_X, self.MODE_Y)
            color_rect = self._draw_player_color(self.OPTIONS_X, self.COLOR_Y)

            self._draw_player_preview()

            volver_rect = self._draw_scaled_back_option(mid_w)
            option_rects = {"BT": bt_rect, "MODO": mode_rect, "COLOR": color_rect, "VOLVER": volver_rect,}
            self.draw_cursor(option_rects[self.state])
            self.blit_screen()

        pygame.key.stop_text_input()
        pygame.key.set_repeat()

    def _draw_nickname_field(self):
        """Dibuja el nickname sobre una linea, como en el boceto."""
        line_left, line_right = self.NICKNAME_LINE
        y = self.NICKNAME_Y
        pygame.draw.line(self.game.display, (255, 255, 255), (line_left, y + 9), (line_right, y + 9), 1,)

        if self.nickname_buffer:
            nickname = self.game.render_bitmap_text_surface(self.nickname_buffer, self.game.bitmap_font_white, spacing=self.game.bitmap_spacing,)
            nickname_rect = nickname.get_rect(midbottom=((line_left + line_right) // 2, y + 7))
            self.game.display.blit(nickname, nickname_rect)
        else:
            nickname_rect = None

        if (pygame.time.get_ticks() // self.cursor_blink_ms) % 2 == 0:
            caret = self.game.render_bitmap_text_surface("|", self.game.bitmap_font_white, spacing=self.game.bitmap_spacing,)
            if nickname_rect:
                caret_rect = caret.get_rect(midleft=(nickname_rect.right + 2, nickname_rect.centery))
            else:
                caret_rect = caret.get_rect(center=((line_left + line_right) // 2, y-3))
            self.game.display.blit(caret, caret_rect)

    def _load_preview_folder(self, folder_path):
        """ Carga todas las imagenes PNG """
        if not os.path.isdir(folder_path):
            print("[MENU] No se encontro la carpeta de previsualizacion:",folder_path,)
            return []

        image_names = sorted(file_name for file_name in os.listdir(folder_path) if file_name.lower().endswith(".png"))
        frames = []

        for image_name in image_names:
            image_path = os.path.join(folder_path, image_name)
            try:
                frame = pygame.image.load(image_path).convert_alpha()
                frames.append(frame)
            except pygame.error as error:
                print(f"[MENU] No se pudo cargar {image_path}: {error}")
        return frames


    def _load_external_preview_frames(self):
        """ Carga una sola vez las animaciones de las manos y ARIADNE. """
        if self.preview_external_frames_loaded:
            return

        self.preview_hand_right_frames = self._load_preview_folder(self.HAND_RIGHT_PREVIEW_FOLDER)
        self.preview_hand_left_frames = self._load_preview_folder(self.HAND_LEFT_PREVIEW_FOLDER)
        self.preview_ariadne_frames = self._load_preview_folder(self.ARIADNE_PREVIEW_FOLDER)
        self.preview_external_frames_loaded = True

    def _update_preview_state(self):
        """ Actualiza el contenido de la tarjeta segun la opcion """
        if self.state in ("BT", "MODO", "COLOR"):
            self.preview_state = self.state

    def _get_current_preview_animation(self):
        """ Devuelve: animation_key, frames, frame_duration_ms, preview_scale """
        if self.preview_state == "COLOR":
            self._refresh_preview_frames()
            animation_key = ("COLOR", self.game.player_color,)
            return (animation_key, self.preview_idle_frames, self.COLOR_PREVIEW_FRAME_MS, self.COLOR_PREVIEW_SCALE,)

        if self.preview_state == "MODO":
            if self.game.hand_mode == "ZURDO":
                animation_key = ("MODO", "ZURDO")
                frames = self.preview_hand_left_frames
            else:
                animation_key = ("MODO", "DIESTRO")
                frames = self.preview_hand_right_frames
            return (animation_key, frames, self.HAND_PREVIEW_FRAME_MS, self.HAND_PREVIEW_SCALE,)

        if self.preview_state == "BT":
            animation_key = ("BT",self.game.bluetooth_enabled,)
            return (animation_key, self.preview_ariadne_frames, self.ARIADNE_PREVIEW_FRAME_MS, self.ARIADNE_PREVIEW_SCALE,)

        self._refresh_preview_frames()
        return (("COLOR", self.game.player_color),self.preview_idle_frames, self.COLOR_PREVIEW_FRAME_MS, self.COLOR_PREVIEW_SCALE,)

    def _refresh_preview_frames(self):
        """Recolorea el idle solo cuando cambia el color seleccionado."""
        if self.preview_color == self.game.player_color and self.preview_idle_frames:
            return

        self.preview_idle_frames = self.game._recolor_player_images(self.game.original_player_images["idle"],self.game.player_color,)
        self.preview_color = self.game.player_color

    def _draw_player_preview(self):
        """Dibuja la tarjeta de previsualizacion."""
        preview_rect = self.PREVIEW_RECT
        self._load_external_preview_frames()
        preview_bg = pygame.Surface(preview_rect.size, pygame.SRCALPHA)
        preview_bg.fill((50, 90, 120, 255))
        self.game.display.blit(preview_bg, preview_rect.topleft)
        pygame.draw.rect(self.game.display, (255, 255, 255), preview_rect, 2)
        animation_key, frames, frame_duration_ms, preview_scale = self._get_current_preview_animation()
        
        if animation_key != self.preview_animation_key:
            self.preview_animation_key = animation_key
            self.preview_animation_start_ms = pygame.time.get_ticks()
            
        if not frames:
            return
            
        frame_duration_ms = max(1, frame_duration_ms)
        elapsed_ms = pygame.time.get_ticks() - self.preview_animation_start_ms
        frame_index = (elapsed_ms // frame_duration_ms) % len(frames)
        frame = frames[frame_index]
        frame_width = frame.get_width()
        frame_height = frame.get_height()
        
        if frame_width <= 0 or frame_height <= 0:
            return
            
        preview_scale = max(0.01, preview_scale)
        scaled_width = max(1, round(frame_width * preview_scale))
        scaled_height = max(1, round(frame_height * preview_scale))
        preview_frame = pygame.transform.scale(frame, (scaled_width, scaled_height))
        frame_rect = preview_frame.get_rect(center=preview_rect.center)
        self.game.display.blit(preview_frame, frame_rect)

    def _draw_bt_switch(self, x, y):
        """Dibuja ARIADNE junto con su estado ON/OFF."""
        enabled = self.game.bluetooth_enabled
        text_str = "ON" if enabled else "OFF"
        box_color = (0, 200, 80) if enabled else (0, 0, 0)

        label = self.game.render_bitmap_text_surface("ARIADNE", self.game.bitmap_font_white, spacing=self.game.bitmap_spacing)
        switch_text = self.game.render_bitmap_text_surface(text_str, self.game.bitmap_font_white, spacing=self.game.bitmap_spacing)

        spacing_between = 4
        padding_x, padding_y = 4, 2
        
        box_w = switch_text.get_width() + (padding_x * 2)
        box_h = switch_text.get_height() + (padding_y * 2)
        total_width = label.get_width() + spacing_between + box_w
        start_x = x - total_width // 2

        label_rect = label.get_rect(midleft=(start_x, y))
        box_rect = pygame.Rect(label_rect.right + spacing_between, y - box_h // 2, box_w, box_h)
        
        self.game.display.blit(label, label_rect)
        pygame.draw.rect(self.game.display, box_color, box_rect, border_radius=3)
        self.game.display.blit(switch_text, switch_text.get_rect(center=box_rect.center))
        
        return label_rect.union(box_rect)

    def _draw_hand_mode(self, x, y):
        """Dibuja unicamente DIESTRO o ZURDO."""
        mode_surface = self.game.render_bitmap_text_surface(self.game.hand_mode,self.game.bitmap_font_white, spacing=self.game.bitmap_spacing,)
        mode_rect = mode_surface.get_rect(center=(x, y))
        self.game.display.blit(mode_surface, mode_rect)
        return mode_rect

    def _draw_player_color(self, x, y):
        """Dibuja el nombre del color sobre su cuadro de color."""
        color_surface = self.game.render_bitmap_text_surface(self.game.player_color, self.game.bitmap_font_white, spacing=self.game.bitmap_spacing,)
        padding_x, padding_y = 4, 2
        color_box_rect = pygame.Rect(
            x - (color_surface.get_width() + padding_x * 2) // 2,
            y - (color_surface.get_height() + padding_y * 2) // 2,
            color_surface.get_width() + padding_x * 2,
            color_surface.get_height() + padding_y * 2,
        )
        pygame.draw.rect(self.game.display, self.game.get_player_menu_color(), color_box_rect, border_radius=3,)
        self.game.display.blit(color_surface, color_surface.get_rect(center=color_box_rect.center),)
        return color_box_rect

    def check_input(self):
        """ Navegacion vertical: ARIADNE -> MODO -> COLOR -> VOLVER """
        options = ["BT", "MODO", "COLOR", "VOLVER"]
        current_index = options.index(self.state)

        if self.game.DOWN_KEY:
            self.state = options[(current_index + 1) % len(options)]
        elif self.game.UP_KEY:
            self.state = options[(current_index - 1) % len(options)]

        if self.state == "BT":
            if (self.game.LEFT_KEY or self.game.RIGHT_KEY or self.game.START_KEY):
                self.game.bluetooth_enabled = not self.game.bluetooth_enabled
                
        elif self.state == "MODO":
            if (self.game.LEFT_KEY or self.game.RIGHT_KEY or self.game.START_KEY):
                self.game.toggle_hand_mode()
                
        elif self.state == "COLOR":
            if self.game.RIGHT_KEY or self.game.START_KEY:
                self.game.next_player_color()
            elif self.game.LEFT_KEY:
                self.game.previous_player_color()
                
        elif self.state == "VOLVER":
            if self.game.START_KEY:
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

    # =========================================================
    # DISTRIBUCION VISUAL DEL MENU DE NIVELES
    # =========================================================
    TITLE_Y = 20
    TITLE_SCALE = 1.5
    DECORATION_LINE_Y = 40
    DECORATION_LINE_WIDTH = 2
    LEFT_LINE_START_X = 175
    LEFT_LINE_END_X = 225
    RIGHT_LINE_START_X = 255
    RIGHT_LINE_END_X = 305
    SNOWFLAKE_SIZE = 16
    SNOWFLAKE_IMAGE_PATH = os.path.join("data", "images", "menu", "snowflake.png")
    LEVEL_1_X = 120
    LEVEL_2_X = 360
    LEVEL_CARDS_Y = 79
    LEVEL_CARD_WIDTH = 180
    LEVEL_CARD_HEIGHT = 120
    LEVEL_CARD_PADDING = 5
    LEVEL_LABEL_Y = 212
    BACK_TEXT_SCALE = 0.7
    BACK_Y = 250
    CARD_BORDER_WIDTH = 2
    LEVEL_1_BASE_COLOR = (50, 90, 120)
    LEVEL_2_BASE_COLOR = (50, 90, 120)

    def __init__(self, game):
        super().__init__(game)
        self.options = ["NIVEL 1", "NIVEL 2", "VOLVER"]
        self.index = 0
        self.snowflake_image = self._load_menu_image(self.SNOWFLAKE_IMAGE_PATH,(self.SNOWFLAKE_SIZE, self.SNOWFLAKE_SIZE),)
        self.level_preview_layers = {
            1: self._load_level_preview_layers(1),
            2: self._load_level_preview_layers(2),
        }

    def _load_level_preview_layers(self, level_id):
        """ Carga las cuatro imágenes correspondientes al nivel indicado. """
        folder_name = f"level{level_id}"
        folder_path = os.path.join("data", "images", "background", folder_name,)
        inner_width = self.LEVEL_CARD_WIDTH - (self.LEVEL_CARD_PADDING * 2)
        inner_height = self.LEVEL_CARD_HEIGHT - (self.LEVEL_CARD_PADDING * 2)
        layers = []

        for image_number in range(1, 5):
            image_path = os.path.join(folder_path,f"{image_number:02d}.png",)
            image = pygame.image.load(image_path).convert_alpha()
            image = pygame.transform.scale(image,(inner_width, inner_height),)
            layers.append(image)
        return layers

    def _get_level_card_rect(self, center_x):
        """ Devuelve el rectángulo completo de una tarjeta """
        return pygame.Rect(center_x - self.LEVEL_CARD_WIDTH // 2, self.LEVEL_CARDS_Y, self.LEVEL_CARD_WIDTH, self.LEVEL_CARD_HEIGHT,)

    def _draw_level_card(self, level_id, center_x, base_color):
        """ Dibuja una tarjeta de nivel """
        card_rect = self._get_level_card_rect(center_x)
        inner_rect = card_rect.inflate(-(self.LEVEL_CARD_PADDING * 2), -(self.LEVEL_CARD_PADDING * 2))
        pygame.draw.rect(self.game.display, base_color, card_rect)

        for layer in self.level_preview_layers[level_id]:
            self.game.display.blit(layer, inner_rect.topleft)

        pygame.draw.rect(self.game.display, (255, 255, 255), card_rect, self.CARD_BORDER_WIDTH)
        return card_rect

    def _draw_level_label(self, text, center_x):
        """ Dibuja el nombre del nivel debajo de su tarjeta. """
        label = self.game.render_bitmap_text_surface(text, self.game.bitmap_font_white, spacing=self.game.bitmap_spacing,)
        label_rect = label.get_rect(center=(center_x, self.LEVEL_LABEL_Y))
        self.game.display.blit(label, label_rect)
        return label_rect

    def display_menu(self):
        self.run_display = True
        
        while self.run_display:
            self.game.check_events()
            self.check_input()
            self.game.display.blit(self.game.menu_bg, (0, 0))
            mid_w, _ = self.game.display.get_rect().center

            # --- TITULO
            self._draw_secondary_title("NIVELES")

            # --- TARJETAS DE LOS NIVELES
            level_1_card_rect = self._draw_level_card(level_id=1,center_x=self.LEVEL_1_X,base_color=self.LEVEL_1_BASE_COLOR,)
            level_2_card_rect = self._draw_level_card(level_id=2,center_x=self.LEVEL_2_X,base_color=self.LEVEL_2_BASE_COLOR,)

            # --- NOMBRES DE LOS NIVELES
            level_1_label_rect = self._draw_level_label("NIVEL 1",self.LEVEL_1_X,)
            level_2_label_rect = self._draw_level_label("NIVEL 2",self.LEVEL_2_X,)

            # --- OPCION VOLVER
            volver_rect = self._draw_scaled_back_option(mid_w)

            # --- AREAS UTILIZADAS POR EL CURSOR
            level_1_option_rect = level_1_card_rect.union(level_1_label_rect)
            level_2_option_rect = level_2_card_rect.union(level_2_label_rect)
            option_rects = [level_1_option_rect,level_2_option_rect,volver_rect,]
            
            self.draw_cursor(option_rects[self.index])
            self.blit_screen()

    def check_input(self):
        """ Distribucion de las opciones: NIVEL 1  <-->  NIVEL 2 y VOLVER """

        # --- NAVEGACION VERTICAL
        if self.game.DOWN_KEY:
            if self.index == 0 or self.index == 1:
                self.index = 2
        elif self.game.UP_KEY:
            if self.index == 2:
                self.index = 0

        # --- NAVEGACION HORIZONTAL
        elif self.game.RIGHT_KEY:
            if self.index == 0:
                self.index = 1
            elif self.index == 1:
                self.index = 0
        elif self.game.LEFT_KEY:
            if self.index == 1:
                self.index = 0
            elif self.index == 0:
                self.index = 1

        # --- REGRESAR CON ESC
        if self.game.BACK_KEY:
            self.game.curr_menu = self.game.main_menu
            self.run_display = False
            return

        # ---SELECCIONAR CON ENTER
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