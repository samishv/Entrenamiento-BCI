import pygame
import os

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

    # =========================================================
    # TARJETA DE PREVISUALIZACION
    # =========================================================
    
    PREVIEW_RECT = pygame.Rect(300, 100, 120, 120)
    
    # Duracion de cada frame, expresada en milisegundos.
    # Un valor menor produce una animacion mas rapida.
    COLOR_PREVIEW_FRAME_MS = 300
    HAND_PREVIEW_FRAME_MS = 150
    ARIADNE_PREVIEW_FRAME_MS = 300
    
    # =========================================================
    # ESCALAS DE LAS ANIMACIONES
    # =========================================================
    #
    # 1.0 = tamaño original
    # 2.0 = doble del tamaño original
    # 0.5 = mitad del tamaño original
    
    COLOR_PREVIEW_SCALE = 2.0
    HAND_PREVIEW_SCALE = 1.0
    ARIADNE_PREVIEW_SCALE = 1.0
    
    # Carpetas de las animaciones.
    HAND_RIGHT_PREVIEW_FOLDER = os.path.join(
        "data", "images", "entities", "hand"
    )
    
    HAND_LEFT_PREVIEW_FOLDER = os.path.join(
        "data", "images", "entities", "hand_left"
    )
    
    ARIADNE_PREVIEW_FOLDER = os.path.join(
        "data", "images", "entities", "disp_vib"
    )

    def __init__(self, game):
        super().__init__(game)
        self.state = "VOLVER"  # opciones: "BT", "MODO", "COLOR" o "VOLVER"
        self.nickname_buffer = ""
        self.max_nickname_len = 20
        self.cursor_blink_ms = 500

        # =========================================================
        # ESTADO DE LA PREVISUALIZACION
        # =========================================================
        
        # La vista previa comienza mostrando el personaje.
        self.preview_state = "COLOR"
        
        # Permite reiniciar la animacion cuando cambia el contenido
        # mostrado en la tarjeta.
        self.preview_animation_key = None
        self.preview_animation_start_ms = pygame.time.get_ticks()
        
        # Cache del personaje recoloreado.
        self.preview_color = None
        self.preview_idle_frames = []
        
        # Las animaciones se cargan de manera diferida porque ConfigMenu
        # se construye antes de que Game termine de cargar todos sus assets.
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
            self._update_preview_state()
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
            
    def _load_preview_folder(self, folder_path):
        """
        Carga todas las imagenes PNG de una carpeta y devuelve
        una lista de superficies ordenadas por nombre.
    
        Ejemplo:
            01.png
            02.png
            03.png
            04.png
        """
        if not os.path.isdir(folder_path):
            print(
                "[MENU] No se encontro la carpeta de previsualizacion:",
                folder_path,
            )
            return []
    
        image_names = sorted(
            file_name
            for file_name in os.listdir(folder_path)
            if file_name.lower().endswith(".png")
        )
    
        frames = []
    
        for image_name in image_names:
            image_path = os.path.join(folder_path, image_name)
    
            try:
                frame = pygame.image.load(image_path).convert_alpha()
                frames.append(frame)
    
            except pygame.error as error:
                print(
                    f"[MENU] No se pudo cargar {image_path}: {error}"
                )
    
        return frames
    
    
    def _load_external_preview_frames(self):
        """
        Carga una sola vez las animaciones de las manos y ARIADNE.
    
        La carga se hace al abrir el menu, no durante __init__,
        porque ConfigMenu se crea antes de que Game termine de
        inicializar todos sus recursos.
        """
        if self.preview_external_frames_loaded:
            return
    
        self.preview_hand_right_frames = self._load_preview_folder(
            self.HAND_RIGHT_PREVIEW_FOLDER
        )
    
        self.preview_hand_left_frames = self._load_preview_folder(
            self.HAND_LEFT_PREVIEW_FOLDER
        )
    
        self.preview_ariadne_frames = self._load_preview_folder(
            self.ARIADNE_PREVIEW_FOLDER
        )
    
        self.preview_external_frames_loaded = True
        
    def _update_preview_state(self):
        """
        Actualiza el contenido de la tarjeta segun la opcion
        seleccionada por el cursor.
    
        VOLVER no modifica preview_state, por lo que conserva
        la ultima animacion seleccionada.
        """
        if self.state in ("BT", "MODO", "COLOR"):
            self.preview_state = self.state
            
    def _get_current_preview_animation(self):
        """
        Devuelve:
    
            animation_key
            frames
            frame_duration_ms
            preview_scale
    
        preview_scale controla directamente el tamaño de cada
        tipo de animacion:
    
            1.0 = tamaño original
            2.0 = doble del tamaño original
            0.5 = mitad del tamaño original
        """
    
        # ---------------------------------------------------------
        # COLOR DEL PERSONAJE
        # ---------------------------------------------------------
        if self.preview_state == "COLOR":
            self._refresh_preview_frames()
    
            animation_key = (
                "COLOR",
                self.game.player_color,
            )
    
            return (
                animation_key,
                self.preview_idle_frames,
                self.COLOR_PREVIEW_FRAME_MS,
                self.COLOR_PREVIEW_SCALE,
            )
    
        # ---------------------------------------------------------
        # MODO DE MANO
        # ---------------------------------------------------------
        if self.preview_state == "MODO":
            if self.game.hand_mode == "ZURDO":
                animation_key = ("MODO", "ZURDO")
                frames = self.preview_hand_left_frames
            else:
                animation_key = ("MODO", "DIESTRO")
                frames = self.preview_hand_right_frames
    
            return (
                animation_key,
                frames,
                self.HAND_PREVIEW_FRAME_MS,
                self.HAND_PREVIEW_SCALE,
            )
    
        # ---------------------------------------------------------
        # ARIADNE
        # ---------------------------------------------------------
        if self.preview_state == "BT":
            animation_key = (
                "BT",
                self.game.bluetooth_enabled,
            )
    
            return (
                animation_key,
                self.preview_ariadne_frames,
                self.ARIADNE_PREVIEW_FRAME_MS,
                self.ARIADNE_PREVIEW_SCALE,
            )
    
        # ---------------------------------------------------------
        # RESPALDO
        # ---------------------------------------------------------
        self._refresh_preview_frames()
    
        return (
            ("COLOR", self.game.player_color),
            self.preview_idle_frames,
            self.COLOR_PREVIEW_FRAME_MS,
            self.COLOR_PREVIEW_SCALE,
        )

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
        """
        Dibuja la tarjeta de previsualizacion y reproduce una
        animacion distinta segun la opcion seleccionada:
    
            BT     -> ARIADNE
            MODO   -> mano derecha o izquierda
            COLOR  -> personaje idle recoloreado
            VOLVER -> conserva la ultima seleccion
        """
        preview_rect = self.PREVIEW_RECT
    
        # Cargar las animaciones externas solo una vez.
        self._load_external_preview_frames()
    
        # Fondo solido de la tarjeta.
        preview_bg = pygame.Surface(
            preview_rect.size,
            pygame.SRCALPHA,
        )
        preview_bg.fill((73, 103, 141, 255))
    
        self.game.display.blit(
            preview_bg,
            preview_rect.topleft,
        )
    
        # Borde exterior.
        pygame.draw.rect(
            self.game.display,
            (255, 255, 255),
            preview_rect,
            2,
        )
    
        (
            animation_key,
            frames,
            frame_duration_ms,
            preview_scale,
        ) = self._get_current_preview_animation()
    
        # Si cambia la animacion, se reinicia desde el primer frame.
        if animation_key != self.preview_animation_key:
            self.preview_animation_key = animation_key
            self.preview_animation_start_ms = pygame.time.get_ticks()
    
        if not frames:
            return
    
        # Evitar una division entre cero si se coloca accidentalmente
        # una duracion igual o menor que cero.
        frame_duration_ms = max(1, frame_duration_ms)
    
        elapsed_ms = (
            pygame.time.get_ticks()
            - self.preview_animation_start_ms
        )
    
        frame_index = (
            elapsed_ms // frame_duration_ms
        ) % len(frames)
    
        frame = frames[frame_index]
    
        # Dimensiones originales del frame.
        frame_width = frame.get_width()
        frame_height = frame.get_height()
        
        if frame_width <= 0 or frame_height <= 0:
            return
        
        # Evita escalas iguales o menores que cero.
        preview_scale = max(0.01, preview_scale)
        
        # Calcula el nuevo tamaño usando la escala decimal.
        scaled_width = max(
            1,
            round(frame_width * preview_scale),
        )
        
        scaled_height = max(
            1,
            round(frame_height * preview_scale),
        )
        
        # Escala la imagen al tamaño calculado.
        preview_frame = pygame.transform.scale(
            frame,
            (
                scaled_width,
                scaled_height,
            ),
        )
    
        frame_rect = preview_frame.get_rect(
            center=preview_rect.center
        )
    
        self.game.display.blit(
            preview_frame,
            frame_rect,
        )


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
        Navegacion vertical:
        ARIADNE
           |
          MODO
           |
          COLOR
           |
         VOLVER
    
        En COLOR:
        - DERECHA avanza al siguiente color.
        - IZQUIERDA regresa al color anterior.
        - ENTER avanza al siguiente color.
        """
    
        options = ["BT", "MODO", "COLOR", "VOLVER"]
        current_index = options.index(self.state)
    
        # Navegacion vertical entre opciones
        if self.game.DOWN_KEY:
            self.state = options[(current_index + 1) % len(options)]
    
        elif self.game.UP_KEY:
            self.state = options[(current_index - 1) % len(options)]
    
        # Acciones de la opcion seleccionada
        if self.state == "BT":
            if (
                self.game.LEFT_KEY
                or self.game.RIGHT_KEY
                or self.game.START_KEY
            ):
                self.game.bluetooth_enabled = not self.game.bluetooth_enabled
    
        elif self.state == "MODO":
            if (
                self.game.LEFT_KEY
                or self.game.RIGHT_KEY
                or self.game.START_KEY
            ):
                self.game.toggle_hand_mode()
    
        elif self.state == "COLOR":
            if self.game.RIGHT_KEY or self.game.START_KEY:
                self.game.next_player_color()
    
            elif self.game.LEFT_KEY:
                self.game.previous_player_color()
    
        elif self.state == "VOLVER":
            if self.game.START_KEY:
                self._save_and_return()
    
        # ESC también permite regresar
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

    # Posicion vertical del titulo principal.
    TITLE_Y = 20

    # Centro horizontal de cada tarjeta.
    LEVEL_1_X = 120
    LEVEL_2_X = 360

    # Posicion vertical superior de ambas tarjetas.
    LEVEL_CARDS_Y = 70

    # Tamaño completo de las tarjetas.
    LEVEL_CARD_WIDTH = 180
    LEVEL_CARD_HEIGHT = 120

    # Espacio entre el borde de la tarjeta y la imagen.
    LEVEL_CARD_PADDING = 5

    # Posicion vertical de los nombres de los niveles.
    LEVEL_LABEL_Y = 210

    # Posicion vertical de la opcion VOLVER.
    BACK_Y = 250

    # Grosor del borde blanco que rodea cada tarjeta.
    CARD_BORDER_WIDTH = 2

    # Colores sólidos de fondo.
    # Estos colores se muestran debajo de las cuatro capas.
    LEVEL_1_BASE_COLOR = (55, 73, 103)
    LEVEL_2_BASE_COLOR = (55, 73, 103)

    def __init__(self, game):
        super().__init__(game)

        self.options = ["NIVEL 1", "NIVEL 2", "VOLVER"]
        self.index = 0

        # Las cuatro capas de cada nivel se cargan una sola vez.
        self.level_preview_layers = {
            1: self._load_level_preview_layers(1),
            2: self._load_level_preview_layers(2),
        }

    def _load_level_preview_layers(self, level_id):
        """
        Carga las cuatro imágenes 01.png, 02.png, 03.png y 04.png
        correspondientes al nivel indicado.

        Las imágenes preservan su transparencia y se escalan al área
        interior de la tarjeta.
        """
        folder_name = f"level{level_id}"

        folder_path = os.path.join(
            "data",
            "images",
            "background",
            folder_name,
        )

        inner_width = self.LEVEL_CARD_WIDTH - (
            self.LEVEL_CARD_PADDING * 2
        )
        inner_height = self.LEVEL_CARD_HEIGHT - (
            self.LEVEL_CARD_PADDING * 2
        )

        layers = []

        for image_number in range(1, 5):
            image_path = os.path.join(
                folder_path,
                f"{image_number:02d}.png",
            )

            image = pygame.image.load(image_path).convert_alpha()

            # Se utiliza scale en lugar de smoothscale para conservar
            # mejor el aspecto del pixel art.
            image = pygame.transform.scale(
                image,
                (inner_width, inner_height),
            )

            layers.append(image)

        return layers

    def _get_level_card_rect(self, center_x):
        """
        Devuelve el rectángulo completo de una tarjeta tomando
        como referencia su centro horizontal.
        """
        return pygame.Rect(
            center_x - self.LEVEL_CARD_WIDTH // 2,
            self.LEVEL_CARDS_Y,
            self.LEVEL_CARD_WIDTH,
            self.LEVEL_CARD_HEIGHT,
        )

    def _draw_level_card(self, level_id, center_x, base_color):
        """
        Dibuja una tarjeta de nivel:

        1. Rectángulo de color sólido.
        2. Las cuatro capas del fondo.
        3. Un borde blanco.
        """
        card_rect = self._get_level_card_rect(center_x)

        inner_rect = card_rect.inflate(
            -(self.LEVEL_CARD_PADDING * 2),
            -(self.LEVEL_CARD_PADDING * 2),
        )

        # Base de color sólido.
        pygame.draw.rect(
            self.game.display,
            base_color,
            card_rect,
        )

        # Las capas se dibujan en orden:
        # 01.png, 02.png, 03.png y 04.png.
        for layer in self.level_preview_layers[level_id]:
            self.game.display.blit(layer, inner_rect.topleft)

        # Borde exterior de la tarjeta.
        pygame.draw.rect(
            self.game.display,
            (255, 255, 255),
            card_rect,
            self.CARD_BORDER_WIDTH,
        )

        return card_rect

    def _draw_level_label(self, text, center_x):
        """
        Dibuja el nombre del nivel debajo de su tarjeta.
        """
        label = self.game.render_bitmap_text_surface(
            text,
            self.game.bitmap_font_white,
            spacing=self.game.bitmap_spacing,
        )

        label_rect = label.get_rect(
            center=(center_x, self.LEVEL_LABEL_Y)
        )

        self.game.display.blit(label, label_rect)

        return label_rect

    def display_menu(self):
        self.run_display = True

        while self.run_display:
            self.game.check_events()
            self.check_input()

            # Fondo general del menu.
            self.game.display.blit(self.game.menu_bg, (0, 0))

            mid_w, _ = self.game.display.get_rect().center

            # =====================================================
            # TITULO
            # =====================================================
            title = self.game.render_bitmap_text_surface(
                "NIVELES",
                self.game.bitmap_font_white,
                spacing=self.game.bitmap_spacing,
            )

            title_rect = title.get_rect(
                center=(
                    mid_w,
                    self.TITLE_Y + self.offset_y,
                )
            )

            self.game.display.blit(title, title_rect)

            # =====================================================
            # TARJETAS DE LOS NIVELES
            # =====================================================
            level_1_card_rect = self._draw_level_card(
                level_id=1,
                center_x=self.LEVEL_1_X,
                base_color=self.LEVEL_1_BASE_COLOR,
            )

            level_2_card_rect = self._draw_level_card(
                level_id=2,
                center_x=self.LEVEL_2_X,
                base_color=self.LEVEL_2_BASE_COLOR,
            )

            # =====================================================
            # NOMBRES DE LOS NIVELES
            # =====================================================
            level_1_label_rect = self._draw_level_label(
                "NIVEL 1",
                self.LEVEL_1_X,
            )

            level_2_label_rect = self._draw_level_label(
                "NIVEL 2",
                self.LEVEL_2_X,
            )

            # =====================================================
            # OPCION VOLVER
            # =====================================================
            volver = self.game.render_bitmap_text_surface(
                "VOLVER",
                self.game.bitmap_font_white,
                spacing=self.game.bitmap_spacing,
            )

            volver_rect = volver.get_rect(
                center=(mid_w, self.BACK_Y)
            )

            self.game.display.blit(volver, volver_rect)

            # =====================================================
            # AREAS UTILIZADAS POR EL CURSOR
            # =====================================================
            # La unión de cada tarjeta con su etiqueta permite que
            # el cursor quede centrado respecto al conjunto visual.
            level_1_option_rect = level_1_card_rect.union(
                level_1_label_rect
            )

            level_2_option_rect = level_2_card_rect.union(
                level_2_label_rect
            )

            option_rects = [
                level_1_option_rect,
                level_2_option_rect,
                volver_rect,
            ]

            self.draw_cursor(option_rects[self.index])

            self.blit_screen()

    def check_input(self):
        """
        Distribucion de las opciones:
    
            NIVEL 1  <-->  NIVEL 2
                 \          /
                     VOLVER
    
        Indices:
            0 = NIVEL 1
            1 = NIVEL 2
            2 = VOLVER
        """
    
        # =========================================================
        # NAVEGACION VERTICAL
        # =========================================================
    
        if self.game.DOWN_KEY:
            # Desde cualquiera de los dos niveles baja a VOLVER.
            if self.index == 0 or self.index == 1:
                self.index = 2
    
            # Si ya está en VOLVER, no realiza ningún movimiento.
    
        elif self.game.UP_KEY:
            # Desde VOLVER sube directamente a NIVEL 1.
            if self.index == 2:
                self.index = 0
    
            # Si ya está en algún nivel, no realiza ningún movimiento.
    
        # =========================================================
        # NAVEGACION HORIZONTAL
        # =========================================================
    
        elif self.game.RIGHT_KEY:
            # NIVEL 1 -> NIVEL 2
            if self.index == 0:
                self.index = 1
    
            # NIVEL 2 -> NIVEL 1
            elif self.index == 1:
                self.index = 0
    
            # Desde VOLVER no hace nada.
    
        elif self.game.LEFT_KEY:
            # NIVEL 2 -> NIVEL 1
            if self.index == 1:
                self.index = 0
    
            # NIVEL 1 -> NIVEL 2
            elif self.index == 0:
                self.index = 1
    
            # Desde VOLVER no hace nada.
    
        # =========================================================
        # REGRESAR CON ESC
        # =========================================================
    
        if self.game.BACK_KEY:
            self.game.curr_menu = self.game.main_menu
            self.run_display = False
            return
    
        # =========================================================
        # SELECCIONAR CON ENTER
        # =========================================================
    
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