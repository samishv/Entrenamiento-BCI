import os
import csv
import pygame
import asyncio
import threading
from datetime import datetime
from bleak import BleakClient, BleakScanner

from scripts.utils import load_images, Animation
from scripts.entities import Player
from scripts.tilemap import Tilemap
from menu import MainMenu, ConfigMenu, LevelsMenu


# =============================================================================
# CONFIGURACIÓN MODIFICABLE
# Valores que se pueden ajustar para cambiar el comportamiento del juego sin
# tocar el resto del código.
# =============================================================================

# ===== JUGABILIDAD =====
AUTO_SPEED = 2                      # Velocidad del autorun (px por frame) ERA 2
FAIL_RETURN_SPEED = 1.2             # Velocidad del autorun mientras el jugador se recupera de un fallo ERA 1.2
TOTAL_CHALLENGES = 10               # Cantidad de desafíos (globos) por nivel
FIRST_STOP_TILE = 38                # Tile donde se detiene el jugador en el primer desafío
CHALLENGE_SPACING_TILES = 60        # Distancia (en tiles) entre un desafío y el siguiente
MAX_PRESSES = 1                     # Pulsaciones necesarias para reventar un globo
MAX_LIVES = 3                       # Vidas por partida
MAX_SCORE_PER_CHALLENGE = 100       # Puntos máximos por desafío (bajan conforme pasa el tiempo)
FPS = 60                            # Cuadros por segundo del juego

# ===== TIEMPO PARA PULSAR (segundos) =====
LEVEL_TIME_LIMITS = {               # {id_nivel: segundos}
    1: 2,
    2: 1,
}
DEFAULT_LEVEL_TIME_LIMIT = 2        # Tiempo para los niveles que no estén en LEVEL_TIME_LIMITS

# ===== COLORES DEL JUGADOR =====
PLAYER_COLOR_PALETTES = {
    "VERDE": (
        (177, 212, 128),  # #B1D480
        (128, 184, 120),  # #80B878
        (101, 141, 120),  # #658D78
    ),
    "TURQUESA": (
        (127, 216, 232),  # #7FD8E8
        (72, 183, 200),   # #48B7C8
        (46, 127, 143),   # #2E7F8F
    ),
    "AZUL": (
        (93, 169, 255),   # #5DA9FF
        (59, 130, 246),   # #3B82F6
        (30, 58, 138),    # #1E3A8A
    ),
    "MORADO": (
        (192, 132, 252),  # #C084FC
        (139, 92, 246),   # #8B5CF6
        (107, 33, 168),   # #6B21A8
    ),
    "ROSA": (
        (255, 167, 209),  # #FFA7D1
        (244, 114, 182),  # #F472B6
        (214, 51, 132),   # #D63384
    ),
    "MARRON": (
        (217, 179, 140),  # #D9B38C
        (166, 120, 86),   # #A67856
        (107, 79, 55),    # #6B4F37
    ),
    "GRIS": (
        (200, 205, 212),  # #C8CDD4
        (154, 163, 173),  # #9AA3AD
        (92, 101, 112),   # #5C6570
    ),
    "ROJO": (
        (255, 107, 107),  # #FF6B6B
        (230, 57, 70),    # #E63946
        (178, 34, 60),    # #B2223C
    ),
    "NARANJA": (
        (255, 179, 102),  # #FFB366
        (251, 140, 0),    # #FB8C00
        (180, 83, 9),     # #B45309
    ),
    "AMARILLO": (
        (255, 224, 102),  # #FFE066
        (250, 204, 21),   # #FACC15
        (176, 137, 0),    # #B08900
    ),
}

# ===== FONDO PARALLAX =====
PARALLAX_LAYER_SPEEDS = [0.25, 0.50, 0.85]   # Factor de velocidad de cada capa móvil
PARALLAX_BASE_SPEED = 20.0                   # Velocidad base del fondo (px/s)
PARALLAX_ACCEL = 14.0                        # Aceleración al arrancar
PARALLAX_DECEL = 14.0                        # Desaceleración al detenerse

# ===== FADES (HUD / PANTALLA FINAL / MENSAJES), en segundos =====
FADE_HEARTS_IN_DURATION = 0.5
FADE_HEARTS_OUT_DURATION = 0.25
FADE_END_TITLE_DURATION = 1.0
FADE_MSG_DURATION = 0.5

# ===== MENSAJES DE PROGRESO =====
PROGRESS_MSG_INTERVALS = [                                              # Pantallas de Score actual y globos restantes
    (10, 20) if i == 0 else (62 + (i - 1) * 60, 80 + (i - 1) * 60)
    for i in range(TOTAL_CHALLENGES)
]

# ===== BLUETOOTH (ESP32) =====
DEVICE_NAME = "ESP32-DRV2605"
CHAR_WRITE_UUID = "abcd1234-ab12-cd34-ef56-abcdef123456"
CHAR_NOTIFY_UUID = "abcd1234-ab12-cd34-ef56-abcdef123457"
INTENSIDAD_ARIADNE = 150            # Intensidad de la vibración
DELAY_ARIADNE = 200                 # Duración / retardo de la vibración (ms)
BLE_SCAN_TIMEOUT = 10               # Segundos buscando el dispositivo

# ===== GUARDADO DE RESULTADOS =====
SCORES_FOLDER_NAME = "scores"
SCORES_FILE_NAME = "historial_partidas.csv"

SCORES_HEADERS = [
    "Fecha",
    "Hora",
    "Nickname",
    "Nivel",
    "Resultado",
    "Globos rotos",
    "Puntuación",
    "ARIADNE",
    "Modo",
    "Color",
]


class Game:
    # ===== ESTADOS DEL JUEGO =====
    RUNNING = 0
    WAITING = 1
    RESOLVED = 2
    FAILED = 3
    GAME_OVER = 5
    WIN = 6

    def __init__(self):
        pygame.init()
        self.running, self.playing = True, False
        pygame.display.set_caption("juego v1_v12")

        # ===== VENTANA Y PANTALLA =====
        self.base_size = (480, 270)
        self.display = pygame.Surface(self.base_size)

        self.window_size = (1280, 720)
        self.fullscreen = False
        self.screen = pygame.display.set_mode(self.window_size)

        # ===== FONDO DEL MENÚ =====
        menu_bg_path = os.path.join("data", "images", "background", "menu_b", "1.png")
        self.menu_bg = pygame.image.load(menu_bg_path).convert_alpha()
        self.menu_bg = pygame.transform.scale(self.menu_bg, self.base_size)

        self._recalc_scaling()
        self.clock = pygame.time.Clock()

        # ===== MENÚS =====
        self.main_menu = MainMenu(self)
        self.config_menu = ConfigMenu(self)
        self.levels_menu = LevelsMenu(self)
        self.curr_menu = self.main_menu
        self.selected_level = 1

        # ===== TECLAS E INPUT DE TEXTO (MENÚ CONFIG) =====
        self.reset_keys()

        # ===== MODO DE MANO =====
        self.hand_mode = "DIESTRO"

        # ===== PERSONALIZACIÓN DE COLOR DEL JUGADOR =====
        self.player_color_names = list(PLAYER_COLOR_PALETTES)
        self.player_color_index = 0
        self.player_color = self.player_color_names[self.player_color_index]

        # ===== ASSETS =====
        self.original_player_images = {
            "run": load_images("entities/player/walk"),
            "idle": load_images("entities/player/idle"),
            "jump": load_images("entities/player/dive"),
        }

        # "ice" y "water" los carga apply_level_theme() al iniciar cada nivel.
        self.assets = {
            "hand": load_images(self._get_hand_folder()),
            "heart": load_images("entities/heart"),
            "clock": load_images("entities/clock"),
            "solidwater": load_images("tiles/solidwater"),
            "balloon": load_images("entities/balloon/states"),
            "balloon/pop": load_images("entities/balloon/pop"),
            "player/frozen": Animation(load_images("entities/player/frozen"), img_dur=16),
            "player/damage": Animation(load_images("entities/player/damage"), img_dur=5),
        }
        self.apply_player_color()

        # ===== CORAZONES REESCALADOS =====
        self.heart_small = []
        for img in self.assets["heart"]:
            w, h = img.get_size()
            self.heart_small.append(pygame.transform.scale(img, (w // 2, h // 2)))

        self.clock_frames = self.assets["clock"]

        # ===== BLE =====
        self.bluetooth_enabled = False   # el menú lo togglea
        self.ble_client = None           # BleakClient activo
        self.ble_loop = None             # loop asyncio del hilo BLE
        self.ble_thread = None

        # ===== BITMAP FONT =====
        self._font_metrics_cache = {}
        self.bitmap_charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789$¤+-*=%”’#@&_(),.;:?|{}<>[]§^~© "
        self.bitmap_spacing = 1
        self.bitmap_font = self.load_numbered_bitmap_font("data/fonts", self.bitmap_charset, start_index=1)
        self.bitmap_font_white = self.recolor_bitmap_font(self.bitmap_font, (255, 255, 255))

        self.bitmap_spacing_big = self.bitmap_spacing * self.BITMAP_SCALE_BIG
        self.bitmap_spacing_mid = self.bitmap_spacing * self.BITMAP_SCALE_MID
        self.bitmap_spacing_chibi = self.bitmap_spacing * self.BITMAP_SCALE_CHIBI
        self.bitmap_font_big = self._scale_font(self.bitmap_font, self.BITMAP_SCALE_BIG)
        self.bitmap_font_mid = self._scale_font(self.bitmap_font, self.BITMAP_SCALE_MID)
        self.bitmap_font_chibi = self._scale_font(self.bitmap_font, self.BITMAP_SCALE_CHIBI)

        # ===== GAMER TAG (lo asigna el menú de configuración) =====
        self.gamer_tag = ""

        # ===== ESTADO INICIAL DEL NIVEL 1 (mundo, jugador, retos, HUD) =====
        self.reset_level(1)

    # =========================================================================
    # VENTANA Y ESCALADO
    # =========================================================================
    def _recalc_scaling(self):
        sw, sh = self.screen.get_size()
        bw, bh = self.base_size
        scale = min(sw / bw, sh / bh)
        dw, dh = int(bw * scale), int(bh * scale)
        self.blit_size = (dw, dh)
        self.blit_offset = ((sw - dw) // 2, (sh - dh) // 2)

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(self.window_size)
        self._recalc_scaling()

    def _present_frame(self):
        self.screen.fill((0, 0, 0))
        scaled = pygame.transform.scale(self.display, self.blit_size)
        self.screen.blit(scaled, self.blit_offset)
        pygame.display.update()
        self.clock.tick(FPS)

    # =========================================================================
    # FUENTE BITMAP
    # =========================================================================
    BITMAP_SCALE_BIG = 5
    BITMAP_SCALE_MID = 2.3
    BITMAP_SCALE_CHIBI = 0.8

    def load_numbered_bitmap_font(self, folder, charset, start_index=1):
        font = {}
        for i, ch in enumerate(charset):
            path = os.path.join(folder, f"{start_index + i:02d}.png")
            try:
                font[ch] = pygame.image.load(path).convert_alpha()
            except FileNotFoundError:
                raise FileNotFoundError(f"No se encontró {path}. Revisa que existan 01.png..40.png")
        return font

    def recolor_bitmap_font(self, font, color=(255, 255, 255)):
        new_font = {}
        for ch, glyph in font.items():
            mask = pygame.mask.from_surface(glyph.convert_alpha())
            colored = mask.to_surface(setcolor=(*color, 255), unsetcolor=(0, 0, 0, 0))
            new_font[ch] = colored.convert_alpha()
        return new_font

    def _scale_font(self, font, scale):
        scaled_font = {}
        for ch, img in font.items():
            w, h = img.get_size()
            scaled_font[ch] = pygame.transform.scale(img, (w * scale, h * scale))
        return scaled_font

    def _get_font_metrics(self, font):
        key = id(font)
        if key not in self._font_metrics_cache:
            self._font_metrics_cache[key] = {ch: img.get_bounding_rect() for ch, img in font.items()}
        return self._font_metrics_cache[key]

    def _glyph(self, font, metrics, ch):
        """Devuelve (imagen, borde inferior) del carácter; usa el espacio si no existe."""
        img = font.get(ch) or font.get(' ')
        if img is None:
            return None, 0
        rect = metrics.get(ch) or metrics.get(' ')
        bottom = rect.bottom if (rect and rect.width and rect.height) else img.get_height()
        return img, bottom

    def _text_baseline(self, text, font, metrics):
        return max((self._glyph(font, metrics, ch)[1] for ch in text), default=0)

    def measure_bitmap_text(self, text, font, spacing=1):
        if not text:
            return 0, 0
        metrics = self._get_font_metrics(font)
        baseline = self._text_baseline(text, font, metrics)

        w = 0
        for i, ch in enumerate(text):
            img = font.get(ch) or font.get(' ')
            if img is None:
                continue
            w += img.get_width()
            if i != len(text) - 1:
                w += spacing
        return w, baseline

    def draw_bitmap_text(self, surface, text, x, y, font, spacing=1, baseline=None):
        if not text:
            return
        metrics = self._get_font_metrics(font)
        if baseline is None:
            baseline = self._text_baseline(text, font, metrics)

        cx = x
        for ch in text:
            img, bottom = self._glyph(font, metrics, ch)
            if img is None:
                cx += spacing
                continue
            surface.blit(img, (cx, y + baseline - bottom))
            cx += img.get_width() + spacing

    def render_bitmap_text_surface(self, text, font, spacing=1):
        w, h = self.measure_bitmap_text(text, font, spacing=spacing)
        surf = pygame.Surface((max(1, w), max(1, h)), pygame.SRCALPHA)
        self.draw_bitmap_text(surf, text, 0, 0, font, spacing=spacing, baseline=h)
        return surf

    def _approach_alpha(self, alpha, target, duration, dt):
        if duration <= 0:
            return target
        step = 255.0 * (dt / duration)
        if alpha < target:
            return min(target, alpha + step)
        return max(target, alpha - step)

    def _approach_value(self, value, target, rate, dt):
        if rate <= 0:
            return target
        step = rate * dt
        if value < target:
            return min(target, value + step)
        return max(target, value - step)

    # =========================================================================
    # BLUETOOTH (ESP32)
    # =========================================================================
    def start_ble_if_needed(self):
        """Llamar justo antes de iniciar niveles para preparar BLE si está habilitado en el menú."""
        if self.bluetooth_enabled and self.ble_client is None:
            self.ble_thread = threading.Thread(target=self._ble_thread_runner, daemon=True)
            self.ble_thread.start()
        if not self.bluetooth_enabled:
            self.stop_ble()  # si se deshabilitó, aseguramos detener cualquier conexión activa

    def stop_ble(self):
        """Llamar al salir del nivel o al togglear OFF desde el menú."""
        self.ble_client = None  # la corutina detecta esto y sale

    def send_vibration(self, intensity: int, delay_ms: int):
        """Usar esto desde cualquier parte del juego."""
        if not self.bluetooth_enabled or self.ble_client is None:
            return
        cmd = f"{intensity},{delay_ms}"
        asyncio.run_coroutine_threadsafe(
            self.ble_client.write_gatt_char(CHAR_WRITE_UUID, cmd.encode()),
            self.ble_loop
        )

    def _ble_thread_runner(self):
        self.ble_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.ble_loop)
        self.ble_loop.run_until_complete(self._ble_task())

    async def _ble_task(self):
        device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=BLE_SCAN_TIMEOUT)
        if device is None:
            print("ESP32 no encontrado")
            return
        async with BleakClient(device) as client:
            self.ble_client = client
            await client.start_notify(CHAR_NOTIFY_UUID, self._on_notify)
            while self.ble_client is not None:
                await asyncio.sleep(0.1)
            await client.stop_notify(CHAR_NOTIFY_UUID)

    def _on_notify(self, sender, data):
        print(f"[ESP32] {data.decode()}")

    # =========================================================================
    # JUGADOR: COLOR Y MODO DE MANO
    # =========================================================================
    # Tonos verdes de los sprites originales; son los que se sustituyen por la paleta elegida.
    ORIGINAL_PLAYER_PALETTE = (
        (177, 212, 128),  # #B1D480
        (128, 184, 120),  # #80B878
        (101, 141, 120),  # #658D78
    )
    # Animaciones recoloreables: (clave en original_player_images, duración por cuadro)
    PLAYER_ANIMATIONS = (("run", 5), ("idle", 16), ("jump", 5))

    def _recolor_player_image(self, original_image, color_name):
        """Copia la imagen sustituyendo los tres tonos verdes originales por los de la paleta."""
        recolored_image = original_image.copy()
        target_palette = PLAYER_COLOR_PALETTES[color_name]

        pixel_array = pygame.PixelArray(recolored_image)
        for original_color, target_color in zip(self.ORIGINAL_PLAYER_PALETTE, target_palette):
            pixel_array.replace(
                recolored_image.map_rgb(original_color),
                recolored_image.map_rgb(target_color)
            )
        pixel_array.close()

        return recolored_image

    def _recolor_player_images(self, original_images, color_name):
        return [self._recolor_player_image(image, color_name) for image in original_images]

    def apply_player_color(self):
        """Genera las animaciones run, idle y jump con el color seleccionado."""
        for name, img_dur in self.PLAYER_ANIMATIONS:
            self.assets[f"player/{name}"] = Animation(
                self._recolor_player_images(self.original_player_images[name], self.player_color),
                img_dur=img_dur
            )

    def next_player_color(self):
        """Avanza al siguiente color de la rueda y actualiza las animaciones."""
        self.player_color_index = (self.player_color_index + 1) % len(self.player_color_names)
        self.player_color = self.player_color_names[self.player_color_index]
        self.apply_player_color()

    def get_player_menu_color(self):
        """Color 2 de la gama seleccionada (bloque detrás del nombre del color en el menú)."""
        return PLAYER_COLOR_PALETTES[self.player_color][1]

    def _get_hand_folder(self):
        return "entities/hand_left" if self.hand_mode == "ZURDO" else "entities/hand"

    def set_hand_mode(self, mode):
        """Cambia el modo de mano y vuelve a cargar las imágenes correspondientes."""
        mode = mode.upper()
        if mode not in ("DIESTRO", "ZURDO"):
            raise ValueError("El modo debe ser DIESTRO o ZURDO")

        self.hand_mode = mode
        self.assets["hand"] = load_images(self._get_hand_folder())
        self.hand_frames = self.assets["hand"]

        # Reiniciar la animación de la mano.
        self.hand_frame_idx = 0
        self.hand_timer = 0.0

    def toggle_hand_mode(self):
        self.set_hand_mode("ZURDO" if self.hand_mode == "DIESTRO" else "DIESTRO")

    # =========================================================================
    # TEMA DE NIVEL Y FONDO PARALLAX
    # =========================================================================
    def apply_level_theme(self, level_id: int):
        self.current_level_theme = level_id
        suffix = "" if level_id == 1 else "2"
        self._reload_level_tiles(suffix)

        bg_level_folder = "level1" if level_id == 1 else "level2"
        self._load_parallax_background(os.path.join("data", "images", "background", bg_level_folder))

    def _reload_level_tiles(self, suffix: str):
        for key in ("ice", "water"):
            base = f"tiles/{key}"
            candidate = f"{base}{suffix}"
            has_variant = os.path.isdir(os.path.join("data", "images", candidate))
            self.assets[key] = load_images(candidate if has_variant else base)

    def _load_parallax_background(self, folder_path: str):
        self.bg_folder = folder_path
        bg_layers = []
        for i in range(1, 5):
            img = pygame.image.load(os.path.join(folder_path, f"{i:02d}.png")).convert_alpha()
            bg_layers.append(pygame.transform.scale(img, self.base_size))

        self.bg_static = bg_layers[0]
        self.bg_moving = bg_layers[1:]
        self.bg_offsets = [0.0] * len(self.bg_moving)

    def _update_parallax(self, dt, speed_px_s):
        if speed_px_s <= 0:
            return
        base = speed_px_s * dt
        for i, layer in enumerate(self.bg_moving):
            self.bg_offsets[i] = (self.bg_offsets[i] + base * PARALLAX_LAYER_SPEEDS[i]) % layer.get_width()

    def _draw_parallax(self):
        self.display.blit(self.bg_static, (0, 0))
        for i, layer in enumerate(self.bg_moving):
            w = layer.get_width()
            x = -self.bg_offsets[i]
            while x < self.base_size[0]:
                self.display.blit(layer, (int(x), 0))
                x += w

    # =========================================================================
    # DESAFÍOS: PARADAS, GLOBO, MANO Y BLOQUE DE HIELO FLOTANTE
    # =========================================================================
    BALLOON_OFFSET_TILES_X = 7           # Tiles entre la parada y el globo
    BALLOON_WORLD_Y = 32                 # Altura del globo en el mundo (px)
    BALLOON_POP_FRAME_DURATION = 0.1     # Segundos por cuadro de la animación de reventar

    HAND_OFFSET_TILES_Y = 8.5            # Altura de la mano respecto al globo (en tiles)
    HAND_FRAME_DURATION = 0.10           # Segundos por cuadro de la animación de la mano

    BRIDGE_W_TILES = 10                  # Ancho del bloque de hielo (tiles)
    BRIDGE_H_TILES = 3                   # Alto del bloque de hielo (tiles)
    BRIDGE_OFFSET_Y = 32                 # Distancia (px) bajo el globo donde flota el bloque
    BRIDGE_GRAVITY = 400.0               # Gravedad del bloque al caer
    BRIDGE_VARIANTS = [                  # Índice de tile de hielo para cada celda del bloque
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        [7, 8, 8, 8, 8, 8, 8, 8, 8, 3],
        [6, 5, 5, 5, 5, 5, 5, 5, 5, 4],
    ]

    def _setup_stops(self):
        self.stop_tiles = [FIRST_STOP_TILE + i * CHALLENGE_SPACING_TILES for i in range(TOTAL_CHALLENGES)]
        self.stop_index = 0
        self.stop_x_px = self.stop_tiles[self.stop_index] * self.tile_size
        self.active_challenge = True

    def _build_challenges(self):
        ts = self.tile_size
        challenges = []
        for stop_tile in self.stop_tiles:
            balloon_tile_x = stop_tile + self.BALLOON_OFFSET_TILES_X
            bridge_tile_x0 = balloon_tile_x - (self.BRIDGE_W_TILES // 2)

            challenges.append({
                "balloon_tile_x": balloon_tile_x,
                "balloon_world_y": self.BALLOON_WORLD_Y,
                "hand_world_y": self.BALLOON_WORLD_Y + (self.HAND_OFFSET_TILES_Y * ts),
                "balloon_visible": True,
                "balloon_is_popping": False,
                "balloon_pop_frame": 0,
                "balloon_pop_timer": 0.0,
                "press_count": 0,
                "scored": False,
                "bridge_tile_x0": bridge_tile_x0,
                "bridge_world_x": bridge_tile_x0 * ts,
                "bridge_world_y": self.BALLOON_WORLD_Y + self.BRIDGE_OFFSET_Y,
                "bridge_visible": True,
                "bridge_state": "FLOATING",
                "bridge_vel_y": 0.0,
                "bridge_target_top_tile_y": None,
                "bridge_done": False,
            })
        return challenges

    def current_challenge(self):
        if 0 <= self.stop_index < len(self.challenges):
            return self.challenges[self.stop_index]
        return None

    def advance_stop(self):
        self.stop_index += 1
        if self.stop_index < len(self.stop_tiles):
            self.stop_x_px = self.stop_tiles[self.stop_index] * self.tile_size
        else:
            self.stop_x_px = float("inf")
            self.active_challenge = False

    def _find_water_top_y_for_bridge(self, bridge_tile_x0):
        x0 = bridge_tile_x0
        x1 = bridge_tile_x0 + self.BRIDGE_W_TILES - 1
        water_ys = [
            tile["pos"][1]
            for tile in self.tilemap.tilemap.values()
            if tile.get("type") == "water" and x0 <= tile["pos"][0] <= x1
        ]
        return min(water_ys, default=None)

    def _rect_touches_tile_type(self, rect, tile_type):
        ts = self.tilemap.tile_size
        for tx in range(rect.left // ts, (rect.right - 1) // ts + 1):
            for ty in range(rect.top // ts, (rect.bottom - 1) // ts + 1):
                tile = self.tilemap.tilemap.get(f"{tx};{ty}")
                if tile and tile.get("type") == tile_type:
                    return True
        return False

    # =========================================================================
    # REINICIO DE NIVEL
    # =========================================================================
    TILE_SIZE = 16
    MAP_FILE = "map.json"
    PLAYER_START_POS = (0, 170)
    PLAYER_SIZE = (32, 48)

    def reset_level(self, level_id: int):
        self.apply_level_theme(level_id)
        
        # GUARDADO DE DATOS
        self.result_saved = False
        self.balloons_broken = 0

        self.time_limit = LEVEL_TIME_LIMITS.get(level_id, DEFAULT_LEVEL_TIME_LIMIT)
        self.time_left = self.time_limit

        self.tilemap = Tilemap(self, tile_size=self.TILE_SIZE)
        self.tilemap.load(self.MAP_FILE)
        self.tile_size = self.tilemap.tile_size

        self.player = Player(self, self.PLAYER_START_POS, self.PLAYER_SIZE)
        self.scroll = [0, 0]
        self.fixed_scroll_y = None

        self.auto_speed = AUTO_SPEED
        self.auto_speed_normal = self.auto_speed
        self.fail_return_active = False

        now = pygame.time.get_ticks()
        self.last_time = now
        self.anim_last_time = now

        self.lives = MAX_LIVES
        self.score = 0

        self._setup_stops()
        self.challenges = self._build_challenges()

        self.state = self.RUNNING

        self.hand_frames = self.assets["hand"]
        self.hand_frame_idx = 0
        self.hand_timer = 0.0
        self.hand_active = False

        self.hearts_alpha = 0
        self.hearts_fading_in = True
        self.hearts_fading_out = False
        self.end_title_alpha = 0
        self.end_title_fade_in = False
        self.prev_show_end = False

        self.msg_alpha = 0
        self.msg_state = "HIDDEN"
        self.msg_interval_index = 0

        self.recovering = False
        self.escape_started = False
        self.escape_x_px = None
        self.escape_target_y = None
        self.pending_life_loss = False
        self.go_phase = "NONE"
        self.go_walk_end_x = None

        self.bg_speed_current = 0.0

    # =========================================================================
    # BUCLE PRINCIPAL
    # =========================================================================
    CAMERA_ANCHOR_X = 0.23                   # Posición horizontal del jugador en pantalla (0 a 1)
    CAMERA_SMOOTHING = 30                    # Mayor valor = cámara más suave

    ESCAPE_END_OFFSET_TILES = 11             # Tiles tras la parada fallida donde el jugador salta fuera del agua
    ESCAPE_JUMP_TILES = 6                    # Altura del salto de escape (tiles)
    ESCAPE_JUMP_VELOCITY = -5                # Impulso vertical inicial del salto de escape
    GAME_OVER_WALK_TILES = 8                 # Tiles que camina el jugador tras salir del agua en Game Over
    WIN_END_TILE = TOTAL_CHALLENGES*60 + 8   # Tile de la meta del nivel

    def run(self):
        self.start_ble_if_needed()
        # === ROUTER DE NIVELES (cualquier nivel distinto de 2 usa el nivel 1) ===
        self.reset_level(2 if self.selected_level == 2 else 1)
        self._game_loop()

    def _is_end_screen(self):
        return self.state == self.WIN or (self.state == self.GAME_OVER and self.go_phase == "STOP")
    
    def _get_scores_file_path(self):
        game_directory = os.path.dirname(os.path.abspath(__file__))
        scores_directory = os.path.join(game_directory,SCORES_FOLDER_NAME)
        scores_file_path = os.path.join(scores_directory,SCORES_FILE_NAME)
        return scores_directory, scores_file_path
    
    def _save_finished_game_result(self):
        if self.result_saved: return
        if self.state == self.WIN: result = "NIVEL COMPLETADO"
        elif self.state == self.GAME_OVER and self.go_phase == "STOP": result = "GAME OVER"
        else: return
    
        now = datetime.now()
        nickname = self.gamer_tag.strip() or "SIN_NOMBRE"
        level_name = "NIVEL 2" if self.selected_level == 2 else "NIVEL 1"
        ariadne_status = "ON" if self.bluetooth_enabled else "OFF"
    
        row = [now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), nickname, level_name, result, self.balloons_broken, self.score, ariadne_status, self.hand_mode, self.player_color]
    
        try:
            scores_directory, scores_file_path = self._get_scores_file_path()
            os.makedirs(scores_directory, exist_ok=True)
            write_headers = not os.path.exists(scores_file_path) or os.path.getsize(scores_file_path) == 0
    
            with open(scores_file_path, mode="a", newline="", encoding="utf-8-sig") as csv_file:
                writer = csv.writer(csv_file)
                if write_headers: writer.writerow(SCORES_HEADERS)
                writer.writerow(row)
                csv_file.flush()
                os.fsync(csv_file.fileno())
    
            self.result_saved = True
            print(f"[CSV] Resultado guardado correctamente en: {scores_file_path}")
        except (OSError, csv.Error) as error:
            print(f"[CSV] No se pudo guardar el resultado de la partida: {error}")
        except Exception as error:
            print(f"[CSV] Error inesperado al guardar el resultado: {error}")


    def _game_loop(self):
        while self.playing and self.running:
            now_anim = pygame.time.get_ticks()
            frame_dt = (now_anim - self.anim_last_time) / 1000.0
            self.anim_last_time = now_anim

            if not self._handle_events():
                return

            self._update_camera()
            self._update_background(frame_dt)
            render_scroll = (int(self.scroll[0]), int(self.scroll[1]))

            # ===== RENDER: JUGADOR, TILEMAP, BLOQUES Y MANO =====
            self.player.render(self.display, offset=render_scroll)
            self.tilemap.render(self.display, offset=render_scroll)
            self._render_bridges(render_scroll)
            self._update_and_render_hand(frame_dt, render_scroll)

            # ===== MOVIMIENTO Y ESTADOS DEL JUGADOR =====
            move_x = self._compute_move_x()
            self._start_escape_jump()
            self.player.update(self.tilemap, (move_x, 0))
            self._apply_life_loss()
            self._finish_escape_jump()
            self._update_game_over_walk()
            self._update_waiting_timer()

            # ===== ANIMACIONES DE LOS DESAFÍOS =====
            self._update_balloon_pops(frame_dt)
            self._update_falling_bridges(frame_dt)
            self._render_balloons(render_scroll)
            self._check_win()
            self._save_finished_game_result()

            # ===== HUD Y PANTALLA FINAL =====
            show_end = self._is_end_screen()
            self._update_fades(show_end, frame_dt)
            self._update_progress_messages(show_end, frame_dt)
            self._draw_hud(show_end)
            self._draw_progress_messages(show_end)
            if show_end:
                self._draw_end_screen()

            self._present_frame()

    # ----- Eventos -----
    def _handle_events(self):
        """Procesa los eventos del juego. Devuelve False si se cerró la ventana."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.playing = False
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    self._toggle_fullscreen()
                if event.key == pygame.K_ESCAPE:
                    self.playing = False
                if event.key == pygame.K_SPACE:
                    self._on_space_pressed()
        return True

    def _on_space_pressed(self):
        if self.state != self.WAITING or not self.active_challenge:
            return

        self.send_vibration(INTENSIDAD_ARIADNE, DELAY_ARIADNE)
        cur = self.current_challenge()
        if not cur:
            return

        cur["press_count"] += 1
        if cur["press_count"] < MAX_PRESSES:
            return

        self.state = self.RESOLVED
        self.hand_active = False
        if not cur["scored"]:
            elapsed = max(0.0, self.time_limit - self.time_left)
            gained = int(max(0.0, (self.time_limit - elapsed)) * MAX_SCORE_PER_CHALLENGE / self.time_limit)
            gained = max(0, min(gained, MAX_SCORE_PER_CHALLENGE))
            self.score += gained
            self.balloons_broken += 1
            cur["scored"] = True
        cur["balloon_is_popping"] = True
        cur["bridge_state"] = "FALLING"
        cur["bridge_vel_y"] = 0.0
        cur["bridge_target_top_tile_y"] = self._find_water_top_y_for_bridge(cur["bridge_tile_x0"])

    # ----- Cámara y fondo -----
    def _update_camera(self):
        if self.fixed_scroll_y is None:
            self.fixed_scroll_y = self.scroll[1]

        target_scroll_x = self.player.rect().centerx - self.display.get_width() * self.CAMERA_ANCHOR_X
        self.scroll[0] += (target_scroll_x - self.scroll[0]) / self.CAMERA_SMOOTHING
        self.scroll[1] = self.fixed_scroll_y

    def _update_background(self, frame_dt):
        """Parallax sincronizado a la cámara: se mueve mientras el jugador corre."""
        parallax_active = (
            self.state in (self.RUNNING, self.FAILED)
            or (self.state == self.GAME_OVER and self.go_phase != "STOP")
        )
        target_speed = PARALLAX_BASE_SPEED if parallax_active else 0.0
        rate = PARALLAX_ACCEL if target_speed > self.bg_speed_current else PARALLAX_DECEL
        self.bg_speed_current = self._approach_value(self.bg_speed_current, target_speed, rate, frame_dt)

        self._update_parallax(frame_dt, self.bg_speed_current)
        self._draw_parallax()

    # ----- Render del mundo -----
    def _render_bridges(self, render_scroll):
        ts = self.tilemap.tile_size
        ice = self.assets["ice"]
        for ch in self.challenges:
            if not ch["bridge_visible"]:
                continue
            for ix in range(self.BRIDGE_W_TILES):
                for iy in range(self.BRIDGE_H_TILES):
                    wx = ch["bridge_world_x"] + ix * ts
                    wy = ch["bridge_world_y"] + iy * ts
                    ice_img = ice[self.BRIDGE_VARIANTS[iy][ix]]
                    self.display.blit(ice_img, (wx - render_scroll[0], wy - render_scroll[1]))

    def _update_and_render_hand(self, frame_dt, render_scroll):
        if not (self.state == self.WAITING and self.hand_active and self.active_challenge):
            return

        self.hand_timer += frame_dt
        while self.hand_timer >= self.HAND_FRAME_DURATION:
            self.hand_timer -= self.HAND_FRAME_DURATION
            self.hand_frame_idx = (self.hand_frame_idx + 1) % len(self.hand_frames)

        cur = self.current_challenge()
        if cur:
            hand_img = self.hand_frames[self.hand_frame_idx]
            hand_world_x = cur["balloon_tile_x"] * self.tilemap.tile_size
            hand_screen_x = int(hand_world_x - render_scroll[0] - hand_img.get_width() // 2)
            hand_screen_y = int(cur["hand_world_y"] - render_scroll[1] - hand_img.get_height() // 2)
            self.display.blit(hand_img, (hand_screen_x, hand_screen_y))

    def _render_balloons(self, render_scroll):
        ts = self.tilemap.tile_size
        pop_frames = self.assets["balloon/pop"]
        for ch in self.challenges:
            if not ch["balloon_visible"]:
                continue

            if ch["balloon_is_popping"]:
                img = pop_frames[ch["balloon_pop_frame"]] if ch["balloon_pop_frame"] < len(pop_frames) else None
            else:
                img = self.assets["balloon"][min(ch["press_count"], 2)]

            if img:
                balloon_screen_x = ch["balloon_tile_x"] * ts - render_scroll[0]
                balloon_screen_y = ch["balloon_world_y"] - render_scroll[1]
                self.display.blit(
                    img,
                    (int(balloon_screen_x - img.get_width() // 2),
                     int(balloon_screen_y - img.get_height() // 2)))

    # ----- Movimiento y estados del jugador -----
    def _compute_move_x(self):
        move_x = self.auto_speed

        if self.state == self.RUNNING:
            player_right_now = self.player.pos[0] + self.player.size[0]
            player_right_next = player_right_now + move_x

            # Llegó a la parada del desafío actual: se detiene y espera la pulsación.
            if player_right_now < self.stop_x_px <= player_right_next:
                move_x = self.stop_x_px - player_right_now

                cur = self.current_challenge()
                if cur:
                    cur["press_count"] = 0
                    # Activar la mano en WAITING
                    self.hand_active = True
                    self.hand_timer = 0.0
                    self.hand_frame_idx = 0

                self.state = self.WAITING
                self.time_left = self.time_limit
                self.last_time = pygame.time.get_ticks()

        if self.state in (self.WAITING, self.WIN) or (self.state == self.GAME_OVER and self.go_phase == "STOP"):
            move_x = 0
        elif self.state == self.RESOLVED:
            cur = self.current_challenge()
            move_x = self.auto_speed if (cur and cur["bridge_done"]) else 0

        return move_x

    def _start_escape_jump(self):
        """Salto para salir del agua cuando el jugador llega al tile de escape."""
        if self.state not in (self.FAILED, self.GAME_OVER) or not self.recovering or self.escape_started:
            return

        if self.player.pos[0] + self.player.size[0] >= self.escape_x_px:
            self.escape_started = True
            self.escape_target_y = self.player.pos[1] - (self.ESCAPE_JUMP_TILES * self.tile_size)
            if self.state == self.GAME_OVER:
                self.go_phase = "RISE"
            self.player.velocity[1] = self.ESCAPE_JUMP_VELOCITY

    def _apply_life_loss(self):
        """Al tocar el agua se pierde una vida; sin vidas es Game Over."""
        if self.pending_life_loss and self._rect_touches_tile_type(self.player.rect(), "water"):
            self.pending_life_loss = False
            self.lives = max(0, self.lives - 1)

            if self.lives == 0:
                self.state = self.GAME_OVER
                self.go_phase = "NONE"

    def _finish_escape_jump(self):
        """Termina el salto de escape al alcanzar la altura objetivo."""
        if not (self.state in (self.FAILED, self.GAME_OVER) and self.recovering
                and self.escape_started and self.escape_target_y is not None):
            return
        if self.player.pos[1] > self.escape_target_y:
            return

        self.player.pos[1] = self.escape_target_y
        self.player.velocity[1] = 0

        self.recovering = False
        self.escape_started = False
        self.escape_target_y = None

        if self.state == self.GAME_OVER:
            self.go_phase = "WALK2"
            self.go_walk_end_x = self.player.pos[0] + (self.GAME_OVER_WALK_TILES * self.tile_size)
        else:
            self.state = self.RUNNING

        if self.fail_return_active:
            self.auto_speed = self.auto_speed_normal
            self.fail_return_active = False

    def _update_game_over_walk(self):
        if self.state == self.GAME_OVER and self.go_phase == "WALK2" and self.player.pos[0] >= self.go_walk_end_x:
            self.go_phase = "STOP"
            self.auto_speed = 0

    def _update_waiting_timer(self):
        """Cuenta regresiva mientras se espera la pulsación; al agotarse el desafío se falla."""
        if self.state != self.WAITING:
            return

        now = pygame.time.get_ticks()
        dt = (now - self.last_time) / 1000.0
        self.last_time = now

        self.time_left -= dt
        if self.time_left > 0:
            return

        if not self.fail_return_active:
            self.fail_return_active = True
            self.auto_speed = FAIL_RETURN_SPEED

        self.state = self.FAILED
        self.hand_active = False
        failed_stop_tile = self.stop_tiles[self.stop_index]
        self.escape_x_px = (failed_stop_tile + self.ESCAPE_END_OFFSET_TILES) * self.tile_size

        cur = self.current_challenge()
        if cur and cur["bridge_state"] == "FLOATING":
            cur["bridge_state"] = "STUCK"

        if not self.recovering:
            self.pending_life_loss = True
            self.recovering = True
            self.escape_started = False
            self.escape_target_y = None

        self.advance_stop()

    def _check_win(self):
        if self.state in (self.GAME_OVER, self.WIN):
            return
        if self.stop_index >= TOTAL_CHALLENGES and self.lives > 0:
            if self.player.pos[0] + self.player.size[0] >= self.WIN_END_TILE * self.tile_size:
                self.state = self.WIN
                self.auto_speed = 0

    # ----- Animaciones de los desafíos -----
    def _update_balloon_pops(self, frame_dt):
        pop_frame_count = len(self.assets["balloon/pop"])
        for ch in self.challenges:
            if not (ch["balloon_is_popping"] and ch["balloon_visible"]):
                continue

            ch["balloon_pop_timer"] += frame_dt
            while ch["balloon_pop_timer"] >= self.BALLOON_POP_FRAME_DURATION:
                ch["balloon_pop_timer"] -= self.BALLOON_POP_FRAME_DURATION
                ch["balloon_pop_frame"] += 1
                if ch["balloon_pop_frame"] >= pop_frame_count:
                    ch["balloon_is_popping"] = False
                    ch["balloon_visible"] = False
                    break

    def _update_falling_bridges(self, frame_dt):
        ts = self.tilemap.tile_size
        for ch in self.challenges:
            if ch["bridge_state"] != "FALLING":
                continue

            if ch["bridge_target_top_tile_y"] is None:
                ch["bridge_state"] = "STUCK"
                continue

            ch["bridge_vel_y"] += self.BRIDGE_GRAVITY * frame_dt
            ch["bridge_world_y"] += ch["bridge_vel_y"] * frame_dt

            target_top_y_px = ch["bridge_target_top_tile_y"] * ts
            if ch["bridge_world_y"] < target_top_y_px:
                continue

            # El bloque aterriza: se convierte en tiles de hielo sólidos del tilemap.
            ch["bridge_world_y"] = target_top_y_px
            ch["bridge_state"] = "LANDED"
            ch["bridge_done"] = True

            x0 = ch["bridge_tile_x0"]
            y0 = ch["bridge_target_top_tile_y"]
            for dy in range(self.BRIDGE_H_TILES):
                for dx in range(self.BRIDGE_W_TILES):
                    x, y = x0 + dx, y0 + dy
                    self.tilemap.tilemap[f"{x};{y}"] = {
                        "type": "ice",
                        "variant": self.BRIDGE_VARIANTS[dy][dx],
                        "pos": [x, y]}

            ch["bridge_visible"] = False

            if ch is self.current_challenge():
                self.state = self.RUNNING
                self.advance_stop()
                self.hand_active = False

    # ----- Fades y mensajes -----
    def _update_fades(self, show_end, frame_dt):
        # Al entrar a la pantalla final: corazones se desvanecen y aparecen los títulos.
        if show_end and not self.prev_show_end:
            self.hearts_fading_out = True
            self.hearts_fading_in = False

            self.end_title_alpha = 0
            self.end_title_fade_in = True

        self.prev_show_end = show_end

        if self.hearts_fading_in:
            self.hearts_alpha = self._approach_alpha(self.hearts_alpha, 255, FADE_HEARTS_IN_DURATION, frame_dt)
            if self.hearts_alpha >= 255:
                self.hearts_fading_in = False

        if self.hearts_fading_out:
            self.hearts_alpha = self._approach_alpha(self.hearts_alpha, 0, FADE_HEARTS_OUT_DURATION, frame_dt)
            if self.hearts_alpha <= 0:
                self.hearts_fading_out = False

        if self.end_title_fade_in:
            self.end_title_alpha = self._approach_alpha(self.end_title_alpha, 255, FADE_END_TITLE_DURATION, frame_dt)
            if self.end_title_alpha >= 255:
                self.end_title_fade_in = False

    def _update_progress_messages(self, show_end, frame_dt):
        if not (self.state == self.RUNNING and not show_end and self.msg_interval_index < len(PROGRESS_MSG_INTERVALS)):
            self.msg_state = "HIDDEN"
            self.msg_alpha = 0
            return

        player_right_tile = int((self.player.pos[0] + self.player.size[0]) // self.tile_size)
        start_t, end_t = PROGRESS_MSG_INTERVALS[self.msg_interval_index]

        if self.msg_state == "HIDDEN" and player_right_tile >= start_t:
            self.msg_state = "FADING_IN"
            self.msg_alpha = 0

        if self.msg_state in ("FADING_IN", "SHOWN") and player_right_tile >= end_t:
            self.msg_state = "FADING_OUT"

        if self.msg_state == "FADING_IN":
            self.msg_alpha = self._approach_alpha(self.msg_alpha, 255, FADE_MSG_DURATION, frame_dt)
            if self.msg_alpha >= 255:
                self.msg_state = "SHOWN"

        elif self.msg_state == "FADING_OUT":
            self.msg_alpha = self._approach_alpha(self.msg_alpha, 0, FADE_MSG_DURATION, frame_dt)
            if self.msg_alpha <= 0:
                self.msg_state = "HIDDEN"
                self.msg_interval_index += 1

    # ----- HUD y pantalla final -----
    HUD_MARGIN = 8                       # Margen del HUD respecto a los bordes (px)
    HUD_CLOCK_PADDING = 8                # Separación entre el gamer tag y el reloj (px)

    def _draw_hud(self, show_end):
        margin = self.HUD_MARGIN
        hud_visible = self.hearts_alpha > 0 and (not show_end or self.hearts_fading_out)

        # ===== GAMER TAG =====
        tag_height = 0
        if hud_visible:
            tag_surf = self.render_bitmap_text_surface(
                self.gamer_tag, self.bitmap_font_chibi, spacing=self.bitmap_spacing_chibi
            )
            tag_surf.set_alpha(int(self.hearts_alpha))
            self.display.blit(tag_surf, (margin, margin))
            tag_height = tag_surf.get_height()

        # ===== RELOJ =====
        if self.state == self.WAITING:
            progress = (self.time_limit - self.time_left) / self.time_limit
            progress = max(0.0, min(progress, 0.9999))
            clock_img = self.clock_frames[int(progress * len(self.clock_frames))]

            cy = margin + tag_height + (self.HUD_CLOCK_PADDING if tag_height > 0 else 0)
            self.display.blit(clock_img, (margin, cy))

        # ===== VIDAS =====
        if hud_visible:
            heart_idx = max(0, min(MAX_LIVES - self.lives, 3))
            heart_img = self.heart_small[heart_idx].copy()
            heart_img.set_alpha(int(self.hearts_alpha))
            self.display.blit(heart_img, (self.base_size[0] - heart_img.get_width() - margin, margin))

    def _draw_progress_messages(self, show_end):
        if not (self.state == self.RUNNING and self.msg_alpha > 0 and not show_end):
            return

        remaining = max(0, TOTAL_CHALLENGES - self.stop_index)
        texts = ("GLOBOS RESTANTES", str(remaining), "PUNTUACION", str(self.score))
        s1_label, s1_val, s2_label, s2_val = [
            self.render_bitmap_text_surface(t, self.bitmap_font, spacing=self.bitmap_spacing) for t in texts
        ]
        for s in (s1_label, s1_val, s2_label, s2_val):
            s.set_alpha(int(self.msg_alpha))

        cx = self.base_size[0] // 2
        cy = self.base_size[1] // 4
        y_gap = 6

        x_label = cx - 144
        x1_val = cx + 144
        x2_val = cx + 144 if self.score == 0 else cx + 128

        y1 = cy - s1_label.get_height() - y_gap
        y2 = cy + y_gap

        self.display.blit(s1_label, (x_label, y1))
        self.display.blit(s1_val, (x1_val, y1))
        self.display.blit(s2_label, (x_label, y2))
        self.display.blit(s2_val, (x2_val, y2))

    def _draw_end_screen(self):
        """Pantalla final (WIN / GAME OVER) estilo Minecraft."""
        end_alpha = int(self.end_title_alpha)
        completed = sum(1 for ch in self.challenges if ch["scored"])

        # Textos normales (1x)
        render = self.render_bitmap_text_surface
        s_gr = render("GLOBOS ROTOS", self.bitmap_font, spacing=self.bitmap_spacing)
        s_completed = render(str(completed), self.bitmap_font, spacing=self.bitmap_spacing)
        s_pts = render("PUNTUACION", self.bitmap_font, spacing=self.bitmap_spacing)
        s_score = render(str(self.score), self.bitmap_font, spacing=self.bitmap_spacing)

        for s in (s_gr, s_completed, s_pts, s_score):
            s.set_alpha(end_alpha)

        self.display.blit(s_gr, (11 * 16, 11 * 16))
        self.display.blit(s_completed, ((25 if self.score == 0 else 26) * 16, 11 * 16))
        self.display.blit(s_pts, (11 * 16, 12 * 16))
        self.display.blit(s_score, (25 * 16, 12 * 16))

        # Títulos grandes
        if self.state == self.WIN:
            t1 = render("NIVEL", self.bitmap_font_big, spacing=self.bitmap_spacing_big)
            t2 = render("COMPLETADO", self.bitmap_font_mid, spacing=self.bitmap_spacing_mid)
        else:
            t1 = render("GAME", self.bitmap_font_big, spacing=self.bitmap_spacing_big)
            t2 = render("OVER", self.bitmap_font_big, spacing=self.bitmap_spacing_big)

        t1.set_alpha(end_alpha)
        t2.set_alpha(end_alpha)
        self.display.blit(t1, (11 * 16, 1 * 16))
        self.display.blit(t2, (11 * 16, 6 * 16))

    # =========================================================================
    # ENTRADA PARA LOS MENÚS
    # =========================================================================
    _MENU_KEY_FLAGS = {
        pygame.K_UP: "UP_KEY",
        pygame.K_DOWN: "DOWN_KEY",
        pygame.K_LEFT: "LEFT_KEY",
        pygame.K_RIGHT: "RIGHT_KEY",
        pygame.K_RETURN: "START_KEY",
        pygame.K_ESCAPE: "BACK_KEY",
    }

    def reset_keys(self):
        self.UP_KEY = False
        self.DOWN_KEY = False
        self.LEFT_KEY = False
        self.RIGHT_KEY = False
        self.START_KEY = False
        self.BACK_KEY = False

        self.TEXT_INPUT = ""
        self.BACKSPACE = False

    def check_events(self):
        self.reset_keys()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.playing = False
                pygame.quit()
                return

            if event.type == pygame.TEXTINPUT:
                self.TEXT_INPUT += event.text

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    self._toggle_fullscreen()
                elif event.key in self._MENU_KEY_FLAGS:
                    setattr(self, self._MENU_KEY_FLAGS[event.key], True)
                elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
                    self.BACKSPACE = True


# ===== EJECUTABLE =====

if __name__ == "__main__":
    Game().run()