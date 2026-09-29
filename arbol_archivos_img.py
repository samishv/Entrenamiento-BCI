# -*- coding: utf-8 -*-

"""
Created on Sun Apr 19 15:34:42 2026

@author: luiso
"""

import os
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIGURACIÓN
# ============================================================

# 1 = Carpetas + archivos
# 2 = Solo carpetas
OPCION_MOSTRAR = 2


# ============================================================
# ELEMENTOS A IGNORAR
# ============================================================

CARPETAS_IGNORADAS = [
    ".git", "DINAM", "EEG", "EMG", "TXT"
]

ARCHIVOS_IGNORADOS = [
    "Thumbs.db",
    ".DS_Store",
]


# ============================================================
# CONFIGURACIÓN DE LA IMAGEN
# ============================================================

# Factor de escala respecto al tamaño de letra (imagen nítida al hacer zoom)
ESCALA = 2

# Tamaño de letra (en px, antes de aplicar ESCALA)
TAMANO_TEXTO = 16

# Alto de línea como múltiplo del tamaño de texto
INTERLINEA = 1.35

# Margen alrededor del contenido (en px, antes de aplicar ESCALA)
MARGEN_HORIZONTAL = 20
MARGEN_VERTICAL = 20

# Colores
COLOR_FONDO = (255, 255, 255)
COLOR_TEXTO = (20, 20, 20)
COLOR_LINEAS = (120, 120, 120)
COLOR_ICONO_CARPETA = (240, 180, 60)
COLOR_ICONO_ARCHIVO = (110, 150, 220)

# Ruta a una fuente monoespaciada que sí incluye los caracteres de
# líneas (├── └── │). Se usa la que trae matplotlib para no depender
# de que el sistema operativo tenga una instalada.
RUTA_FUENTE = fm.findfont(
    fm.FontProperties(family="DejaVu Sans Mono")
)


# ============================================================
# CONSTRUIR JERARQUÍA
# ============================================================

def construir_jerarquia(ruta, prefijo=""):
    """
    Devuelve una lista de tuplas:
        (prefijo_lineas, conector, nombre, es_carpeta)
    en vez de una sola cadena, para poder dibujar el ícono
    de carpeta/archivo por separado del texto.
    """

    elementos_info = []

    try:

        elementos = sorted(
            os.listdir(ruta),
            key=lambda x: (
                not os.path.isdir(os.path.join(ruta, x)),
                x.lower()
            )
        )

    except PermissionError:

        return [
            (prefijo, "└── ", "[Acceso denegado]", False)
        ]


    # --------------------------------------------------------
    # FILTRAR ELEMENTOS
    # --------------------------------------------------------

    elementos_filtrados = []

    for elemento in elementos:

        ruta_completa = os.path.join(
            ruta,
            elemento
        )

        # CARPETA
        if os.path.isdir(ruta_completa):

            if elemento in CARPETAS_IGNORADAS:
                continue

            elementos_filtrados.append(elemento)

        # ARCHIVO
        else:

            if OPCION_MOSTRAR == 2:
                continue

            if elemento in ARCHIVOS_IGNORADOS:
                continue

            elementos_filtrados.append(elemento)


    # --------------------------------------------------------
    # CONSTRUIR ÁRBOL
    # --------------------------------------------------------

    for i, elemento in enumerate(elementos_filtrados):

        ruta_completa = os.path.join(
            ruta,
            elemento
        )

        es_ultimo = (
            i == len(elementos_filtrados) - 1
        )

        conector = (
            "└── "
            if es_ultimo
            else "├── "
        )

        es_carpeta = os.path.isdir(ruta_completa)

        elementos_info.append(
            (prefijo, conector, elemento, es_carpeta)
        )


        # ----------------------------------------------------
        # RECURSIÓN EN CARPETAS
        # ----------------------------------------------------

        if es_carpeta:

            nuevo_prefijo = prefijo + (
                "    "
                if es_ultimo
                else "│   "
            )

            elementos_info.extend(
                construir_jerarquia(
                    ruta_completa,
                    nuevo_prefijo
                )
            )


    return elementos_info


# ============================================================
# DIBUJAR ÍCONOS (sin depender de fuentes de emoji)
# ============================================================

def dibujar_icono_carpeta(draw, x, y, tamano, color):
    """
    Dibuja un ícono simple de carpeta (rectángulo con pestaña)
    centrado verticalmente en (x, y) - (x, y + tamano).
    """

    ancho = tamano
    alto = tamano * 0.78

    y0 = y + (tamano - alto) / 2

    pestana_ancho = ancho * 0.45
    pestana_alto = alto * 0.22

    # Pestaña superior izquierda
    draw.rectangle(
        [
            x,
            y0,
            x + pestana_ancho,
            y0 + pestana_alto
        ],
        fill=color
    )

    # Cuerpo de la carpeta
    draw.rectangle(
        [
            x,
            y0 + pestana_alto,
            x + ancho,
            y0 + alto
        ],
        fill=color
    )


def dibujar_icono_archivo(draw, x, y, tamano, color):
    """
    Dibuja un ícono simple de archivo (rectángulo con esquina
    doblada) centrado verticalmente en (x, y) - (x, y + tamano).
    """

    ancho = tamano * 0.72
    alto = tamano * 0.9

    y0 = y + (tamano - alto) / 2
    doblez = ancho * 0.35

    puntos = [
        (x, y0),
        (x + ancho - doblez, y0),
        (x + ancho, y0 + doblez),
        (x + ancho, y0 + alto),
        (x, y0 + alto),
    ]

    draw.polygon(puntos, fill=color, outline=color)


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    # ========================================================
    # RUTA DE ENTRADA
    # ========================================================

    ruta = (
        r"C:\Users\ikerf\Desktop\Upiita\TT\Entrenamiento-gamificado-BCI"
    )


    # ========================================================
    # RUTA DE SALIDA
    # ========================================================

    ruta_carpeta_salida = r"C:\Users\ikerf\Desktop\Upiita\TT"

    nombre_archivo = "Arbol carpetas.png"

    salida_png = os.path.join(
        ruta_carpeta_salida,
        nombre_archivo
    )


    # ========================================================
    # VALIDAR OPCIÓN
    # ========================================================

    if OPCION_MOSTRAR not in [1, 2]:

        raise ValueError(
            "OPCION_MOSTRAR debe ser 1 o 2."
        )


    # ========================================================
    # CONSTRUIR CONTENIDO
    # ========================================================

    nombre_base = os.path.basename(ruta.rstrip("\\/"))

    jerarquia = construir_jerarquia(ruta)

    # Primera línea: nombre de la carpeta raíz (sin ícono de archivo)
    lineas = [("", "", nombre_base, True)] + jerarquia


    # ========================================================
    # CREAR CARPETA DE SALIDA
    # ========================================================

    os.makedirs(
        ruta_carpeta_salida,
        exist_ok=True
    )


    # ========================================================
    # PREPARAR FUENTE Y MÉTRICAS
    # ========================================================

    tamano_texto_px = TAMANO_TEXTO * ESCALA
    margen_h_px = MARGEN_HORIZONTAL * ESCALA
    margen_v_px = MARGEN_VERTICAL * ESCALA

    fuente = ImageFont.truetype(RUTA_FUENTE, tamano_texto_px)

    # Imagen temporal solo para medir texto
    img_temp = Image.new("RGB", (10, 10))
    draw_temp = ImageDraw.Draw(img_temp)

    alto_linea = int(tamano_texto_px * INTERLINEA)
    icono_tamano = tamano_texto_px * 0.85
    espacio_icono = tamano_texto_px * 1.3


    # --------------------------------------------------------
    # Medir el ancho máximo real de cada línea (prefijo de
    # líneas + conector + espacio para ícono + nombre)
    # --------------------------------------------------------

    ancho_maximo = 0

    textos_medidos = []

    for prefijo, conector, nombre, es_carpeta in lineas:

        texto_lineas = prefijo + conector

        bbox_lineas = draw_temp.textbbox(
            (0, 0), texto_lineas, font=fuente
        )
        ancho_lineas = bbox_lineas[2] - bbox_lineas[0]

        bbox_nombre = draw_temp.textbbox(
            (0, 0), nombre, font=fuente
        )
        ancho_nombre = bbox_nombre[2] - bbox_nombre[0]

        ancho_total = (
            ancho_lineas
            + espacio_icono
            + ancho_nombre
        )

        ancho_maximo = max(ancho_maximo, ancho_total)

        textos_medidos.append(
            (texto_lineas, ancho_lineas, nombre, es_carpeta)
        )


    # ========================================================
    # CALCULAR TAMAÑO EXACTO DE LA IMAGEN
    # ========================================================

    ancho_img = int(ancho_maximo + margen_h_px * 2)
    alto_img = int(len(lineas) * alto_linea + margen_v_px * 2)


    # ========================================================
    # DIBUJAR
    # ========================================================

    img = Image.new("RGB", (ancho_img, alto_img), COLOR_FONDO)
    draw = ImageDraw.Draw(img)

    y = margen_v_px

    for texto_lineas, ancho_lineas, nombre, es_carpeta in textos_medidos:

        x = margen_h_px

        # Caracteres de líneas jerárquicas (├── └── │)
        draw.text(
            (x, y),
            texto_lineas,
            font=fuente,
            fill=COLOR_LINEAS
        )

        x_icono = x + ancho_lineas

        # Ícono (carpeta o archivo)
        if es_carpeta:
            dibujar_icono_carpeta(
                draw, x_icono, y, icono_tamano, COLOR_ICONO_CARPETA
            )
        else:
            dibujar_icono_archivo(
                draw, x_icono, y, icono_tamano, COLOR_ICONO_ARCHIVO
            )

        # Nombre del archivo/carpeta
        x_nombre = x + ancho_lineas + espacio_icono

        draw.text(
            (x_nombre, y),
            nombre,
            font=fuente,
            fill=COLOR_TEXTO
        )

        y += alto_linea


    # ========================================================
    # GUARDAR
    # ========================================================

    img.save(salida_png)


    # ========================================================
    # MENSAJE
    # ========================================================

    if OPCION_MOSTRAR == 1:
        modo = "CARPETAS + ARCHIVOS"
    else:
        modo = "SOLO CARPETAS"


    print("--------------------------------------------")
    print("ÁRBOL DE CARPETAS GENERADO")
    print("--------------------------------------------")
    print(f"Modo: {modo}")
    print(
        f"Carpetas ignoradas: "
        f"{CARPETAS_IGNORADAS}"
    )
    print(
        f"Archivos ignorados: "
        f"{ARCHIVOS_IGNORADOS}"
    )
    print(f"Líneas: {len(lineas)}")
    print(f"Imagen: {salida_png}")
    print(f"Tamaño: {ancho_img}x{alto_img} px")
    print("--------------------------------------------")


main()