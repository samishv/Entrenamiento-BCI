"""
uml_generator.py
----------------
Genera automáticamente un diagrama de clases UML en PNG a partir de
archivos .py de Python.

Uso:
    python uml_generator.py <ruta>

    <ruta> puede ser:
      - Un archivo .py individual  →  analiza ese archivo
      - Una carpeta               →  analiza todos los .py en ella (recursivo)

Salida:
    uml_diagram.png  en el mismo directorio donde se ejecuta el script.

Dependencias (solo stdlib + matplotlib):
    pip install matplotlib
"""

import ast
import sys
import os
import textwrap
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ─────────────────────────────────────────────
# Estructuras de datos
# ─────────────────────────────────────────────

@dataclass
class MethodInfo:
    name: str
    args: list[str]
    returns: Optional[str] = None
    is_static: bool = False
    is_class_method: bool = False
    visibility: str = "+"          # + público  - privado  # protegido

@dataclass
class ClassInfo:
    name: str
    bases: list[str] = field(default_factory=list)
    methods: list[MethodInfo] = field(default_factory=list)
    attributes: list[tuple[str, Optional[str]]] = field(default_factory=list)  # (nombre, tipo)
    source_file: str = ""

# ─────────────────────────────────────────────
# Extracción AST
# ─────────────────────────────────────────────

def get_visibility(name: str) -> str:
    if name.startswith("__") and not name.endswith("__"):
        return "-"
    if name.startswith("_"):
        return "#"
    return "+"

def annotation_to_str(node) -> Optional[str]:
    """Convierte un nodo de anotación AST a string legible."""
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None

def extract_classes(source: str, filename: str) -> list[ClassInfo]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    classes = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        cls = ClassInfo(name=node.name, source_file=filename)

        # Herencia
        for base in node.bases:
            cls.bases.append(annotation_to_str(base) or "?")

        # Atributos de clase (anotaciones a nivel de clase)
        attr_names_seen = set()
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                attr_name = item.target.id
                attr_type = annotation_to_str(item.annotation)
                if attr_name not in attr_names_seen:
                    cls.attributes.append((attr_name, attr_type))
                    attr_names_seen.add(attr_name)

        # Métodos
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            method_name = item.name
            visibility = get_visibility(method_name)

            # Decoradores
            is_static = any(
                (isinstance(d, ast.Name) and d.id == "staticmethod")
                for d in item.decorator_list
            )
            is_classmethod = any(
                (isinstance(d, ast.Name) and d.id == "classmethod")
                for d in item.decorator_list
            )

            # Argumentos (sin self/cls)
            args = []
            for arg in item.args.args:
                if arg.arg in ("self", "cls"):
                    continue
                ann = annotation_to_str(arg.annotation)
                args.append(f"{arg.arg}: {ann}" if ann else arg.arg)

            returns = annotation_to_str(item.returns)

            # Atributos de instancia definidos en __init__
            if method_name == "__init__":
                for stmt in ast.walk(item):
                    if isinstance(stmt, ast.Assign):
                        for t in stmt.targets:
                            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                                if t.attr not in attr_names_seen:
                                    cls.attributes.append((t.attr, None))
                                    attr_names_seen.add(t.attr)
                    elif isinstance(stmt, ast.AnnAssign):
                        if isinstance(stmt.target, ast.Attribute) and isinstance(stmt.target.value, ast.Name) and stmt.target.value.id == "self":
                            attr_name = stmt.target.attr
                            if attr_name not in attr_names_seen:
                                cls.attributes.append((attr_name, annotation_to_str(stmt.annotation)))
                                attr_names_seen.add(attr_name)

            minfo = MethodInfo(
                name=method_name,
                args=args,
                returns=returns,
                is_static=is_static,
                is_class_method=is_classmethod,
                visibility=visibility,
            )
            cls.methods.append(minfo)

        classes.append(cls)

    return classes

def collect_files(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file() and p.suffix == ".py":
        return [p]
    if p.is_dir():
        return sorted(p.rglob("*.py"))
    return []

def parse_all(path: str) -> list[ClassInfo]:
    all_classes = []
    for fp in collect_files(path):
        try:
            source = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        all_classes.extend(extract_classes(source, fp.name))
    return all_classes

# ─────────────────────────────────────────────
# Layout  (empaquetado en grilla simple)
# ─────────────────────────────────────────────

FONT_SIZE      = 8.5
LINE_H         = 0.38          # altura por línea de texto (unidades de datos)
PADDING        = 0.18
BOX_W          = 4.2           # ancho fijo de cada caja
COL_GAP        = 1.2
ROW_GAP        = 1.0
COLS           = 3             # clases por fila

# Colores (tema oscuro coherente con el resto del proyecto)
BG_COLOR       = "#0D1117"
CLASS_HEADER   = "#1f2d3d"
CLASS_BODY     = "#161B22"
CLASS_BORDER   = "#6FC7B7"
TEXT_HEADER    = "#BFEEF6"
TEXT_ATTR      = "#a4e1b6"
TEXT_METHOD    = "#99b2ff"
TEXT_TYPE      = "#8fc3eb"
SEP_COLOR      = "#41CCE3"
ARROW_INH      = "#6FC7B7"
ARROW_DEP      = "#fefb6e"

def wrap(text: str, width: int = 38) -> list[str]:
    return textwrap.wrap(text, width=width) or [text]

def measure_class(cls: ClassInfo) -> float:
    """Devuelve la altura necesaria para la caja (en unidades de datos)."""
    # Encabezado
    h = LINE_H + PADDING * 2        # nombre
    # Atributos
    for name, typ in cls.attributes:
        label = f"{get_visibility(name)} {name}" + (f": {typ}" if typ else "")
        h += LINE_H * len(wrap(label))
    h += PADDING  # separador
    # Métodos
    for m in cls.methods:
        sig = method_sig(m)
        h += LINE_H * len(wrap(sig))
    h += PADDING * 2
    return h

def method_sig(m: MethodInfo) -> str:
    prefix = ""
    if m.is_static:
        prefix = "«static» "
    elif m.is_class_method:
        prefix = "«cls» "
    args_str = ", ".join(m.args) if m.args else ""
    ret = f": {m.returns}" if m.returns else ""
    return f"{m.visibility} {prefix}{m.name}({args_str}){ret}"

def compute_layout(classes: list[ClassInfo]) -> tuple[list[dict], float, float]:
    """Devuelve lista de {cls, x, y, w, h}, ancho total, alto total."""
    boxes = []
    col = 0
    row_heights: list[float] = []
    max_h_in_row = 0.0
    cur_x = 0.0
    cur_y = 0.0

    for i, cls in enumerate(classes):
        if col == COLS:
            col = 0
            cur_y -= max_h_in_row + ROW_GAP
            row_heights.append(max_h_in_row)
            max_h_in_row = 0.0
            cur_x = 0.0

        h = measure_class(cls)
        boxes.append({"cls": cls, "x": cur_x, "y": cur_y, "w": BOX_W, "h": h})
        max_h_in_row = max(max_h_in_row, h)
        cur_x += BOX_W + COL_GAP
        col += 1

    # Dimensiones totales
    total_w = COLS * BOX_W + (COLS - 1) * COL_GAP
    total_h = abs(cur_y) + max_h_in_row
    return boxes, total_w, total_h

# ─────────────────────────────────────────────
# Renderizado matplotlib
# ─────────────────────────────────────────────

def draw_diagram(classes: list[ClassInfo], output_path: str):
    if not classes:
        print("⚠  No se encontraron clases en los archivos proporcionados.")
        return

    boxes, total_w, total_h = compute_layout(classes)

    margin = 1.5
    fig_w = total_w + margin * 2
    fig_h = total_h + margin * 2
    dpi = 150

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-margin, total_w + margin)
    ax.set_ylim(-total_h - margin, margin)
    ax.axis("off")

    # Índice nombre→box para dibujar flechas
    name_to_box = {b["cls"].name: b for b in boxes}

    # ── Cajas ──────────────────────────────────────────────────────────────
    for b in boxes:
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        cur_y_text = y - PADDING

        # Borde exterior
        rect = FancyBboxPatch(
            (x, y - h), w, h,
            boxstyle="round,pad=0.05",
            linewidth=1.5,
            edgecolor=CLASS_BORDER,
            facecolor=CLASS_BODY,
            zorder=2,
        )
        ax.add_patch(rect)

        # Encabezado coloreado
        header_h = LINE_H + PADDING
        header = FancyBboxPatch(
            (x, y - header_h), w, header_h,
            boxstyle="round,pad=0.05",
            linewidth=0,
            edgecolor="none",
            facecolor=CLASS_HEADER,
            zorder=3,
        )
        ax.add_patch(header)

        # Nombre de la clase
        ax.text(
            x + w / 2, cur_y_text - LINE_H / 2,
            b["cls"].name,
            ha="center", va="center",
            fontsize=FONT_SIZE + 1,
            fontweight="bold",
            color=TEXT_HEADER,
            zorder=4,
        )
        cur_y_text -= LINE_H + PADDING

        # Separador 1
        ax.plot([x + 0.1, x + w - 0.1], [cur_y_text, cur_y_text],
                color=SEP_COLOR, lw=0.8, zorder=4)

        # Atributos
        for attr_name, attr_type in b["cls"].attributes:
            vis = get_visibility(attr_name)
            label = f"{vis} {attr_name}"
            type_str = f": {attr_type}" if attr_type else ""
            for k, line in enumerate(wrap(label + type_str)):
                ax.text(
                    x + PADDING, cur_y_text - LINE_H / 2,
                    line,
                    ha="left", va="center",
                    fontsize=FONT_SIZE - 0.5,
                    color=TEXT_ATTR if k == 0 else TEXT_TYPE,
                    zorder=4,
                )
                cur_y_text -= LINE_H

        cur_y_text -= PADDING * 0.5
        # Separador 2
        ax.plot([x + 0.1, x + w - 0.1], [cur_y_text, cur_y_text],
                color=SEP_COLOR, lw=0.8, zorder=4)

        # Métodos
        for m in b["cls"].methods:
            sig = method_sig(m)
            for line in wrap(sig):
                ax.text(
                    x + PADDING, cur_y_text - LINE_H / 2,
                    line,
                    ha="left", va="center",
                    fontsize=FONT_SIZE - 0.5,
                    color=TEXT_METHOD,
                    zorder=4,
                )
                cur_y_text -= LINE_H

    # ── Flechas de herencia ─────────────────────────────────────────────────
    for b in boxes:
        for base in b["cls"].bases:
            if base in name_to_box:
                pb = name_to_box[base]
                # Punto de inicio: centro superior de la caja hija
                x0 = b["x"] + b["w"] / 2
                y0 = b["y"]
                # Punto de fin: centro inferior de la caja padre
                x1 = pb["x"] + pb["w"] / 2
                y1 = pb["y"] - pb["h"]
                ax.annotate(
                    "",
                    xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=ARROW_INH,
                        lw=1.5,
                        mutation_scale=14,
                    ),
                    zorder=1,
                )

    # ── Título y leyenda ────────────────────────────────────────────────────
    ax.set_title(
        "Diagrama de Clases UML",
        fontsize=13, fontweight="bold",
        color=TEXT_HEADER, pad=10,
    )

    legend_patches = [
        mpatches.Patch(color=TEXT_ATTR, label="Atributos"),
        mpatches.Patch(color=TEXT_METHOD, label="Métodos"),
        mpatches.Patch(color=ARROW_INH, label="Herencia (→)"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        fontsize=7.5,
        facecolor="#1f2d3d",
        edgecolor=CLASS_BORDER,
        labelcolor="white",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"✅  Diagrama guardado en: {output_path}")

# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN  ←  edita solo este bloque antes de darle Run en Spyder
# ═══════════════════════════════════════════════════════════════════════════════

# Ruta al archivo .py o a la carpeta que quieres analizar.
# Puedes usar ruta absoluta con r"..." para no preocuparte por las barras.
INPUT_PATH = r"D:\luiso\Documentos\Luis\UPIITA\TT\Entrenamiento-BCI\Juego\game.py"

# Ruta y nombre del PNG de salida.
# Por defecto se guarda en la misma carpeta que el archivo/carpeta analizado.
# Cámbiala si quieres guardarlo en otro lugar.
OUTPUT_PATH = ""   # déjala vacía ("") para usar la ruta automática

# ═══════════════════════════════════════════════════════════════════════════════

def main():
    input_path = INPUT_PATH.strip()

    # Ruta de salida: si OUTPUT_PATH está vacío, se construye automáticamente
    if OUTPUT_PATH.strip():
        out_file = Path(OUTPUT_PATH.strip())
    else:
        base = Path(input_path)
        out_dir = base.parent if base.is_file() else base
        out_file = out_dir / "uml_diagram.png"

    if not os.path.exists(input_path):
        print(f"❌  Ruta no encontrada: {input_path}")
        print("    Verifica que INPUT_PATH sea correcto.")
        return

    print(f"🔍  Analizando: {input_path}")
    classes = parse_all(input_path)

    if not classes:
        print("⚠  No se encontraron clases. Verifica que los archivos sean .py válidos.")
        return

    print(f"📦  Clases encontradas: {len(classes)}")
    for c in classes:
        print(f"    • {c.name}  ({c.source_file})  bases={c.bases}")

    draw_diagram(classes, str(out_file))

main()