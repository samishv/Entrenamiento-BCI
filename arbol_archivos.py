import os

def construir_jerarquia(ruta, prefijo=""):
    lineas = []
    try:
        elementos = sorted(
            os.listdir(ruta), 
            key=lambda x: (not os.path.isdir(os.path.join(ruta, x)), x.lower())
        )
    except PermissionError:
        return [prefijo + "└── [Acceso denegado]"]
        
    for i, elemento in enumerate(elementos):
        ruta_completa = os.path.join(ruta, elemento)
        es_ultimo = i == len(elementos) - 1
        conector = "└── " if es_ultimo else "├── "
        
        if os.path.isdir(ruta_completa):
            linea = prefijo + conector + "📁 " + elemento
            lineas.append(linea)
            nuevo_prefijo = prefijo + ("    " if es_ultimo else "│   ")
            lineas.extend(construir_jerarquia(ruta_completa, nuevo_prefijo))
        else:
            linea = prefijo + conector + "📄 " + elemento
            lineas.append(linea)
            
    return lineas

if __name__ == "__main__":
    ruta = r"D:\luiso\Documentos\Luis\UPIITA\TT\Entrenamiento-gamificado-BCI"
    
    ruta_carpeta_salida = r"D:\luiso\Escritorio"
    nombre_archivo = "Arbol carpetas.txt"
    salida_txt = os.path.join(ruta_carpeta_salida, nombre_archivo)
    
    nombre_base = os.path.basename(ruta)
    contenido = [nombre_base]
    contenido.extend(construir_jerarquia(ruta))
    
    if not os.path.exists(ruta_carpeta_salida):
        os.makedirs(ruta_carpeta_salida)
        
    with open(salida_txt, "w", encoding="utf-8") as f:
        for linea in contenido:
            f.write(linea + "\n")
            
    print(f"Archivo generado en: {salida_txt}")