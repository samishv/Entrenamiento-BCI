import asyncio
import struct
from bleak import BleakClient, BleakScanner

DEVICE_NAME     = "XIAO-LRA"
CHARACTERISTIC  = "abcd1234-ab12-ab12-ab12-abcdef123456"

def empaquetar(intensidad: int, duracion_ms: int) -> bytes:
    """Empaqueta un pulso en 3 bytes: [intensidad, dur_high, dur_low]"""
    intensidad  = max(0, min(255, intensidad))
    duracion_ms = max(0, min(65535, duracion_ms))
    return struct.pack(">BH", intensidad, duracion_ms)

async def conectar():
    print(f"Buscando '{DEVICE_NAME}'...")
    dispositivo = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10)
    if not dispositivo:
        raise RuntimeError(f"No se encontró '{DEVICE_NAME}'")
    print(f"Conectando a {dispositivo.address}...")
    return dispositivo

async def main():
    dispositivo = await conectar()

    async with BleakClient(dispositivo) as cliente:
        print("✓ Conectado\n")

        # ── Ejemplos de pulsos ────────────────────────────────
        pulsos = [
            (255, 300),   # Intensidad máxima, 300 ms
            (128, 500),   # 50%, 500 ms
            (200, 1000),  # 78%, 1 segundo
            (80,  200),   # Suave, 200 ms
        ]

        for intensidad, duracion in pulsos:
            print(f"→ Enviando: intensidad={intensidad}, duración={duracion}ms")
            datos = empaquetar(intensidad, duracion)
            await cliente.write_gatt_char(CHARACTERISTIC, datos, response=False)
            await asyncio.sleep(duracion / 1000 + 0.5)  # Esperar que termine + margen

        print("\n✓ Secuencia completada")

if __name__ == "__main__":
    asyncio.run(main())