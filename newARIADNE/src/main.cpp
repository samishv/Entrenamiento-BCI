#include <Arduino.h>

#include <Wire.h>
#include <Adafruit_DRV2605.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// Configuración de pines
#define LED_NARANJITA 2
#define LED_VERDE 3
#define DRV_ENABLE_A D2
#define DRV_ENABLE_B D3 

// ─── UUIDs del servicio GATT ─────────────────────────────────
#define SERVICE_UUID        "12345678-1234-1234-1234-123456789abc"
#define CHARACTERISTIC_UUID "abcd1234-ab12-ab12-ab12-abcdef123456"

// ─── Objetos globales ────────────────────────────────────────
Adafruit_DRV2605 drv;
BLEServer*          pServer         = nullptr;
BLECharacteristic*  pCharacteristic = nullptr;
bool                clienteConectado = false;

// ════════════════════════════════════════════════════════════
//  CALLBACKS BLE
// ════════════════════════════════════════════════════════════
class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) override {
    clienteConectado = true;
    Serial.println("[BLE] Cliente conectado");
    digitalWrite(LED_NARANJITA, LOW);
    digitalWrite(LED_VERDE, HIGH);
  }
  void onDisconnect(BLEServer* pServer) override {
    clienteConectado = false;
    Serial.println("[BLE] Cliente desconectado — reanunciando...");
    drv.setRealtimeValue(0);          // Parar motor al desconectar
    BLEDevice::startAdvertising();
  }
};

class CaracteristicaCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* pChar) override {
    std::string datos = pChar->getValue();

    // Validar que llegaron exactamente 3 bytes
    if (datos.length() != 3) {
      Serial.printf("[BLE] Paquete inválido (%d bytes)\n", datos.length());
      return;
    }

    uint8_t  intensidad    = (uint8_t)datos[0];
    uint16_t duracionMs    = ((uint8_t)datos[1] << 8) | (uint8_t)datos[2];

    Serial.printf("[RTP] Intensidad: %d | Duración: %d ms\n",
                  intensidad, duracionMs);

    // Ejecutar vibración
    drv.setMode(DRV2605_MODE_REALTIME);
    drv.setRealtimeValue(intensidad);
    delay(duracionMs);
    drv.setRealtimeValue(0);
  }
};

// ════════════════════════════════════════════════════════════
//  AUTOCALIBRACIÓN
// ════════════════════════════════════════════════════════════
bool autocalibrar() {
  Serial.println("[CAL] Iniciando autocalibración...");
  delay(2000);

  drv.setMode(DRV2605_MODE_AUTOCAL);
  drv.writeRegister8(DRV2605_REG_FEEDBACK,
    drv.readRegister8(DRV2605_REG_FEEDBACK) | 0x80);
  drv.go();
  delay(1500);

  uint8_t status = drv.readRegister8(DRV2605_REG_STATUS);
  bool exito = !(status & 0x08);

  if (exito) {
    uint8_t periodo = drv.readRegister8(0x22);
    float freq = 1000000.0 / (periodo * 98.46);
    Serial.printf("[CAL] ✓ Frecuencia de resonancia: %.1f Hz\n", freq);
  } else {
    Serial.println("[CAL] ✗ Falló — continuando sin calibración");
  }
  return exito;
}

// ════════════════════════════════════════════════════════════
//  SETUP
// ════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("=== DRV2605 + BLE — XIAO ESP32-C3 ===");

  pinMode(LED_NARANJITA, OUTPUT);
  pinMode(LED_VERDE, OUTPUT);
  pinMode(DRV_ENABLE_A, OUTPUT);
  pinMode(DRV_ENABLE_B, OUTPUT);
  digitalWrite(DRV_ENABLE_A, HIGH); // Habilitar DRV2605
  digitalWrite(DRV_ENABLE_B, HIGH); 
  digitalWrite(LED_NARANJITA, HIGH);
  
  // Iniciar I2C y DRV2605
  Wire.begin(6, 7);
  if (!drv.begin()) {
    Serial.println("[ERROR] DRV2605 no encontrado");
    while (true) delay(1000);
  }
  drv.selectLibrary(6);
  drv.setMode(DRV2605_MODE_INTTRIG);
  autocalibrar();

  // Iniciar BLE
  BLEDevice::init("XIAO-LRA");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new ServerCallbacks());

  BLEService* pService = pServer->createService(SERVICE_UUID);

  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_WRITE
  );
  pCharacteristic->setCallbacks(new CaracteristicaCallbacks());

  pService->start();
  BLEDevice::startAdvertising();

  Serial.println("[BLE] Anunciando como 'XIAO-LRA' — esperando conexión...");
}

// ════════════════════════════════════════════════════════════
//  LOOP
// ════════════════════════════════════════════════════════════
void loop() {
  delay(10);  // BLE se maneja por callbacks, el loop queda libre
}