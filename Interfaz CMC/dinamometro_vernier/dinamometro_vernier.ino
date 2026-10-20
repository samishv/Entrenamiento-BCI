// Hand Dynamometer Vernier HD-BTA (HD462DR2) -> Arduino Uno
// Envía la fuerza en N, SIN filtrar, a FS_HZ (una línea por muestra).
// Comandos por serial:  's' = iniciar envío   |   'p' = pausar envío
// Imprime "READY" al terminar setup() para que Python sepa cuándo puede empezar.
//
// La tara (cero) y el filtrado se hacen en Python (protocolo_cmc.py), de modo que
// el stream guardado conserva la señal original.

const int SENSOR_PIN = A0;
const unsigned long BAUD_RATE = 115200;

// Debe coincidir con FS_DINO en protocolo_cmc.py
const float FS_HZ = 200.0;
const unsigned long SAMPLE_INTERVAL_US = (unsigned long)(1000000.0 / FS_HZ);

// Lecturas consecutivas del ADC que se promedian por muestra (anti-ruido casi sin retardo:
// 8 lecturas ≈ 1 ms con el prescaler por defecto)
const int N_OVERSAMPLE = 8;

const float VOLTAGE_REF = 5.0;
const float ADC_MAX = 1023.0;

// Calibración oficial HD-BTA
const float SLOPE = 176.8325;      // N/V
const float INTERCEPT = -19.5057;  // N

bool sendingData = false;
unsigned long nextSample = 0;

float leerFuerza() {
  unsigned long suma = 0;
  for (int i = 0; i < N_OVERSAMPLE; i++) {
    suma += analogRead(SENSOR_PIN);
  }
  float adc = suma / (float)N_OVERSAMPLE;
  float voltage = adc * VOLTAGE_REF / ADC_MAX;
  return voltage * SLOPE + INTERCEPT;     // sin recorte de negativos
}

void setup() {
  Serial.begin(BAUD_RATE);
  pinMode(SENSOR_PIN, INPUT);
  analogRead(SENSOR_PIN);                 // descarta la primera lectura
  Serial.println("READY");
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 's' || c == 'S') {
      if (!sendingData) {
        sendingData = true;
        nextSample = micros() + SAMPLE_INTERVAL_US;
      }
    } else if (c == 'p' || c == 'P') {
      sendingData = false;
    }
  }

  // Diferencia con signo: seguro ante el desbordamiento de micros()
  if (sendingData && (long)(micros() - nextSample) >= 0) {
    nextSample += SAMPLE_INTERVAL_US;     // sin acumular deriva
    Serial.println(leerFuerza(), 2);
  }
}
