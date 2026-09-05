/*
 * Eye controlled robot: reads single-character direction commands from serial
 * and drives two DC motors through an L293D motor driver.
 *
 * Wiring:
 *   MOT1F -> D4    MOT1R -> D3     (left motor forward / reverse)
 *   MOT2F -> D11   MOT2R -> D10    (right motor forward / reverse)
 *   L293D powered from a 9V battery, Arduino Uno over USB.
 *
 * Originally uploaded as BLUETOOTH_CONTROL_ROBOT.ino.
 */

int const MOT1F = 4;
int const MOT1R = 3;

int const MOT2F = 11;
int const MOT2R = 10;

void setup() {
  pinMode(MOT1F, OUTPUT);
  pinMode(MOT1R, OUTPUT);
  pinMode(MOT2F, OUTPUT);
  pinMode(MOT2R, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    switch (Serial.read()) {
      case 'f':
        digitalWrite(MOT1F, HIGH);
        digitalWrite(MOT1R, LOW);
        digitalWrite(MOT2F, HIGH);
        digitalWrite(MOT2R, LOW);
        break;

      case 'r':
        digitalWrite(MOT1F, HIGH);
        digitalWrite(MOT1R, LOW);
        digitalWrite(MOT2F, LOW);
        digitalWrite(MOT2R, LOW);
        break;

      case 'l':
        digitalWrite(MOT1F, LOW);
        digitalWrite(MOT1R, LOW);
        digitalWrite(MOT2F, HIGH);
        digitalWrite(MOT2R, LOW);
        break;

      case 's':
        digitalWrite(MOT1F, LOW);
        digitalWrite(MOT1R, LOW);
        digitalWrite(MOT2F, LOW);
        digitalWrite(MOT2R, LOW);
        break;
    }
  }
}
