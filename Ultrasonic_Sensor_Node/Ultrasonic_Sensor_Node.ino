#include <WiFi.h>
#include <WebServer.h>

// 1. SET YOUR WI-FI CREDENTIALS HERE
const char* ssid = "Yakoot";
const char* password = "22032005";

// 2. Ultrasonic Pins
#define TRIG_PIN 14
#define ECHO_PIN 12

WebServer server(80);

void handleDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  
  long duration = pulseIn(ECHO_PIN, HIGH);
  int distance_cm = duration * 0.0343 / 2;
  
  // Send the distance to whoever asks for it
  server.send(200, "text/plain", String(distance_cm));
}

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\nWiFi connected!");
  Serial.print("Sensor IP: ");
  Serial.println(WiFi.localIP());

  server.on("/distance", handleDistance);
  server.begin();
}

void loop() {
  server.handleClient();
}