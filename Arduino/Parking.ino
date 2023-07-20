#include <Car_Library.h>

int val;  // 수신될 가변 저항 값 저장할 변수
int angle;
int bangle;
int speed;
int del;
bool parking_signal = false;
String command = "";

// 초음파 pin
//int trigPins[NUM_SENSORS] = {50, 48, 46, 44, 34, 32};  //정면, 우측 정면, 우측 옆면, 후면, 좌측 옆면, 좌측 정면
//int echoPins[NUM_SENSORS] = {51, 49, 47, 45 ,35, 33};  //정면, 우측 정면, 우측 옆면, 후면, 좌측 옆면, 좌측 정

int trigPins = 46; //우측 옆면만
int echoPins = 47;

int motor_left_1 = 2;  //왼쪽 뒷바퀴
int motor_left_2 = 3;

int motor_right_1 = 4;  //오른쪽 뒷바퀴
int motor_right_2 = 5;

int motor_steer_1 = 6;  // 앞바퀴 조향
int motor_steer_2 = 7;

int analogPin = A5;  //가변 저항

// Define name of the pins
const char* Strings = "정면";

long distances;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  Serial.setTimeout(50);
  pinMode(trigPins, OUTPUT);
  pinMode(echoPins, INPUT);

  pinMode(motor_left_1, OUTPUT);
  pinMode(motor_left_2, OUTPUT);
  pinMode(motor_right_1, OUTPUT);
  pinMode(motor_right_2, OUTPUT);
  pinMode(motor_steer_1, OUTPUT);
  pinMode(motor_steer_2, OUTPUT);

}


void executeCommand(String cmd) {
  val = potentiometer_Read(analogPin); 
  distances = ultrasonic_distance(trigPins, echoPins);
  speed = 80;
  if (cmd.startsWith("A:")) {
    motor_forward(motor_right_1, motor_right_2, speed);
    motor_backward(motor_left_1, motor_left_2, speed);
    
    cmd.remove(0, 2);
    cmd.remove(cmd.length() - 1);
    int angle = cmd.toInt();
    
    Serial.println("A");
    
    if (distances <= 500 && parking_signal == false){
    Serial.println("P");  // 파이썬으로 주차 시작 신호인 'P' 신호를 보냄
    motor_hold(motor_right_1, motor_right_2);
    motor_hold(motor_left_1, motor_left_2);
    parking_signal = true;  // Update the flag
  }
    angle = map(angle, -16, 16, 150, 121);
    
    if(val <= angle)
    {
      motor_forward(motor_steer_1, motor_steer_2, 140);  //좌회전
      
      if (val >= angle){
        motor_hold(motor_steer_1, motor_steer_2);
      }
    }
    
    else if(val >= angle)  //
    {
      motor_backward(motor_steer_1, motor_steer_2, 140); //우회전
      
      if (val <= angle){
        motor_hold(motor_steer_1, motor_steer_2);
      }    
    }  
  } 
  
  else if(cmd.startsWith("P:")) {
      motor_forward(motor_left_1, motor_left_2, speed);
      motor_backward(motor_right_1, motor_right_2, speed);
      cmd.remove(0, 2);
      cmd.remove(cmd.length() - 1);
      int angle = cmd.toInt();
      
      if (angle == 90) {
        motor_hold(motor_right_1, motor_right_2);
        motor_hold(motor_left_1, motor_left_2);
      }
      angle = map(angle, -16, 16, 150, 121);
      
      if(val <= angle)
      {
        motor_forward(motor_steer_1, motor_steer_2, 140);  //좌회전
        
        if (val >= angle){
          motor_hold(motor_steer_1, motor_steer_2);
        }
      }
      
      else if(val >= angle)  //
      {
        motor_backward(motor_steer_1, motor_steer_2, 140); //우회전
        
        if (val <= angle){
          motor_hold(motor_steer_1, motor_steer_2);
        }    
      }  
  }
}
void loop() {
  while (Serial.available()){

    char ch = Serial.read();
    command += ch;
    if(ch == ';') {
      executeCommand(command);
      command = "";
    }
  }
}