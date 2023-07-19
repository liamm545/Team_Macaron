#include <Car_Library.h>

int val;  // 수신될 가변 저항 값 저장할 변수
int angle;
int bangle;
int speed;
int del;
bool parking_signal = false;

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

void motor_straight(speed) {
  motor_forward(motor_right_1, motor_right_2, speed);
  motor_backward(motor_left_1, motor_left_2, speed);
}
void loop() {
  val = potentiometer_Read(analogPin); 
  
  speed = 1;
  motor_straight(speed);
  
  distances = ultrasonic_distance(trigPins, echoPins);
  
  if (Serial.available()){

    if (distances <= "Tunning"){
    Serial.println("P");  // 파이썬으로 주차 시작 신호인 'P' 신호를 보냄
    motor_hold(motor_steer_1, motor_steer_2);
    parking_signal = true;  // Update the flag
  }
  
    del = Serial.parseInt();
    angle = Serial.parseInt();

    int angle_1 = map(angle, -17, 16, 151, 120);
    
    if(speed == 0){  //멈추는 조건. 이거는 
      motor_hold();
      delay(300);  //몇 초 멈춰야 하는지.
    }

    // 파이게임

    if(angle >= 24 && angle <= 56){   // 후진
      motor_forward(motor_left_1, motor_left_2, speed);
      motor_backward(motor_right_1, motor_right_2, speed);
    }
    
    else{   //직진 -> 평소에 쓸 친구
       motor_straight(speed);
    }

    if(val <= angle_1)   //일반 주행일 때
    {
      motor_forward(motor_steer_1, motor_steer_2, 70);  //좌회전
      if (val >= angle_1){
        motor_hold(motor_steer_1, motor_steer_2);
      }
    }
    else if(val >= angle_1)  //
    {
      motor_backward(motor_steer_1, motor_steer_2, 70); //우회전
      if (val <= angle_1){
        motor_hold(motor_steer_1, motor_steer_2);
      }    
    }  
    
    delay(10);
  }
}
