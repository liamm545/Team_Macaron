#include <Car_Library.h>

int val;  // 수신될 가변 저항 값 저장할 변수
int angle;
int speed;
char rr;

#define NUM_SENSORS 6  // Define the number of sensors

// 초음파 pin
int trigPins[NUM_SENSORS] = {50, 48, 46, 44, 34, 32};  //정면, 우측 정면, 우측 옆면, 후면, 좌측 옆면, 좌측 정면
int echoPins[NUM_SENSORS] = {51, 49, 47, 45 ,35, 33};  //정면, 우측 정면, 우측 옆면, 후면, 좌측 옆면, 좌측 정면

int motor_left_1 = 2;  //왼쪽 뒷바퀴
int motor_left_2 = 3;

int motor_right_1 = 4;  //오른쪽 뒷바퀴
int motor_right_2 = 5;

int motor_steer_1 = 6;  // 앞바퀴 조향
int motor_steer_2 = 7;

int analogPin = A5;  //가변 저항

// Define name of the pins
const char* Strings[NUM_SENSORS] = {"정면", "우측 정면", "우측 옆면", "후면", "좌측 옆면", "좌측 정면"};

long distances[NUM_SENSORS];

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  

  pinMode(motor_left_1, OUTPUT);
  pinMode(motor_left_2, OUTPUT);
  pinMode(motor_right_1, OUTPUT);
  pinMode(motor_right_2, OUTPUT);
  pinMode(motor_steer_1, OUTPUT);
  pinMode(motor_steer_2, OUTPUT);
}

void loop() {
  val = potentiometer_Read(analogPin); 
  //신호등 만났을 때는 멈춰야 됨.
  //신호등 부분 제외하고 직진은 계속 해야하므로, delay 아주 작게 줘서 계속 뒷바퀴는 움직이도록 한다.


  // 조향 부분. 저항값을 받아와서 조향을 하는 거 같음.
    if(Serial.available()) {
      rr = Serial.read();
      if(rr == '1'){
        motor_forward(motor_right_1, motor_right_2, 50);
        motor_backward(motor_left_1, motor_left_2, 50);        
      }else{
        motor_forward(motor_right_1, motor_right_2, 10);
        motor_backward(motor_left_1, motor_left_2, 10);    
      }
    }
}
//    speed = Serial.parseInt();
//    angle = Serial.parseInt();

    //int angle_1 = map(angle, -16, 16, 151, 122);
    //if(speed == 0){  //멈추는 조건. 이거는 
//    motor_hold(motor_right_1, motor_right_2);
//    motor_hold(motor_left_1, motor_left_2);
//    delay(3000);  //몇 초 멈춰야 하는지.
 // }
 // else{
      //motor_forward(motor_right_1, motor_right_2, speed);
      //motor_backward(motor_left_1, motor_left_2, speed);
//      Serial.println(speed);
   //   delay(10);  
 // }
    // right 122, left 152 mid 137
//    Serial.println(val);
//    delay(10);
//    if(val < angle) //여기를 정해진 각도 값으로 해서 조향해야할 듯.
//    {
//      motor_forward(motor_steer_1, motor_steer_2, 40);  //좌회전
//      delay(5);
//      if (val >= angle){
//        motor_hold(motor_steer_1, motor_steer_2);
//        delay(5);
//      }
//    }
//    else if(val > angle)
//    {
//      motor_backward(motor_steer_1, motor_steer_2, 40); //우회전
//      delay(5);
//      if (val <= angle){
//        motor_hold(motor_steer_1, motor_steer_2);
//        delay(5);
//      }
//    }  
//  }


  //예를들어 중간이 120이라고 친다.
  // 측정할 것 : 1. 일단 최대 조향 각도
  // 2. motor_forward함수에서 숫자에 따라 조향되는 속도(시간재서)
  // 그래서 만약 val값이 125라면 적절한 속도로 조향되게.

  
