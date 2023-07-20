import argparse
from pathlib import Path
import math
import time  #pygame library
import pygame
import serial
from pygame.locals import *

# 시리얼 통신 설정
serial_ = 1

WIDTH, HEIGHT = 800, 600
game_once = 0;

if serial_:  #######################
    ser = serial.Serial('COM3', 9600)

def send_data1(steering_angle):  #######################
    message = "A:" + str(steering_angle) + ";"
    ser.write(message.encode())

def send_data2(steering_angle):  #######################
    message = "P:" + str(steering_angle) + ";"
    ser.write(message.encode())



def load_map():
    # 맵 파일을 로드하는 함수
    line_points = []
    direction_points = []
    try:
        with open('map_obstacle.txt', 'r') as f:
            for line in f:
                x, y = line.strip().split(',')
                line_points.append((int(x), int(y)))
        with open('map_obstacle_direct.txt', 'r') as f:
            for line in f:
                x = line.strip()
                direction_points.append((int(x)))
    except FileNotFoundError:
        print("맵이 없다.")
    return line_points, direction_points

def save_map(line_points, direction_points):
    # 맵을 파일로 저장하는 함수
    with open('map_obstacle.txt', 'w') as f:
        for point in line_points:
            f.write(f"{point[0]},{point[1]}\n")
    #맵의 dir을 저장하는 함수
    with open('map_obstacle_direct.txt', 'w') as f:
        for point in direction_points:
            f.write(f"{point}\n")

while True:
    ultra = ser.readline().decode()
    if ultra < 1000 and game_once == 0:
        game_once = 1
        # pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        screen.fill((0, 0, 0))
        # pygame.display.set_caption("회피에 미친놈")
        clock = pygame.time.Clock()

        # 선 좌표
        line_points = []
        direction_points = []

        # 멥 불러오자.
        line_points, direction_points = load_map()
        x, y = line_points[0]

        running = True
        draw_mode = False
        follow_mode = False
        break_p = False
        now_dir = 1
        counter = 0

        prev_angle = []
        last_angle = 0

        angle = 0
        while running:
            
            if len(line_points) > 1:

                target_x, target_y = line_points[0]
                dirct = direction_points[0]
                dx = target_x - x
                dy = target_y - y
                distance = math.hypot(dx, dy)

                if distance > 0:
                    speed = 0.7  # 이동 속도 설정
                    steps = min(speed, distance)  # 이동 거리 제한
                    angle = math.atan2(dy, dx)
                    x += math.cos(angle) * steps
                    y += math.sin(angle) * steps

                # 다음 점에 도착했는지 확인
                if abs(x - target_x) < 1 and abs(y - target_y) < 1:
                    line_points.pop(0)  # 도착한 점 삭제
                    direction_points.pop(0)
                    screen.fill((0, 0, 0))
                    # pygame.draw.lines(screen, (255, 255, 255), False, line_points, 1)
                    if len(line_points) < 2:
                        # 선이 부족하여 계속 진행할 수 없음
                        follow_mode = False
                        continue
                    if (direction_points[1] == 0):
                        send_data(90)
                        time.sleep(5)
                        direction_points.pop(1)

                # 현재 진행 각도 계산
                angle = math.atan2(target_y - y, target_x - x)
                angle = math.degrees(angle)
                prev_angle.append(angle)
                if len(prev_angle) > 5:
                    prev_angle.pop(0)
                angle = max(prev_angle)
                use_angle = angle - last_angle

                last_angle = angle
                # 현재 진행 각도를 시리얼 포트를 통해 전송

                if (angle > 16):
                    angle = 16
                elif (angle < -16):
                    angle = -16
                else:
                    angle = angle

                if (angle < 4 and angle > -4):
                    angle = 0
                if (direction_points[1] == -1):
                    angle = angle + 40
                if (angle <= 16) and (angle >= -16):
                    send_data1(angle)
                else:
                    angle -= 40
                    send_data2(angle)  # 전송 형식을 실제 환경에 맞게 수정
                print("현재 진행 각도:", angle)
                print("\n진행방향: ", dirct)
            
            # pygame.display.flip()
            clock.tick(60)

