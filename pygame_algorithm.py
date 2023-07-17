import pygame
import math
import serial
from pygame.locals import *

# 게임 화면 크기 설정
WIDTH, HEIGHT = 800, 600

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("선 따라가기 게임")
    clock = pygame.time.Clock()

    # 시리얼 포트 설정
    ser = serial.Serial('COM1', 9600)  # 포트와 속도를 실제 환경에 맞게 설정
    
    # 시작 위치
    x, y = WIDTH // 2, HEIGHT // 2

    # 선 좌표
    line_points = []

    running = True
    draw_mode = False

    while running:
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_s:
                    # 's' 키를 누르면 맵 저장
                    save_map(line_points)
                elif event.key == K_d:
                    # 'd' 키를 누르면 그리기 모드 전환
                    draw_mode = not draw_mode
                    if not draw_mode:
                        # 그리기 모드 해제 시 선 좌표 초기화
                        line_points = []

            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    # 마우스 왼쪽 버튼을 누르면 그리기 모드 활성화
                    draw_mode = True

            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    # 마우스 왼쪽 버튼을 놓으면 그리기 모드 비활성화
                    draw_mode = False

        if draw_mode:
            # 그리기 모드에서는 선을 그림
            if pygame.mouse.get_pressed()[0]:
                mx, my = pygame.mouse.get_pos()
                if len(line_points) > 0:
                    prev_x, prev_y = line_points[-1]
                    if math.dist((prev_x, prev_y), (mx, my)) > 1:
                        # 선이 끊어짐
                        draw_mode = False
                line_points.append((mx, my))

            pygame.draw.lines(screen, (255, 255, 255), False, line_points, 2)

        if len(line_points) > 1 and not draw_mode:
            # 그리기 모드가 아닐 때 선 따라가기 로직 수행
            target_x, target_y = line_points[0]
            dx = (target_x - x) / 10
            dy = (target_y - y) / 10
            x += dx
            y += dy

            # 다음 점에 도착했는지 확인
            if abs(x - target_x) < 1 and abs(y - target_y) < 1:
                line_points.pop(0)  # 도착한 점 삭제
                if len(line_points) == 1:
                    # 선이 끊어짐
                    continue

            # 현재 진행 각도 계산
            angle = math.atan2(target_y - y, target_x - x)
            angle = math.degrees(angle)
            print("현재 진행 각도:", angle)

            # 현재 진행 각도를 시리얼 포트를 통해 전송
            ser.write(str(angle).encode())  # 전송 형식을 실제 환경에 맞게 수정

        # 선 그리기
        for i in range(len(line_points) - 1):
            pygame.draw.line(screen, (255, 255, 255), line_points[i], line_points[i + 1], 2)

        pygame.display.flip()
        clock.tick(60)

    ser.close()  # 시리얼 포트 닫기
    pygame.quit()

def save_map(line_points):
    # 맵을 파일로 저장하는 함수
    with open('map.txt', 'w') as f:
        for point in line_points:
            f.write(f"{point[0]},{point[1]}\n")

if __name__ == "__main__":
    main()
