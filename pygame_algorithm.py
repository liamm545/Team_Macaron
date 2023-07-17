import pygame
import math
import serial
from pygame.locals import *

# 게임 화면 크기 설정
WIDTH, HEIGHT = 800, 600


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    screen.fill((0, 0, 0))
    pygame.display.set_caption("선 따라가기 게임")
    clock = pygame.time.Clock()

    # 시리얼 포트 설정
    #ser = serial.Serial('COM1', 9600)  # 포트와 속도를 실제 환경에 맞게 설정

    # 시작 위치
    #x, y = WIDTH // 2, HEIGHT // 2

    # 선 좌표
    line_points = []

    running = True
    draw_mode = False
    follow_mode = False

    prev_angle = 0

    while running:

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_s:
                    # 's' 키를 누르면 맵 저장
                    save_map(line_points)
                elif event.key == K_l:
                    # 'l' 키를 누르면 맵 로드
                    line_points = load_map()
                    screen.fill((0, 0, 0))
                    pygame.draw.lines(screen, (255, 255, 255), False, line_points, 1)
                elif event.key == K_d:
                    # 'd' 키를 누르면 그리기 모드 전환
                    draw_mode = not draw_mode
                    if not draw_mode:
                        # 그리기 모드 해제 시 선 좌표 초기화
                        line_points = []
                        screen.fill((0, 0, 0))
                elif event.key == K_SPACE:
                    # 'space' 키를 누르면 따라가기 모드 전환
                    follow_mode = not follow_mode
                    x, y = line_points[0]
                    line_points.pop(0)

            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    # 마우스 왼쪽 버튼을 누르면 그리기 모드 활성화
                    draw_mode = True
                    line_points.append(pygame.mouse.get_pos())  # 현재 마우스 위치 추가

            elif event.type == MOUSEMOTION:
                if draw_mode:
                    # 그리기 모드에서 마우스 움직임을 감지하여 선 추가
                    if(line_points[-1] != pygame.mouse.get_pos()):
                        line_points.append(pygame.mouse.get_pos())  # 현재 마우스 위치 추가

            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    # 마우스 왼쪽 버튼을 놓으면 그리기 모드 비활성화
                    draw_mode = False

        if draw_mode:
            # 그리기 모드에서는 선을 그림
            line_points.append(pygame.mouse.get_pos())
            pygame.draw.lines(screen, (255, 255, 255), False, line_points, 1)

        if len(line_points) > 1 and follow_mode:
            # 따라가기 모드일 때 선 따라가기 로직 수행
            target_x, target_y = line_points[0]
            dx = target_x - x
            dy = target_y - y
            distance = math.hypot(dx, dy)

            if distance > 0:
                speed = 2  # 이동 속도 설정
                steps = min(speed, distance)  # 이동 거리 제한
                angle = math.atan2(dy, dx)
                x += math.cos(angle) * steps
                y += math.sin(angle) * steps

            # 다음 점에 도착했는지 확인
            if abs(x - target_x) < 1 and abs(y - target_y) < 1:
                line_points.pop(0)  # 도착한 점 삭제
                screen.fill((0, 0, 0))
                pygame.draw.lines(screen, (255, 255, 255), False, line_points, 1)
                if len(line_points) < 2:
                    # 선이 부족하여 계속 진행할 수 없음
                    follow_mode = False
                    continue

            # 현재 진행 각도 계산
            angle = math.atan2(target_y - y, target_x - x)
            angle = math.degrees(angle)
            if prev_angle - angle < 10 and prev_angle - angle > -10 :
                prev_angle = angle
            print("현재 진행 각도:", angle)

            # 현재 진행 각도를 시리얼 포트를 통해 전송
            #ser.write(str(angle).encode())  # 전송 형식을 실제 환경에 맞게 수정

        pygame.display.flip()
        clock.tick(60)

    #ser.close()  # 시리얼 포트 닫기
    pygame.quit()


def save_map(line_points):
    # 맵을 파일로 저장하는 함수
    with open('map.txt', 'w') as f:
        for point in line_points:
            f.write(f"{point[0]},{point[1]}\n")

def load_map():
    # 맵 파일을 로드하는 함수
    line_points = []
    try:
        with open('map.txt', 'r') as f:
            for line in f:
                x, y = line.strip().split(',')
                line_points.append((int(x), int(y)))
    except FileNotFoundError:
        print("맵 파일이 존재하지 않습니다.")
    return line_points


if __name__ == "__main__":
    main()
