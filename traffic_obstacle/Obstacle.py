import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import math
import matplotlib as plt
import time  #pygame library
import pygame
import math
import serial
from pygame.locals import *

sensitivity = 60
lower_white = np.array([0, 0, 255 - sensitivity])
upper_white = np.array([255, sensitivity, 255])

# 게임 화면 크기 설정
WIDTH, HEIGHT = 800, 600
game_once = 0;

# 시리얼 통신 설정
serial_ = 0

if serial_:  #######################
    ser = serial.Serial('COM3', 9600)

right_save = [0] * 12  # 숫자 수정 필요

from utils.utils import \
    time_synchronized, select_device, increment_path, \
    scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model, \
    lane_line_mask, plot_one_box, show_seg_result, \
    AverageMeter, \
    LoadImages

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/1.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser

def send_data(steering_angle):  #######################
    # steering_angle = str(steering_angle)
    # data = "Q" + steering_angle
    # ser.write(data.encode("utf-
    data = str(steering_angle) + '\n'
    ser.write(data.encode())

class PID():
    def __init__(self, kp, ki, kd):
        self.Kp = kp  # 비례 Proportional
        self.Ki = ki  # 적분 Integral
        self.Kd = kd  # 미분 Derivative
        self.p_error = 0.0  # 이전 오차 (비례 제어)
        self.i_error = 0.0  # 누적된 오차 (적분 제어)
        self.d_error = 0.0  # 오차의 변화 (미분 제어)

    def pid_control(self, cte):
        """
        주어진 오차값을 기반으로 PID control 수행

        Args:
            cte (float): 오 값 (cross-track error).

        Returns:
            float: PID 제어 결과값
        """
        self.d_error = cte - self.p_error  # 오차의 변화량 계산
        self.p_error = cte  # 현재 오차를 이전 오차로 설정
        self.i_error += cte  # 시간에 따른 오차 누적

        # 비례, 적분, 미분 항을 합산하여 제어 출력 값 반환
        return self.Kp * self.p_error + self.Ki * self.i_error * self.Kd * self.d_error

def plothistogram(image):
    """
        histogram을 통해 2개의 차선 위치를 추출해주는 함수

        Return
        1) leftbase : left lane pixel coords
        2) rightbase : right lane pixel coords
    """
    histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
    midpoint = np.int_(histogram.shape[0] / 2)
    # leftbase = np.argmax(histogram[:midpoint])
    rightbase = np.argmax(histogram[:])
    #print(rightbase)
    return rightbase, histogram

def cal_exactly(array):
    # 중점좌표 구하기
    xmm = 90.0 / 600
    ymm = 98.0 / 780
    rm = ((array[-1] * xmm + array[1] * xmm) / 2, 780*ymm/2)

    # 곡선 기울기
    ra = (array[-1] * xmm - array[1] * xmm) / 98

    # 원의 중심좌표
    ocenter = (-102.5 / ra + rm[0], -53.5)

    # 조향 기울기
    vl = 53.5 / (300 * xmm + 102.5 / ra - rm[0])

    if(vl > 0):
        vl = -math.atan(vl) * 180 / math.pi
    else:
        vl = math.atan(vl) * 180 / math.pi

    return vl

def slide_window_search(binary_warped, right_current):
    """
        histogram을 통해 파악된 차선의 위치에 sliding window search 적용하는 함수

        Return (ret)
        1) left_fitx : 왼쪽 차선에 해당하는 픽셀 x좌표들
        2) right_fitx : 오른쪽 차선에 해당하는 픽셀 x좌표들
        3) ploty : 차선에 해당하는 픽셀 y좌표들
    """
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    nwindows = 18
    window_height = np.int_(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()  # 선이 있는 부분의 인덱스만 저장
    nonzero_y = np.array(nonzero[0])  # 선이 있는 부분 y의 인덱스 값
    nonzero_x = np.array(nonzero[1])  # 선이 있는 부분 x의 인덱스 값
    margin = 100
    minpix = 50
    left_lane = []
    right_lane = []
    color = [0, 255, 0]
    thickness = 2
    mid = [0] * nwindows
    dot_save = [0] * (nwindows-5)  # 차선의 값을 저장해둠.

    for w in range(nwindows-5):
        win_y_low = binary_warped.shape[0] - (w + 1) * window_height  # window 윗부분
        win_y_high = binary_warped.shape[0] - w * window_height  # window 아랫 부분
        # win_xleft_low = left_current - margin  # 왼쪽 window 왼쪽 위
        # win_xleft_high = left_current + margin  # 왼쪽 window 오른쪽 아래
        win_xright_low = right_current - margin  # 오른쪽 window 왼쪽 위
        win_xright_high = right_current + margin  # 오른쪽 window 오른쪽 아래
        # cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)

        # good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (
        #             nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (
                nonzero_x < win_xright_high)).nonzero()[0]
        # left_lane.append(good_left)
        right_lane.append(good_right)

        # if len(good_left) > minpix:
        #     left_current = np.int_(np.mean(nonzero_x[good_left]))
        if len(good_right) > minpix:
            right_current = np.int_(np.mean(nonzero_x[good_right]))
        if len(good_right) <= minpix: #차선 검출을 못했을 경우
            if any(dot_save):
                prev_slopes = [right_current - prev_dot for prev_dot in dot_save if prev_dot > 0]
                mean_slope = np.mean(prev_slopes)
                right_current += np.int_(mean_slope)

        # mid[w] = (right_current + left_current)/2
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
        cv2.line(out_img, (right_current, win_y_high), (right_current, win_y_high), [0, 255, 255], 5)
        dot_save[w] = right_current
    # print(dot_save)
    # left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침
    right_lane = np.concatenate(right_lane)

    # leftx = nonzero_x[left_lane]
    # lefty = nonzero_y[left_lane]
    rightx = nonzero_x[right_lane]
    righty = nonzero_y[right_lane]
    #
    # left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    # left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # ltx = np.trunc(left_fitx)  # np.trunc() -> 소수점 부분을 버림
    rtx = np.trunc(right_fitx)

    # out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
    out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]
    # out_img[ploty.astype('int'), right_fitx.astype('int')] = [0, 255, 255]
    # plt.show(out_img)
    # plt.plot(left_fitx, ploty, color = 'yellow')
    # plt.plot(right_fitx, ploty, color = 'yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()

    ret = {'right_fitx': rtx, 'ploty': ploty}

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 19.1 / 720  # centimeters per pixel in y dimension
    xm_per_pix = 32.5 / 1000  # centimeters per pixel in x dimension
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    # left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    # left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
    #     2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')
    # print(right_curverad, 'cm')
    right_curverad = 1 / right_curverad  # 곡률
    str__ = "curve : %f" % right_curverad
    out_img = cv2.putText(out_img, str__, (350, 40), cv2.FONT_HERSHEY_PLAIN, 2, [255, 0, 0], 1, cv2.LINE_AA)  # 반지름 표시

    line_1 = right_fit[0] * ploty * 2 + right_fit[1]
    # print(-np.mean(line_1))
    line_1 = -np.mean(line_1)  # 기울기
    # print(line_1)
    return out_img, right_curverad, line_1, dot_save

# 메인 실행 코드
def detect():
    # setting and directories
    source, weights, save_txt, imgsz = opt.source, opt.weights, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    save_mid = [None, None]

    current_lane = 'right'
    mid_standard = 544

    max_area = 0

    global game_once

    # Load model
    stride = 32
    model = torch.jit.load(weights)
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = model.to(device)

    if half:
        model.half()  # to FP16
    model.eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()

        [pred, anchor_grid], seg, ll = model(img)
        hsv = cv2.cvtColor(im0s, cv2.COLOR_BGR2HSV)
        ll_white = cv2.inRange(hsv, lower_white, upper_white)
        t2 = time_synchronized()

        # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version
        # but this problem will not appear in offical version
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred, anchor_grid)
        tw2 = time_synchronized()

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        # da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = 1  # to Path
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    area = plot_one_box(xyxy, im0, line_thickness=3)
                    max_area = max(max_area, area)
                print(max_area)
            if max_area > 85000 and game_once == 0:
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


                while running:
                    counter = counter + 1
                    print("counter: ", counter)
                    if (counter == 1350):
                        running = False
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
                        send_data(angle)  # 전송 형식을 실제 환경에 맞게 수정
                        print("현재 진행 각도:", angle)
                        print("\n진행방향: ", dirct)

                    # pygame.display.flip()
                    clock.tick(60)

                print("신호등하자.")
                # ser.close()  # 시리얼 포트 닫기
                # pygame.quit()

            # Print time (inference)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            # send_data(t2 - t1)

            source1 = np.float32([[350, 400], [200, 670], [900, 400], [1180, 670]])
            destination1 = np.float32([[0, 0], [0, 780], [600, 0], [600, 780]])

            ll_white = ll_white.astype(np.uint8).reshape(720, 1280, 1)
            list = ll_seg_mask.astype(np.uint8).reshape(720, 1280, 1) * 255
            merge_list = cv2.bitwise_and(list, list)

            # list = np.concatenate((list*255,list*255,list*255),axis=2)
            minverse = cv2.getPerspectiveTransform(destination1, source1)
            transform_matrix = cv2.getPerspectiveTransform(source1, destination1)

            merge_list = cv2.warpPerspective(merge_list, transform_matrix, (600, 780))  ##조도에 따라서 sensetive 변하는 함수 구현하기
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            # Eroded = cv2.erode(merge_list, kernel, 10)
            # thresh = cv2.cvtColor(list, cv2.COLOR_BGR2GRAY)

            # 소벨 필터
            sobelx = cv2.Sobel(merge_list, cv2.CV_64F, 1, 0, ksize=5)

            kk1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            kk2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            sobelx = cv2.morphologyEx(sobelx, cv2.MORPH_OPEN, kk2)
            sobelx = cv2.dilate(sobelx, kk2)
            sobelx = cv2.dilate(sobelx, kk2)
            sobelx = cv2.dilate(sobelx, kk2)

            # 선 분포도 조사 histogram
            rightbase, hist = plothistogram(sobelx)
            # rightbase, hist = plothistogram(Eroded)
            # plt.plot(hist)
            # plt.show()

            # list1, mid, left_angle, right_angle, right_save = slide_window_search(merge_list, leftbase, rightbase)
            list1, right_cur, slope, right_saving = slide_window_search(sobelx, rightbase)
            # list1, right_cur, slope, right_saving = slide_window_search(Eroded, rightbase)
            mid_custom = np.mean(right_saving)
            # print("차선값:",mid_custom)
            if save_mid[0] is None:
                save_mid[0] = mid_custom  #초기에 전의 mid_custom값 저장
                save_mid[1] = mid_custom
            else:
                save_mid[1] = mid_custom  #현재값 save_mid[1]에 저장
            #print(mid_custom)
            # mid_custom이 갑자기 많이 바뀌면 우측 차선 -> 좌측 차선이라고 판단.
            # print(mid_custom)
            # print("1: %f" %save_mid[0])
            # print("2: %f" %save_mid[1])
            if abs(save_mid[1] - save_mid[0]) > 300:  # 이전 중앙값과 현재 중앙값 차이가 200보다 클 때 -> 인지하는 차선이 바뀜
                lane_changed = True
            else:
                lane_changed = False

            if lane_changed is True:
                # current_lane 값을 바꿔줌
                if current_lane == 'right':
                    current_lane = 'left'
                    mid_standard = -39  # -39 ~
                else:
                    current_lane = 'right'
                    mid_standard = 544
            # print("!!!!!!!!!!!!!!!!!!!!현재 차선:", current_lane)%

            save_mid[0] = save_mid[1]

            cv2.imshow('i', list1)
            cv2.imshow('origin', im0s)
            # cv2.imshow('ddd', sobelx)
            cv2.waitKey(1)

            pid = PID(0.5, 0.015, 0.05)  # 특정 계수를 가진 PID 컨트롤러 생성
            error_mid = (save_mid[1] - mid_standard)
            diff = pid.pid_control(error_mid)  # PID를 사용하여 중간값 오차의 제어 출력 계산

            if abs(cal_exactly(right_saving)) < 3:  # 직선구간
                angle = diff * 0.2 + cal_exactly(right_saving) * 0.2
                print("직선---------------")
            else:  # 곡선 구간 cal_exactly 가중 증가
                angle = diff * 0.07 + cal_exactly(right_saving) * 0.3
                # print("곡선~~~~~~~~~~~~~~~~~~~")
            #
            # print("angle : %f" % angle)
            global serial_
            if serial_:
                if (angle > 16):
                    angle = 16
                elif (angle < -16):
                    angle = -16
                else:
                    angle = angle

                if (angle < 1 and angle > -1):
                    angle = 0
                # print("Helllo!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                speed = 50
                # print(angle)
                send_data(angle)

    inf_time.update(t2 - t1, img.size(0))
    nms_time.update(t4 - t3, img.size(0))
    waste_time.update(tw2 - tw1, img.size(0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')    #

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


if __name__ == '__main__':
    opt = make_parser().parse_args()
    print(opt)

    with torch.no_grad():
        detect()
