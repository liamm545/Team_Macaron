import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import math
import matplotlib as plt
import serial  #######################
from sympy import Derivative, symbols

sensitivity = 60
lower_white = np.array([0, 0, 255 - sensitivity])
upper_white = np.array([255, sensitivity, 255])

# SERIAL
# arduino_port = 'COM3'
# ser = fl.libARDUINO()
# comm = ser.init(arduino_port, 9600)
serial_ = 1

if serial_:  #######################
    serial_port = 'COM3'
    baud_rate = 9600
    ser = serial.Serial(serial_port, baud_rate)

right_save = [0] * 12  # 숫자 수정 필요

# Conclude setting / general reprocessing / plots / metrices / datasets
from utils.utils import \
    time_synchronized, select_device, increment_path, \
    scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model, \
    lane_line_mask, plot_one_box, show_seg_result, \
    AverageMeter, \
    LoadImages


# right_save = None  # 숫자 수정 필요
# right_save_lane = [0] * 10


# a_slope = [None,None]
# b_slope = [None,None] # 전역변수 선언, 초기값 -1로 설정

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
    dot_save = [0] * (nwindows-3)  # 차선의 값을 저장해둠.

    for w in range(nwindows-3):
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
        else:
            w_num = w
            if w_num > 1:  # 최소한 2개의 데이터가 들어가 있을 때
                right_current = int(dot_save[w_num - 1] + (dot_save[w_num - 1] - dot_save[w_num - 2]))

        # mid[w] = (right_current + left_current)/2
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
        # cv2.line(out_img, (right_current, win_y_high), (right_current, win_y_high), [0, 255, 255], 5)
        dot_save[w] = right_current

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
    ym_per_pix = 98.0 / 780  # centimeters per pixel in y dimension
    xm_per_pix = 90.0 / 600  # centimeters per pixel in x dimension
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
    out_img = cv2.putText(out_img, str__, (150, 40), cv2.FONT_HERSHEY_PLAIN, 2, [255, 0, 0], 1, cv2.LINE_AA)  # 반지름 표시

    line_1 = right_fit[0] * ploty * 2 + right_fit[1]
    # print(-np.mean(line_1))
    line_1 = -np.mean(line_1)  # 기울기
    # print(line_1)
    return out_img, right_curverad, line_1, dot_save


# dot_save가 차선을 저장하는 리스트.


# def non_sliding(binary_warped, left_fit, right_fit):
#     nonzero = binary_warped.nonzero()
#     nonzeroy = np.array(nonzero[0])
#     nonzerox = np.array(nonzero[1])
#     margin = 100
#
#     right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin))
#                        & (nonzerox < (
#                         right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))
#
#     # Extract right line pixel positions
#     rightx = nonzerox[right_lane_inds]
#     righty = nonzeroy[right_lane_inds]
#
#     # Fit a second order polynomial to each
#     try:
#         right_fit = np.polyfit(righty, rightx, 2)
#     except:
#         return right_line.current_fit, right_line.radius_of_curvature, None
#
#     else:
#         # Check difference in fit coefficients between last and new fits
#         right_line.diffs = right_line.current_fit - right_fit
#         if (right_line.diffs[0] > 0.001 or right_line.diffs[1] > 0.4 or right_line.diffs[2] > 150):
#             return right_line.current_fit, right_line.radius_of_curvature, None
#         # print(right_line.diffs)
#
#         # Stash away polynomials
#         right_line.current_fit = right_fit
#
#         # Define conversions in x and y from pixels space to meters
#         ym_per_pix = 30 / 720  # meters per pixel in y dimension
#         xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
#
#         # Fit new polynomials to x,y in world space
#         right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, deg=2)
#
#         # Generate x and y values for plotting
#         ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
#         right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
#
#         # Calculate radii of curvature in meters
#         y_eval = np.max(ploty)  # Where radius of curvature is measured
#         right_curverad = ((1 + (
#                     2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
#             2 * right_fit_cr[0])
#
#         # Stash away the curvatures
#         right_line.radius_of_curvature = right_curverad
#
#         return right_fit, right_curverad, None
#

def calculate_angle(x1, y1, x2, y2):
    delta_x = x2 - x1
    delta_y = y2 - y1

    radian = np.arctan2(delta_y, delta_x)
    degree = np.degrees(radian)

    # if degree < 0:  # Convert the range from [-180, 180] to [0, 360] for practical usage
    #     degree += 360

    # if degree >=90:
    #     degree = -(degree-90)
    return degree


# def slide_window_search(binary_warped, left_current, right_current):
#     out_img = np.dstack((binary_warped, binary_warped, binary_warped))
#
#     nwindows = 15
#     window_height = int(binary_warped.shape[0] / nwindows)
#     nonzero = binary_warped.nonzero()
#     nonzero_y = np.array(nonzero[0])
#     nonzero_x = np.array(nonzero[1])
#     margin = 50
#     minpix = 50
#     left_lane = []
#     right_lane = []
#     color1 = [0, 255, 0]  # Green color
#     color2 = [0, 255, 255]  # Yellow color
#     color3 = [0, 0, 255]  # Red color
#     thickness = 2
#
#     # global a_slope
#     # global b_slope
#
#     a_save = [False] * nwindows  # 예외 처리 위해 비교할 리스트
#     left_save = [0] * nwindows
#     b_save = [False] * nwindows  # 예외 처리 위해 비교할 리스트2
#     right_save = [0] * nwindows
#     mid_save = [0] * nwindows  # 조향을 위한 중간값 리스트
#
#     a_angle = [0] * (nwindows)
#     b_angle = [0] * (nwindows)
#
#     # 첫 번째 for문 -> window 개수만큼의 값 저장함.
#     for w in range(nwindows):
#         win_y_low = binary_warped.shape[0] - (w + 1) * window_height
#         win_y_high = binary_warped.shape[0] - w * window_height
#         win_xleft_low = left_current - margin
#         win_xleft_high = left_current + margin
#         win_xright_low = right_current - margin
#         win_xright_high = right_current + margin
#         good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (
#                     nonzero_x < win_xleft_high)).nonzero()[0]
#         good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (
#                     nonzero_x < win_xright_high)).nonzero()[0]
#
#         left_lane.append(good_left)
#         right_lane.append(good_right)
#
#         # 차선 우선 보정
#         if len(good_left) > minpix:
#             left_current = int(np.mean(nonzero_x[good_left]))
#             a_save[w] = True
#         left_save[w] = left_current
#         if len(good_right) > minpix:
#             right_current = int(np.mean(nonzero_x[good_right]))
#             b_save[w] = True
#         right_save[w] = right_current
#
#     # print(left_save)
#     # print(left_save[0])
#     # print(left_save[nwindows//2])
#
#     # 차선이 하나만 보이는데 그 하나가 신뢰하지 못하는 차선일 수 있다.
#
#     for i in range(nwindows):  # 차선이 안 보이는 경우 재보정하는 for문
#         if a_save[i] == False:  # 우선 왼쪽 차선이 안보이는 경우
#             if b_save[i] == True:  # 왼쪽 차선만 안보이는 경우
#                 left_save[i] = right_save[i] - 1030
#                 mid_save[i] = (left_save[i] + right_save[i]) / 2
#
#             else:  # 두 차선 다 안보이는 경우
#                 if i > 1:  # 이전 값을 가지고 추정
#                     # if b_angle[i-1] == 0:
#                     #     b_angle[i-1] = 0.001
#                     # if a_angle[i-1] == 0:
#                     #     a_angle[i-1] = 0.001
#                     # print(b_angle)
#                     # print(i)
#                     right_save[i] = int(right_save[i - 1] + window_height / math.tan(math.radians(b_angle[i - 1])))
#                     left_save[i] = int(left_save[i - 1] + window_height / math.tan(math.radians(a_angle[i - 1])))
#                     mid_save[i] = int(left_save[i] + right_save[i]) / 2
#                 else:  # 첫 인덱스의 경우 그냥 직진.
#                     right_save[i] = 640 + 500
#                     left_save[i] = 640 - 500
#                     mid_save[i] = 640  # 두 차선 다 안보이는 경우에는 안전하게 직진한다.
#
#
#         elif b_save[i] == False:  # 오른쪽 차선만 안보이는 경우
#             right_save[i] = left_save[i] + 1030
#             mid_save[i] = (left_save[i] + right_save[i]) / 2
#
#         else:  # 차선이 둘다 보이는 경우
#             mid_save[i] = (left_save[i] + right_save[i]) / 2
#         if i != 0:
#             left_angle = calculate_angle(left_save[0], 0, left_save[i], window_height * i)
#             right_angle = calculate_angle(right_save[0], 0, right_save[i], window_height * i)
#             a_angle[i] = left_angle
#             b_angle[i] = right_angle
#
#         # print(b_angle)
#     # print(np.mean(a_angle))
#     # print(np.mean(b_angle))
#     # left_en = 0
#     # right_en = 0
#     for k in range(nwindows):  # 차선이 둘다 보이는 경우에도 보정하는 for문, 보이는 경우도 올바른 차선이 아닐 수도 있어서(햇빛, 주름)
#         if k > 1:
#             left_mean_angle = calculate_angle(left_save[k], 0, np.median(left_save), 360)
#             right_mean_angle = calculate_angle(right_save[k], 0, np.median(right_save), 360)
#
#             if a_save[k] == True and b_save[k] == True:  # 두 차선이 다 보이는 경우
#                 left_angle = calculate_angle(left_save[0], 0, left_save[k], window_height * k)
#                 right_angle = calculate_angle(right_save[0], 0, right_save[k], window_height * k)
#                 if abs(left_angle - left_mean_angle) > 10:
#                     left_save[k] = int(left_save[k - 1] + window_height / math.tan(math.radians(a_angle[k - 1])))
#                     # left_en = 1
#                 if abs(right_angle - right_mean_angle) > 10:
#                     right_save[k] = int(right_save[k - 1] + window_height / math.tan(math.radians(b_angle[k - 1])))
#                     # right_en = 1
#                 mid_save[k] = int(left_save[k] + right_save[k]) / 2
#
#     # print(a_slope)
#     # print(b_slope)
#     # print(left_save)
#     for j in range(nwindows):  # sliding window 그리는 for문
#         if a_save[j] == False:  # 우선 왼쪽 차선이 안보이는 경우
#             if b_save[j] == True:  # 왼쪽 차선만 안보이는 경우
#                 # if left_save[j] < 0:
#                 #     left_save[j] = 0
#                 cv2.rectangle(out_img, (left_save[j] - margin, binary_warped.shape[0] - (j + 1) * window_height),
#                               (left_save[j] + margin, binary_warped.shape[0] - j * window_height), color2, thickness)
#                 cv2.rectangle(out_img, (right_save[j] - margin, binary_warped.shape[0] - (j + 1) * window_height),
#                               (right_save[j] + margin, binary_warped.shape[0] - j * window_height), color1, thickness)
#             else:  # 두 차선 다 안보이는 경우
#                 cv2.rectangle(out_img, (left_save[j] - margin, binary_warped.shape[0] - (j + 1) * window_height),
#                               (left_save[j] + margin, binary_warped.shape[0] - j * window_height), color3, thickness)
#                 cv2.rectangle(out_img, (right_save[j] - margin, binary_warped.shape[0] - (j + 1) * window_height),
#                               (right_save[j] + margin, binary_warped.shape[0] - j * window_height), color3, thickness)
#         elif b_save[j] == False:  # 오른쪽 차선만 안보이는 경우
#             # if right_save[j] > 1280:
#             #     right_save[j] = 1280
#             cv2.rectangle(out_img, (left_save[j] - margin, binary_warped.shape[0] - (j + 1) * window_height),
#                           (left_save[j] + margin, binary_warped.shape[0] - j * window_height), color1, thickness)
#             cv2.rectangle(out_img, (right_save[j] - margin, binary_warped.shape[0] - (j + 1) * window_height),
#                           (right_save[j] + margin, binary_warped.shape[0] - j * window_height), color2, thickness)
#         else:  # 차선이 둘다 보이는 경우
#             cv2.rectangle(out_img, (left_save[j] - margin, binary_warped.shape[0] - (j + 1) * window_height),
#                           (left_save[j] + margin, binary_warped.shape[0] - j * window_height), color1, thickness)
#             cv2.rectangle(out_img, (right_save[j] - margin, binary_warped.shape[0] - (j + 1) * window_height),
#                           (right_save[j] + margin, binary_warped.shape[0] - j * window_height), color1, thickness)
#
#         # 오른쪽 차선이 없으면 기본적으로 640
#         # 왼쪽 차선이 없으면 기본적으로 90언저리
#         # 1. mean과 median의 차이 -> 성능의 차이가 있는지?
#         # 2. if문 안들어가면 갱신이 안되니까 저번 값 가지게 되고, 처음에 if문 안들어가면 약간 더미값 들어가는 것 같아서 이걸 처리
#         # 3. else해가지고 여기서 기울기 써서 처리
#         # 4. 처음부터 if 안들어가면 그 반대 차선 가져와서 처리, 만약 둘다 없으면
#         # 즉 우선순위 첫 번째, 기울기 써서 처리
#         # 두 번째, 반대 차선 값 사용해서 처리
#         ##### 그냥 640에서 적절히 해서 정확히 직진 하게?
#
#         # 안보이는 경우에 잠깐 튀는 경우가 조금 있어서 -> 기울기를 추가하면 더 좋을듯?
#         # 맨 밑의 값과 중간의 인덱스와 기울기를 구해놓고, 리스트에 저장해놈
#         # 그리고 맨 밑의 값과 인덱스의 값의 기울기를 구해서 threshold 이상이면 그 직선 위에 있는 값으로 값 수정.
#
#     mid_save = np.mean(mid_save)
#
#     # 여기서 이전 mid_save값 저장해둬서 해도 될듯 -> 만약 제어가 안되면
#
#     return out_img, mid_save, a_angle, b_angle, right_save

# def slide_window_search_(image):
#     """
#         histogram을 통해 파악된 차선의 위치에 sliding window search 적용하는 함수
#
#         Return (ret)
#         1) left_fitx : 왼쪽 차선에 해당하는 픽셀 x좌표들
#         2) right_fitx : 오른쪽 차선에 해당하는 픽셀 x좌표들
#         3) ploty : 차선에 해당하는 픽셀 y좌표들
#     """
#     out_img = np.dstack((image, image, image))
#
#     nwindows = 12
#     window_height = np.int_(image.shape[0] / nwindows)
#     nonzero = image.nonzero()  # 선이 있는 부분의 인덱스만 저장
#     nonzero_y = np.array(nonzero[0])  # 선이 있는 부분 y의 인덱스 값
#     nonzero_x = np.array(nonzero[1])  # 선이 있는 부분 x의 인덱스 값
#     margin = 50
#     margin_1 = 50
#     minpix = 50
#     left_lane = []
#     right_lane = []
#     color1 = [0, 255, 0]  # Green color
#     color2 = [0, 255, 255]  # Yellow color
#     color3 = [0, 0, 255]  # Red color
#     thickness = 2
#
#     b_save = [False] * nwindows  # 예외 처리 위해 비교할 리스트2
#     global right_save
#     mid_save = [0] * nwindows  # 조향을 위한 중간값 리스트
#
#     b_angle = [0] * (nwindows)
#
#     histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
#     midpoint = np.int_(histogram.shape[0] / 2)
#     # leftbase = np.argmax(histogram[:midpoint])
#     rightbase = np.argmax(histogram[np.mean(right_save) - margin_1: np.mean(right_save) + margin_1]) + midpoint
#
#     for w in range(nwindows):
#         win_y_low = image.shape[0] - (w + 1) * window_height  # window 윗부분
#         win_y_high = image.shape[0] - w * window_height  # window 아랫 부분
#
#         win_xright_low = rightbase - margin  # 오른쪽 window 왼쪽 위
#         win_xright_high = rightbase + margin  # 오른쪽 window 오른쪽 아래
#
#         good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (
#                 nonzero_x < win_xright_high)).nonzero()[0]
#
#         right_lane.append(good_right)
#
#         if len(good_right) > minpix:
#             right_current = int(np.mean(nonzero_x[good_right]))
#             b_save[w] = True
#         right_save[w] = right_current
#
#     # left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침
#     right_lane = np.concatenate(right_lane)
#     print(right_lane)
#     print(right_save)

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

    lane_changed = False
    current_lane = 'right'
    lane_type = "straight"
    mid_standard = 544

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
        # print(ll_seg_mask)
        # Process detections
        for i, det in enumerate(pred):  # detections per image

            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = 1  # to Path
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # if len(det):
            if 0:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img:  # Add bbox to image
                        plot_one_box(xyxy, im0, line_thickness=3)

            # Print time (inference)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

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
            # print(mid_custom)
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
                    mid_standard = -39  #-39 ~
                else:
                    current_lane = 'right'
                    mid_standard = 544
            print("!!!!!!!!!!!!!!!!!!!!현재 차선:", current_lane)

            save_mid[0] = save_mid[1]

            # print("hihi %d"%np.mean(right_saving))
            # result, newwarp, color_warp, pts_left_list, pts_right_list, mean_mid, mean_curve = draw_lane_lines(im0s, thresh, minverse, draw_info)
            #show_seg_result(im0s, (ll_seg_mask), is_demo=True) ##############################바꿈
            # cv2.circle(merge_list, (int(mid), 360), 20, (0, 255, 0), -1)
            # mid = np.mean(mid)
            # print(mid)
            # cv2.imshow('Image', merge_list)
            # print(ret)




            cv2.imshow('i', list1)
            # cv2.imshow('origin', im0s)
            # cv2.imshow('ddd', sobelx)
            cv2.waitKey(1)

            pid = PID(0.5, 0.003, 0.03)  # 특정 계수를 가진 PID 컨트롤러 생성
            #
            # divv = 0.5
            #
            # # R_H = (right_save[7]*divv - (right_save[1]*divv + right_save[-1]*divv) / 2) * math.sqrt(2) +  1
            # R_H = math.sqrt(right_save[7] * divv - (right_save[1] * divv + right_save[-1] * divv) / 2) + 1
            # R_W = math.sqrt((right_save[1]*divv  - right_save[-1]*divv ) ** 2 + (640*divv) ** 2)
            #
            # R_RAD = ((R_H / 2) + (R_W ** 2 / (8 * R_H)))
            #
            # if(right_save[0]  - right_save[-1] < 0):
            #     R_RAD = -R_RAD
            #
            # if right_cur < 0.006:  # 차선이 직선일 경우, slope와 중간값 차이 가중치 증가
            #     right_cur = 0
            #     angle = slope * 30
            #     # print(angle)
            #     angle = pid.pid_control(angle)
            #     # print("!!!!!!!!!!직선")
            # else:
            #     angle = right_cur * slope * 700
            #     # print(angle)
            #     angle = pid.pid_control(angle)
            #     # print("@@@@@@@@@@@@곡선")

            #
            #
            # left_angle = int(np.mean(left_angle)) - 90
            # right_angle = int(np.mean(right_angle)) - 90
            #
            # leftangle = pid.pid_control(left_angle)  # PID를 사용하여 곡선 오차의 제어 출력 계산
            # rightangle = pid.pid_control(right_angle)

            # print(mid)
            error_mid = (save_mid[1] - mid_standard)
            diff = pid.pid_control(error_mid)  # PID를 사용하여 중간값 오차의 제어 출력 계산

            # print(diff)
            # print(iff: ", diff, "left:", leftangle, "right:", rightangle, "rrad: ", R_RAD)
            # # angle = R_RAD / 6
            # angle = diff + angle
            # angle = cal_exactly(right_saving) * 8 + diff
            # #
            if abs(cal_exactly(right_saving)) < 3:  #직선구간
                angle = diff*0.2 + cal_exactly(right_saving) * 0.2
                print("직선---------------")
                lane_type = 'straight'
            else:  #곡선 구간 cal_exactly 가중 증가
                angle = diff*0.082 + cal_exactly(right_saving) * 0.4
                print("곡선~~~~~~~~~~~~~~~~~~~")
                lane_type = 'curve'
            # print("diff: ", diff)
            # print("cal: ", cal_exactly(right_saving))
            # print(cal_exactly(right_saving))
            # if angle >6 and angle <12 and current_lane == 'left':
            #     angle += 4
            if current_lane =='left' and lane_type =="curve":
                angle += 7


            # print("newan :", angle)
            #
            # # Here is for the serial
            # print("before angle: ", angle)
            # angle = angle*1.8 + diff
            print("angle : %f" % angle)
            global serial_
            if serial_:
                if (angle > 17):
                    angle = 17
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
                # if ser.in_waiting:
                #     val = ser.readline().decode()
                #     income = int(val)
                #     print("val :", income)

            # bytes를 전송
            # comm.write(bytes_data)

    inf_time.update(t2 - t1, img.size(0))
    nms_time.update(t4 - t3, img.size(0))
    waste_time.update(tw2 - tw1, img.size(0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')

    # 조향각과 속도 값을 보내는 예시
    # steering_angle = 90  # 조향각 값 (0~180 사이의 값)
    # speed = 50  # 속도 값


if __name__ == '__main__':
    opt = make_parser().parse_args()
    print(opt)

    with torch.no_grad():
        detect()
