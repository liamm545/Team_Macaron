import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import math
# import serial #######################

# SERIAL
# arduino_port = 'COM3'
# ser = fl.libARDUINO()
# comm = ser.init(arduino_port, 9600)
# if 1: #######################
#     serial_port = 'COM3'
#     baud_rate = 9600
#     ser = serial.Serial(serial_port, baud_rate)


# Conclude setting / general reprocessing / plots / metrices / datasets
from utils.utils import \
    time_synchronized, select_device, increment_path, \
    scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model, \
    lane_line_mask, plot_one_box, show_seg_result, \
    AverageMeter, \
    LoadImages


# a_slope = [None,None]
# b_slope = [None,None] # 전역변수 선언, 초기값 -1로 설정

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/example.mp4', help='source')  # file/folder, 0 for webcam
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


# def send_data(steering_angle): #######################
#     # steering_angle = str(steering_angle)
#     # data = "Q" + steering_angle
#     # ser.write(data.encode("utf-
#     data = str(steering_angle)
#     ser.write(data.encode())

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
    midpoint = int(histogram.shape[0] / 2)
    leftbase = np.argmax(histogram[:midpoint - 50])
    rightbase = np.argmax(histogram[midpoint + 50:]) + midpoint
    return leftbase, rightbase


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


def slide_window_search(binary_warped, left_current, right_current):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    nwindows = 12
    window_height = int(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    margin = 50
    minpix = 50
    left_lane = []
    right_lane = []
    color1 = [0, 255, 0]  # Green color
    color2 = [0, 255, 255]  # Yellow color
    color3 = [0, 0, 255]  # Red color
    thickness = 2

    # global a_slope
    # global b_slope

    a_save = [False] * nwindows  # 예외 처리 위해 비교할 리스트
    left_save = [0] * nwindows
    b_save = [False] * nwindows  # 예외 처리 위해 비교할 리스트2
    right_save = [0] * nwindows
    mid_save = [0] * nwindows  # 조향을 위한 중간값 리스트

    a_angle = [0] * (nwindows)
    b_angle = [0] * (nwindows)

    # 첫 번째 for문 -> window 개수만큼의 값 저장함.
    for w in range(nwindows):
        win_y_low = binary_warped.shape[0] - (w + 1) * window_height
        win_y_high = binary_warped.shape[0] - w * window_height
        win_xleft_low = left_current - margin
        win_xleft_high = left_current + margin
        win_xright_low = right_current - margin
        win_xright_high = right_current + margin
        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (
                    nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (
                    nonzero_x < win_xright_high)).nonzero()[0]

        left_lane.append(good_left)
        right_lane.append(good_right)

        # 차선 우선 보정
        if len(good_left) > minpix:
            left_current = int(np.mean(nonzero_x[good_left]))
            a_save[w] = True
        left_save[w] = left_current
        if len(good_right) > minpix:
            right_current = int(np.mean(nonzero_x[good_right]))
            b_save[w] = True
        right_save[w] = right_current

    # print(left_save)
    # print(left_save[0])
    # print(left_save[nwindows//2])

    # 차선이 하나만 보이는데 그 하나가 신뢰하지 못하는 차선일 수 있다.

    for i in range(nwindows):  # 차선이 안 보이는 경우 재보정하는 for문
        if a_save[i] == False:  # 우선 왼쪽 차선이 안보이는 경우
            if b_save[i] == True:  # 왼쪽 차선만 안보이는 경우
                left_save[i] = right_save[i] - 1030
                mid_save[i] = (left_save[i] + right_save[i]) / 2

            else:  # 두 차선 다 안보이는 경우
                if i > 1:  # 이전 값을 가지고 추정
                    # if b_angle[i-1] == 0:
                    #     b_angle[i-1] = 0.001
                    # if a_angle[i-1] == 0:
                    #     a_angle[i-1] = 0.001
                    # print(b_angle)
                    # print(i)
                    right_save[i] = int(right_save[i - 1] + window_height / math.tan(math.radians(b_angle[i - 1])))
                    left_save[i] = int(left_save[i - 1] + window_height / math.tan(math.radians(a_angle[i - 1])))
                    mid_save[i] = int(left_save[i] + right_save[i]) / 2
                else:  # 첫 인덱스의 경우 그냥 직진.
                    right_save[i] = 640 + 500
                    left_save[i] = 640 - 500
                    mid_save[i] = 640  # 두 차선 다 안보이는 경우에는 안전하게 직진한다.


        elif b_save[i] == False:  # 오른쪽 차선만 안보이는 경우
            right_save[i] = left_save[i] + 1030
            mid_save[i] = (left_save[i] + right_save[i]) / 2

        else:  # 차선이 둘다 보이는 경우
            mid_save[i] = (left_save[i] + right_save[i]) / 2
        if i != 0:
            left_angle = calculate_angle(left_save[0], 0, left_save[i], window_height * i)
            right_angle = calculate_angle(right_save[0], 0, right_save[i], window_height * i)
            a_angle[i] = left_angle
            b_angle[i] = right_angle

        # print(b_angle)
    # print(np.mean(a_angle))
    # print(np.mean(b_angle))
    # left_en = 0
    # right_en = 0
    for k in range(nwindows):  # 차선이 둘다 보이는 경우에도 보정하는 for문, 보이는 경우도 올바른 차선이 아닐 수도 있어서(햇빛, 주름)
        if k > 1:
            left_mean_angle = calculate_angle(left_save[k], 0, np.median(left_save), 360)
            right_mean_angle = calculate_angle(right_save[k], 0, np.median(right_save), 360)

            if a_save[k] == True and b_save[k] == True:  # 두 차선이 다 보이는 경우
                left_angle = calculate_angle(left_save[0], 0, left_save[k], window_height * k)
                right_angle = calculate_angle(right_save[0], 0, right_save[k], window_height * k)
                if abs(left_angle - left_mean_angle) > 10:
                    left_save[k] = int(left_save[k - 1] + window_height / math.tan(math.radians(a_angle[k - 1])))
                    # left_en = 1
                if abs(right_angle - right_mean_angle) > 10:
                    right_save[k] = int(right_save[k - 1] + window_height / math.tan(math.radians(b_angle[k - 1])))
                    # right_en = 1
                mid_save[k] = int(left_save[k] + right_save[k]) / 2

    # print(a_slope)
    # print(b_slope)
    # print(left_save)
    for j in range(nwindows):  # sliding window 그리는 for문
        if a_save[j] == False:  # 우선 왼쪽 차선이 안보이는 경우
            if b_save[j] == True:  # 왼쪽 차선만 안보이는 경우        
                # if left_save[j] < 0:
                #     left_save[j] = 0
                cv2.rectangle(out_img, (left_save[j] - margin, binary_warped.shape[0] - (j + 1) * window_height),
                              (left_save[j] + margin, binary_warped.shape[0] - j * window_height), color2, thickness)
                cv2.rectangle(out_img, (right_save[j] - margin, binary_warped.shape[0] - (j + 1) * window_height),
                              (right_save[j] + margin, binary_warped.shape[0] - j * window_height), color1, thickness)
            else:  # 두 차선 다 안보이는 경우
                cv2.rectangle(out_img, (left_save[j] - margin, binary_warped.shape[0] - (j + 1) * window_height),
                              (left_save[j] + margin, binary_warped.shape[0] - j * window_height), color3, thickness)
                cv2.rectangle(out_img, (right_save[j] - margin, binary_warped.shape[0] - (j + 1) * window_height),
                              (right_save[j] + margin, binary_warped.shape[0] - j * window_height), color3, thickness)
        elif b_save[j] == False:  # 오른쪽 차선만 안보이는 경우
            # if right_save[j] > 1280:
            #     right_save[j] = 1280
            cv2.rectangle(out_img, (left_save[j] - margin, binary_warped.shape[0] - (j + 1) * window_height),
                          (left_save[j] + margin, binary_warped.shape[0] - j * window_height), color1, thickness)
            cv2.rectangle(out_img, (right_save[j] - margin, binary_warped.shape[0] - (j + 1) * window_height),
                          (right_save[j] + margin, binary_warped.shape[0] - j * window_height), color2, thickness)
        else:  # 차선이 둘다 보이는 경우
            cv2.rectangle(out_img, (left_save[j] - margin, binary_warped.shape[0] - (j + 1) * window_height),
                          (left_save[j] + margin, binary_warped.shape[0] - j * window_height), color1, thickness)
            cv2.rectangle(out_img, (right_save[j] - margin, binary_warped.shape[0] - (j + 1) * window_height),
                          (right_save[j] + margin, binary_warped.shape[0] - j * window_height), color1, thickness)

        # 오른쪽 차선이 없으면 기본적으로 640
        # 왼쪽 차선이 없으면 기본적으로 90언저리
        # 1. mean과 median의 차이 -> 성능의 차이가 있는지?
        # 2. if문 안들어가면 갱신이 안되니까 저번 값 가지게 되고, 처음에 if문 안들어가면 약간 더미값 들어가는 것 같아서 이걸 처리
        # 3. else해가지고 여기서 기울기 써서 처리
        # 4. 처음부터 if 안들어가면 그 반대 차선 가져와서 처리, 만약 둘다 없으면 
        # 즉 우선순위 첫 번째, 기울기 써서 처리
        # 두 번째, 반대 차선 값 사용해서 처리
        ##### 그냥 640에서 적절히 해서 정확히 직진 하게?

        # 안보이는 경우에 잠깐 튀는 경우가 조금 있어서 -> 기울기를 추가하면 더 좋을듯?
        # 맨 밑의 값과 중간의 인덱스와 기울기를 구해놓고, 리스트에 저장해놈
        # 그리고 맨 밑의 값과 인덱스의 값의 기울기를 구해서 threshold 이상이면 그 직선 위에 있는 값으로 값 수정.

    mid_save = np.mean(mid_save)

    # 여기서 이전 mid_save값 저장해둬서 해도 될듯 -> 만약 제어가 안되면

    return out_img, mid_save, a_angle, b_angle


def detect():
    # setting and directories
    source, weights, save_txt, imgsz = opt.source, opt.weights, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

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

        ll_seg_mask = lane_line_mask(ll)
        # print(ll_seg_mask)
        # Process detections
        for i, det in enumerate(pred):  # detections per image

            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            print(f'{s}Done. ({t2 - t1:.3f}s)')

            source1 = np.float32([[395, 460], [225, 680], [1045, 460], [1195, 680]])  # bev 수정 parameter
            destination1 = np.float32([[0, 0], [0, 720], [1280, 0], [1280, 720]])

            list = ll_seg_mask.astype(np.uint8).reshape(720, 1280, 1)
            list = np.concatenate((list * 255, list * 255, list * 255), axis=2)
            gray_list = cv2.cvtColor(list, cv2.COLOR_BGR2GRAY)  # to gray
            ret, thresh = cv2.threshold(gray_list, 170, 255, cv2.THRESH_BINARY)
            transform_matrix = cv2.getPerspectiveTransform(source1, destination1)
            new_list = cv2.warpPerspective(thresh, transform_matrix, (1280, 720))  # bev
            # gray_list = cv2.cvtColor(new_list, cv2.COLOR_BGR2GRAY) # to gray
            # ret, thresh = cv2.threshold(gray_list, 170, 255, cv2.THRESH_BINARY)
            leftbase, rightbase = plothistogram(new_list)  # find leftbase and rightbase
            # print(leftbase)
            # print(rightbase)

            list1, mid, left_angle, right_angle = slide_window_search(new_list, leftbase, rightbase)
            # print(mid)
            list2 = cv2.line(list1, (640, 0), (640, 720), (0, 0, 255), 2)
            show_seg_result(im0s, ll_seg_mask, is_demo=True)  ##############################바꿈
            cv2.imshow('Image', im0s)
            cv2.waitkey(1)
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            # w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            # h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            w, h = im0.shape[1], im0.shape[0]
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(list2)

###################################################################################################################
# pid = PID(0.5, 0.0005, 0.05)  # 특정 계수를 가진 PID 컨트롤러 생성
# left_angle = int(np.mean(left_angle))-90
# right_angle = int(np.mean(right_angle))-90

# leftangle = pid.pid_control(left_angle)  # PID를 사용하여 곡선 오차의 제어 출력 계산
# rightangle = pid.pid_control(right_angle)
# error_mid = (mid - 640)

# diff = pid.pid_control(error_mid)  # PID를 사용하여 중간값 오차의 제어 출력 계산

# print("diff: ", diff, "left:", leftangle, "right:", rightangle)
# angle = leftangle * 0.25 + rightangle * 0.25 + diff * 0.5
# # print(angle)
# # print("\n")
# # c = angle

# # Here is for the serial
# if 1:

#     if (angle > 16):
#         angle = 16
#     elif (angle < -16):
#         angle = -16
#     else:
#         angle = angle
#     speed = 50
#     # print(angle)
#     send_data(angle)

#     time.sleep(0.05)
#     # if ser.in_waiting:
#     #     val = ser.readline().decode()
#     #
#     #     angle = int(val)
#     #
#     # print("val :", angle)

# # # bytes를 전송
# # comm.write(bytes_data)

# inf_time.update(t2 - t1, img.size(0))
# nms_time.update(t4 - t3, img.size(0))
# waste_time.update(tw2 - tw1, img.size(0))
# # print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))
# # print(f'Done. ({time.time() - t0:.3f}s)')
# print(torch.cuda.is_available())
# print(torch.cuda.current_device())
# print(torch.cuda.device_count())
# 조향각과 속도 값을 보내는 예시
# steering_angle = 90  # 조향각 값 (0~180 사이의 값)
# speed = 50  # 속도 값


if __name__ == '__main__':
    opt = make_parser().parse_args()
    print(opt)

    with torch.no_grad():
        detect()
