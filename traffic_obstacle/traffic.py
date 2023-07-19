import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# 이미지파일을 컬러로 읽어온다.
cap = cv2.VideoCapture('/Users/jeong-inhyuk/Hyuk/vscode/Team_Macaron/video/traffic.mp4')
data = []

def color_filter(image):
    """
        HLS 필터 사용
        
        lower & upper : 흰색으로 판단할 minimum pixel 값
        white_mask : lower과 upper 사이의 값만 남긴 mask
        masked : cv2.bitwise_and() 함수를 통해 흰색인 부분 제외하고는 전부 검정색 처리
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # lower & upper : 흰색으로 판단할 minimum pixel 값, maximum pixel 값
    lower = np.array([200, 200, 200])
    upper = np.array([255, 255, 255])

    #masked : cv2.bitwise_and() 함수를 통해 흰색인 부분 제외하고는 전부 검정색 처리
    white_mask = cv2.inRange(rgb, lower, upper)

    # betwise를 통해 원본 이미지에 mask 적용
    mask = cv2.bitwise_and(image, image, mask = white_mask)

    return mask

def warpping(image):
    """
        차선을 BEV로 변환하는 함수
        
        Return
        1) _image : BEV result image
        2) minv : inverse matrix of BEV conversion matrix
    """
    
    source = np.float32([[350, 300], [0, 720], [930, 300], [1280, 720]])
    destination = np.float32([[0, 0], [0, 720], [1280, 0], [1280, 720]])

    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    minv = cv2.getPerspectiveTransform(destination, source)
    _image = cv2.warpPerspective(image, transform_matrix, (1280, 720))

    return _image, minv

def add_sample(new_sample):
    global data
    if len(data) < 2:
        data.append(new_sample)
    else:
        data = data[1:] + [new_sample]
    return data

# 이미지의 높이와 너비를 갖고온다.
while True:
    retval, img = cap.read()
    if not(retval):	# 프레임정보를 정상적으로 읽지 못하면
        break  # while문을 빠져나가기

    warpped_img, minverse = warpping(img)

    blurred_img = cv2.GaussianBlur(warpped_img, (0,0), 2)

    roi2 = blurred_img[600:720, 250:1150]

    w_f_img = color_filter(roi2)

    _gray = cv2.cvtColor(w_f_img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(_gray, 170, 255, cv2.THRESH_BINARY)

    canny_img = cv2.Canny(thresh, 60, 180)

    lines = cv2.HoughLines(canny_img, rho=1, theta = np.pi/180, threshold = 100)
        
        # Hough line detection 후 lines이 발견되면 thresh에 추출된 직선 그리기

    height,width = img.shape[:2]

    # hsv이미지로 변환한다.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # color range
    lower_red1 = np.array([0,100,100])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([160,100,100])
    upper_red2 = np.array([180,255,255])
    lower_green = np.array([40,50,50])
    upper_green = np.array([90,255,255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskg = cv2.inRange(hsv, lower_green, upper_green)

    maskr = cv2.add(mask1, mask2)

    img_result = cv2.bitwise_and(img, img, mask = maskr + maskg)
    img_result = cv2.erode(img_result, None, iterations=5)
    img_result = cv2.dilate(img_result, None, iterations=5)
    roi = img_result[0:150, 100:800]

    h, s, v = cv2.split(roi)
    total_sum = int(np.sum(h)/h.size)
    result = np.where(h != 0, 1, 0)
    total_one_sum = np.sum(result)
    data = add_sample(total_sum)
    a = 100
    if lines is not None:
        
        for line in lines:
            # Extract line parameters
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(thresh, (x1, y1), (x2, y2), (255, 0, 0), 5)

            # print(total_one_sum)
    # print(a)
   
    if (a * 180 / math.pi > -5 and a * 180 / math.pi < 5) and total_one_sum > 9000:  # stop
        if len(data) == 2:  # start
            if abs(data[0] - data[1]) > 30:
                
                print("go!!!")
            else:
                
                print("stop!!!")

    print("angle: ",a*180/math.pi, "total_one_sum: ", total_one_sum, "total_sum: ", data)        
    

    cv2.imshow('lane', thresh)
    cv2.imshow('detected results', roi)
    cv2.waitKey(1)
    # cv2.imshow('maskr', maskr)
    # cv2.imshow('maskg', maskg)
    # cv2.imshow('masky', masky)
