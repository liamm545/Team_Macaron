import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import  numpy

# Conclude setting / general reprocessing / plots / metrices / datasets
from utils.utils import \
    time_synchronized,select_device, increment_path,\
    scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result,\
    AverageMeter,\
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

def plothistogram(image):
    """
        histogram을 통해 2개의 차선 위치를 추출해주는 함수
        
        Return
        1) leftbase : left lane pixel coords
        2) rightbase : right lane pixel coords
    """
    histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftbase = np.argmax(histogram[:midpoint])
    rightbase = np.argmax(histogram[midpoint:]) + midpoint

    return leftbase, rightbase

def slide_window_search(binary_warped, left_current, right_current):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    nwindows = 24
    window_height = int(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    margin = 100
    minpix = 50
    left_lane = []
    right_lane = []
    color = (0, 255, 0)  # Green color as a tuple
    thickness = 2

    for w in range(nwindows):
        win_y_low = binary_warped.shape[0] - (w + 1) * window_height
        win_y_high = binary_warped.shape[0] - w * window_height
        win_xleft_low = left_current - margin
        win_xleft_high = left_current + margin
        win_xright_low = right_current - margin
        win_xright_high = right_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)

        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]

        left_lane.append(good_left)
        right_lane.append(good_right)

        if len(good_left) > minpix:
            left_current = int(np.mean(nonzero_x[good_left]))
        if len(good_right) > minpix:
            right_current = int(np.mean(nonzero_x[good_right]))

    return out_img

    # left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침
    # right_lane = np.concatenate(right_lane)

    # leftx = nonzero_x[left_lane]
    # lefty = nonzero_y[left_lane]
    # rightx = nonzero_x[right_lane]
    # righty = nonzero_y[right_lane]


    # left_fit = np.polyfit(lefty, leftx, 2)
    # right_fit = np.polyfit(righty, rightx, 2)

    # ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    # left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    # right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    
    # ltx = np.trunc(left_fitx)  # np.trunc() -> 소수점 부분을 버림
    # rtx = np.trunc(right_fitx)
    
    # out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
    # out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]

    # # plt.imshow(out_img)
    # # plt.plot(left_fitx, ploty, color = 'yellow')
    # # plt.plot(right_fitx, ploty, color = 'yellow')
    # # plt.xlim(0, 1280)
    # # plt.ylim(720, 0)
    # # plt.show()

    # ret = {'left_fitx' : ltx, 'right_fitx': rtx, 'ploty': ploty}

    # return ret


def detect():
    # setting and directories
    source, weights,  save_txt, imgsz = opt.source, opt.weights,  opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # Load model
    stride =32
    model  = torch.jit.load(weights)
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
        [pred,anchor_grid],seg,ll= model(img)
        t2 = time_synchronized()

        # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version 
        # but this problem will not appear in offical version 
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred,anchor_grid)
        tw2 = time_synchronized()

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)
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
            # if len(det):
            # if 0:
            #     # Rescale boxes from img_size to im0 size
            #     det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            #     # Print results
            #     for c in det[:, -1].unique():
            #         n = (det[:, -1] == c).sum()  # detections per class
            #         #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            #     # Write results
            #     for *xyxy, conf, cls in reversed(det):
            #         if save_txt:  # Write to file
            #             xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            #             line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
            #             with open(txt_path + '.txt', 'a') as f:
            #                 f.write(('%g ' * len(line)).rstrip() % line + '\n')

            #         if save_img :  # Add bbox to image
            #             plot_one_box(xyxy, im0, line_thickness=3)

            # Print time (inference)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            source1 = np.float32([[155, 460], [155, 680], [1125, 460], [1125, 680]])
            destination1 = np.float32([[0, 0], [0, 720], [1280, 0], [1280, 720]])

            list = ll_seg_mask.astype(numpy.uint8).reshape(720,1280,1)
            list = np.concatenate((list*255,list*255,list*255),axis=2)
            transform_matrix = cv2.getPerspectiveTransform(source1, destination1)
            new_list = cv2.warpPerspective(list, transform_matrix, (1280, 720))  # bev
            gray_list = cv2.cvtColor(new_list, cv2.COLOR_BGR2GRAY) # to gray
            ret, thresh = cv2.threshold(gray_list, 170, 255, cv2.THRESH_BINARY)
            leftbase, rightbase = plothistogram(thresh) # find leftbase and rightbase
            print(leftbase)
            print(rightbase)
            list1 = slide_window_search(thresh, leftbase, rightbase)
            # list2 = cv2.line(list1, (640, 0), (640, 720), (0, 0, 255), 2)
            show_seg_result(im0s, (ll_seg_mask,ll_seg_mask), is_demo=True) ##############################바꿈

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
                            #w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            #h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            w,h = im0.shape[1], im0.shape[0]
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(list1)

    inf_time.update(t2-t1,img.size(0))
    nms_time.update(t4-t3,img.size(0))
    waste_time.update(tw2-tw1,img.size(0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    opt =  make_parser().parse_args()
    print(opt)

    with torch.no_grad():
            detect()
