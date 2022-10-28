import argparse
import os
import shutil
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
import math
import numpy as np

from pathlib import Path
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (check_img_size, scale_labels,
                           plot_one_rotated_box, strip_optimizer,
                           set_logging, rotate_non_max_suppression, longsideformat2cvminAreaRect)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.evaluation_utils import rbox2txt, rbox2center
from torch.nn import functional as F


def detect(save_img=False):
    '''
    input: save_img_flag
    output(result):
    '''
    # 获取输出文件夹，输入路径，权重，参数等参数
    out, source, crop, crop_feature, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.crop_path, opt.feature_path, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    # 获取设备
    device = select_device(opt.device)
    # 如果设备为gpu，使用Float16
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # 加载Float32模型，确保用户设定的输入图片分辨率能整除最大步长s=32(如不能则调整为能整除并返回)
    '''
    model = Model(
                  (model): Sequential(
                                       (0): Focus(...)
                                       (1): Conv(...)
                                            ...
                                       (24): Detect(...)
                    )
    '''
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # 设置Float16
    if half:
        model.half()  # to FP16

    # 移除之前的输出文件夹,并新建输出文件夹
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    subject_name = os.listdir(source)
    if not os.path.exists(crop):
        os.mkdir(crop)
    if not os.path.exists(crop_feature):
        os.mkdir(crop_feature)
    for s in subject_name:
        subject_path = os.path.join(source, s)
        crop_subject_path = os.path.join(crop, s)
        if not os.path.exists(crop_subject_path):
            os.mkdir(crop_subject_path)
        crop_subject_feature_path = os.path.join(crop_feature, s)
        if not os.path.exists(crop_subject_feature_path):
            os.mkdir(crop_subject_feature_path)
        out_subject = os.path.join(out, s)
        if not os.path.exists(out_subject):
            os.mkdir(out_subject)

        # Set Dataloader
        # 通过不同的输入源来设置不同的数据加载方式
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            save_img = True
            dataset = LoadImages(subject_path, img_size=imgsz)

        # Get names and colors
        # 获取类别名字    names = ['person', 'bicycle', 'car',...,'toothbrush']
        names = model.module.names if hasattr(model, 'module') else model.names
        # 设置画框的颜色    colors = [[178, 63, 143], [25, 184, 176], [238, 152, 129],....,[235, 137, 120]]随机设置RGB颜色
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        t0 = time.time()
        # 进行一次前向推理,测试程序是否正常  向量维度（1，3，imgsz，imgsz）
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

        """
            path 图片/视频路径  
            img 进行resize+pad之后的图片   1*3*re_size1*resize2的张量 (3,img_height,img_weight)
            im0s 原size图片   (img_height,img_weight,3)          
            vid_cap 当读取图片时为None，读取视频时为视频源   
        """
        for path, img, im0s, vid_cap in dataset:
            print(img.shape)
            img = torch.from_numpy(img).to(device)
            # 图片也设置为Float16
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            # 没有batch_size的话则在最前面添加一个轴
            if img.ndimension() == 3:
                # (in_channels,size1,size2) to (1,in_channels,img_height,img_weight)
                img = img.unsqueeze(0)  # 在[0]维增加一个维度

            # Inference
            t1 = time_synchronized()
            """
            model:
            input: in_tensor (batch_size, 3, img_height, img_weight)
            output: 推理时返回 [z,x]
            z tensor: [small+medium+large_inference]  size=(batch_size, 3 * (small_size1*small_size2 + medium_size1*medium_size2 + large_size1*large_size2), nc)
            x list: [small_forward, medium_forward, large_forward]  eg:small_forward.size=( batch_size, 3种scale框, size1, size2, [xywh,score,num_classes]) 
            '''
                   
            前向传播 返回pred[0]的shape是(1, num_boxes, nc)
            h,w为传入网络图片的长和宽，注意dataset在检测时使用了矩形推理，所以这里h不一定等于w
            num_boxes = 3 * h/32 * w/32 + 3 * h/16 * w/16 + 3 * h/8 * w/8
            pred[0][..., 0:4] 预测框坐标为xywh(中心点+宽长)格式
            pred[0][..., 4]为objectness置信度
            pred[0][..., 5:5+nc]为分类结果
            pred[0][..., 5+nc:]为Θ分类结果
            """
            # prediction : (batch_size, num_boxes, no)  [z, x] tuple
            # z: tensor: [small + medium + large_inference]
            # x: list:
            # save: list[None, None, ..., tensor]
            # the save will be used for skip connection on the model
            prediction, save = model(img, augment=opt.augment)
            # pred = [z, x][0] = z
            pred = prediction[0]

            # Apply NMS
            # 进行NMS
            # input pred : list[tensor(batch_size, num_conf_nms, [xylsθ,conf,classid])] θ∈[0,179]
            # output pred : list[num_conf_nums, 7] length of list is batch_size
            pred = rotate_non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                              agnostic=opt.agnostic_nms, without_iouthres=False)
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # i:image index  det:(num_nms_boxes, [xylsθ,conf,classid]) θ∈[0,179]
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out_subject) / Path(p).name)  # 图片保存路径+图片名字
                txt_path = str(Path(out_subject) / Path(p).stem) + (
                    '_%g' % dataset.frame if dataset.mode == 'video' else '')
                # print(txt_path)
                s += '%gx%g ' % img.shape[2:]  # print string

                if det is not None and len(det):
                    # ========================== Rescale boxes from img_size to im0 size
                    det[:, :5] = scale_labels(img.shape[2:], det[:, :5], im0.shape).round()

                    # Print results    det:(num_nms_boxes, [xylsθ,conf,classid]) θ∈[0,179]
                    for c in det[:, -1].unique():  # unique函数去除其中重复的元素，并按元素（类别）由大到小返回一个新的无元素重复的元组或者列表
                        n = (det[:, -1] == c).sum()  # detections per class  每个类别检测出来的素含量
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string 输出‘数量 类别,’

                    # Write results  det:(num_nms_boxes, [xywhθ,conf,classid]) θ∈[0,179]
                    # ==================== Firstly, we only consider the major finger knuckle
                    one_image_knuckle = 0
                    for *rbox, conf, cls in reversed(det):  # 翻转list的排列结果,改为类别由小到大的排列
                        if cls == 1:
                            continue
                        fk_fm_32, fk_fm_16, fk_fm_8 = assistant_feature(*rbox, save=save)
                        # =============================== for minor finger knuckle angle
                        if cls == 1:
                            pred_angle = int(rbox[4].cpu().float().numpy())
                            if pred_angle > 90:
                                pred_angle = pred_angle - 90
                            else:
                                pred_angle = pred_angle + 90
                            pred_angle = '%s' % int(pred_angle)
                        else:
                            pred_angle = '%s' % int(rbox[4].cpu().float().numpy())
                        # for finger knuckle 0: major knuckle; 1: minor knuckle
                        # for finger nail blood key point 0: bifurcation; 1: ending points
                        if cls == 0:
                            if save_img or view_img:  # Add bbox to image
                                label = '%s %.2f' % (names[int(cls)], conf)
                                angle = pred_angle
                                classname = '%s' % names[int(cls)]
                                conf_str = '%.3f' % conf
                                rbox2txt(rbox, classname, conf_str, Path(p).stem,
                                         str(out_subject + '/result_txt/result_before_merge'))
                                im0, img_crop = plot_one_rotated_box(rbox, im0, label=angle,
                                                                     color=colors[int(cls)],
                                                                     line_thickness=1,
                                                                     pi_format=False)
                                cv2.imwrite(crop_subject_path + '/major-' + str(one_image_knuckle) + '.jpg', img_crop)
                                np.save(crop_subject_feature_path + '/major-' + str(
                                    one_image_knuckle) + '-' + '32' + '.npy', fk_fm_32)
                                np.save(crop_subject_feature_path + '/major-' + str(
                                    one_image_knuckle) + '-' + '16' + '.npy', fk_fm_16)
                                np.save(crop_subject_feature_path + '/major-' + str(
                                    one_image_knuckle) + '-' + '8' + '.npy', fk_fm_8)
                                one_image_knuckle += 1

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results 播放结果
                if view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                        pass
                    else:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            print('   Results saved to %s' % Path(out_subject))

        print('For one subject all done. (%.3fs)' % (time.time() - t0))


def generate_theta(i_radian, i_tx, i_ty, i_batch_size, i_h, i_w, i_dtype):
    # if you want to keep ration when rotation a rectangle image
    theta = torch.tensor([[math.cos(i_radian), math.sin(-i_radian) * i_h / i_w, i_tx],
                          [math.sin(i_radian) * i_w / i_h, math.cos(i_radian), i_ty]],
                         dtype=i_dtype).unsqueeze(0).repeat(i_batch_size, 1, 1)
    # else
    # theta = torch.tensor([[math.cos(i_radian), math.sin(-i_radian), i_tx],
    #                       [math.sin(i_radian), math.cos(i_radian), i_ty]],
    #                      dtype=i_dtype).unsqueeze(0).repeat(i_batch_size, 1, 1)
    return theta


def assistant_feature(*obbox, save):
    # ========================= Using NMS OBB to locate feature maps
    # det:-> tensor [x, y, l, s, θ, confidence, class_id]
    # fm:-> tensor [batch_size, channels, input_h/stride, input_w/stride]
    # obb:-> list [obb_32, obb_16, obb_8]
    rect_32 = longsideformat2cvminAreaRect(obbox[0] / 32, obbox[1] / 32, obbox[2] / 32, obbox[3] / 32, (obbox[4] - 180))
    # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    poly_32 = np.float32(cv2.boxPoints(rect_32))  # from opencv rotated rectangle get four points
    poly_32 = np.int0(poly_32)
    rect_16 = longsideformat2cvminAreaRect(obbox[0] / 16, obbox[1] / 16, obbox[2] / 16, obbox[3] / 16, (obbox[4] - 180))
    poly_16 = np.float32(cv2.boxPoints(rect_16))
    poly_16 = np.int0(poly_16)
    rect_8 = longsideformat2cvminAreaRect(obbox[0] / 8, obbox[1] / 8, obbox[2] / 8, obbox[3] / 8, (obbox[4] - 180))
    poly_8 = np.float32(cv2.boxPoints(rect_8))
    poly_8 = np.int0(poly_8)
    rect = [rect_32, rect_16, rect_8]
    poly = [poly_32, poly_16, poly_8]
    # fm:-> list [fm_32, fm_16, fm_8]
    fm = [save[23], save[20], save[17]]
    fk_fm = []

    for i in range(len(poly)):
        poly_i = poly[i]
        fm_i = fm[i]
        rect_i = rect[i]
        # [b, c, h , w] to [h, w, c]
        fm_i = fm_i.squeeze(0).permute(1, 2, 0)
        fm_i = fm_i.cpu().numpy().astype(np.float64)
        if fm_i.shape[2] > 512:
            div = fm_i.shape[2] // 512
            mod = fm_i.shape[2] % 512
            for d in range(div):
                if d == 0:
                    fk_fm_d = rotated_crop(rect_i, poly_i, fm_i[..., 512 * d:512 * (d + 1)])
                else:
                    fk_fm_d = np.concatenate((fk_fm_d,
                                              rotated_crop(rect_i, poly_i, fm_i[..., 512 * d:512 * (d + 1)])),
                                             axis=2)
            if mod != 0:
                fk_fm_d = np.concatenate((fk_fm_d,
                                          rotated_crop(rect_i, poly_i, fm_i[..., 512 * div:512 * div + mod])),
                                         axis=2)
            fk_fm.append(fk_fm_d)
        else:
            fk_fm.append(rotated_crop(rect_i, poly_i, fm_i))

    return fk_fm[0], fk_fm[1], fk_fm[2]


def rotated_crop(rect, poly, fm):
    # cropping rotated rectangle from image with opencv
    rotated = 0
    if rotated == 90:
        if abs(rect[2]) == 90:
            src_pts = poly.astype("float32")
            width = int(rect[1][0])
            height = int(rect[1][1])
            if width < height:
                dst_pts = np.array([[0, height - 1],
                                    [0, 0],
                                    [width - 1, 0],
                                    [width - 1, height - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                fm = cv2.warpPerspective(fm, M, (1000, 1000))
            else:
                dst_pts = np.array([[height - 1, width - 1],
                                    [0, width - 1],
                                    [0, 0],
                                    [height - 1, 0]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                fm = cv2.warpPerspective(fm, M, (1000, 1000))
        else:
            src_pts = poly.astype("float32")
            width = int(rect[1][0])
            height = int(rect[1][1])
            if width > height:
                dst_pts = np.array([[height - 1, width - 1],
                                    [0, width - 1],
                                    [0, 0],
                                    [height - 1, 0]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                fm = cv2.warpPerspective(fm, M, (1000, 1000))
            else:
                dst_pts = np.array([[0, height - 1],
                                    [0, 0],
                                    [width - 1, 0],
                                    [width - 1, height - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                fm = cv2.warpPerspective(fm, M, (1000, 1000))
    else:
        if abs(rect[2]) == 90:
            src_pts = poly.astype("float32")
            width = int(rect[1][0])
            height = int(rect[1][1])
            if width < height:
                dst_pts = np.array([[height - 1, width - 1],
                                    [0, width - 1],
                                    [0, 0],
                                    [height - 1, 0]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                fm = cv2.warpPerspective(fm, M, (height, width))
            else:
                dst_pts = np.array([[width - 1, 0],
                                    [width - 1, height - 1],
                                    [0, height - 1],
                                    [0, 0]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                fm = cv2.warpPerspective(fm, M, (width, height))
        else:
            src_pts = poly.astype("float32")
            width = int(rect[1][0])
            height = int(rect[1][1])
            if width > height:
                dst_pts = np.array([[width - 1, 0],
                                    [width - 1, height - 1],
                                    [0, height - 1],
                                    [0, 0]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                fm = cv2.warpPerspective(fm, M, (width, height))
            else:
                dst_pts = np.array([[height - 1, width - 1],
                                    [0, width - 1],
                                    [0, 0],
                                    [height - 1, 0]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                fm = cv2.warpPerspective(fm, M, (height, width))

    return fm


if __name__ == '__main__':
    """
        weights:训练的权重
        source:测试数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头),也可以是rtsp等视频流
        output:网络预测之后的图片/视频的保存路径
        img-size:网络输入图片大小
        conf-thres:置信度阈值
        iou-thres:做nms的iou阈值
        device:设置设备
        view-img:是否展示预测之后的图片/视频，默认False
        save-txt:是否将预测的框坐标以txt文件形式保存，默认False
        classes:设置只保留某一部分类别，形如0或者0 2 3
        agnostic-nms:进行nms是否将所有类别框一视同仁，默认False
        augment:推理的时候进行多尺度，翻转等操作(TTA)推理
        update:如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='./weights/finger_knuckle_obb/rog-yolov5x-longside-cw.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/left/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/detection/', help='output folder')  # output folder
    parser.add_argument('--crop_path', type=str, default='/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/left-crop', help='crop finger knuckle folder')
    parser.add_argument('--feature_path', type=str, default='/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/left-feature', help='crop feature map folder')
    parser.add_argument('--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', default=False, help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                # 去除pt文件中的优化器等信息
                strip_optimizer(opt.weights)
        else:
            detect()
