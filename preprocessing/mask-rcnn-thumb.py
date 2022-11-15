import os
import shutil

import cv2

src_path = "/media/zhenyuzhou/Data/finger_knuckle_2018/1_knucklesegmentation/maskrcnn-benchmark/datasets/knuckle/knuckle_seg"
dst_path = "/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/mask-thumb/"

image_name = os.listdir(src_path)
left_num = 0
right_num = 0
for i in image_name:
    subject = i.split('-')[0]
    cls = i.split('-')[1]
    sample = i.split('-')[2]
    subject_path = os.path.join(dst_path, subject)
    if not os.path.exists(subject_path):
        os.mkdir(subject_path)
    if cls == "05":
        cls_path = os.path.join(subject_path, "Left")
        if not os.path.exists(cls_path):
            os.mkdir(cls_path)
        image = cv2.imread(os.path.join(src_path, i))
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(cls_path + '/' + i, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        left_num += 1
    elif cls == "06":
        cls_path = os.path.join(subject_path, "Right")
        if not os.path.exists(cls_path):
            os.mkdir(cls_path)
        image = cv2.imread(os.path.join(src_path, i))
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(cls_path + '/' + i, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        right_num += 1

print(left_num)
print(right_num)
