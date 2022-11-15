import os
import shutil

bboxes_smaller_4 = "/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/yolov5-right/detection/bboxes_smaller_4.txt"
save_path = "/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/bboxes_smaller_4/"

if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.mkdir(save_path)

with open(bboxes_smaller_4, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        subject_name = line.split('/')[-2]
        image_name = line.split('/')[-1]

        subject_path = os.path.join(save_path, subject_name)
        if not os.path.exists(subject_path):
            os.mkdir(subject_path)
        shutil.copy(line, subject_path + '/'+ image_name)

