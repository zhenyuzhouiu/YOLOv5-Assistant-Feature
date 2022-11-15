import os
import shutil

# src_det = "/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/yolov5-left/detection"
# dst_det = "/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/detection/"
#
# src_fea = "/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/yolov5-left/feature"
# dst_fea = "/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/feature/"

src_seg = "/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/mask-thumb/"
dst_seg = "/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/segmentation/"

# left
# cls_dict = {
#     "Little": "01",
#     "Ring": "02",
#     "Middle": "03",
#     "Index": "04"
# }

# right
# cls_dict = {
#     "Little": "10",
#     "Ring": "09",
#     "Middle": "08",
#     "Index": "07"
# }

# thumb
cls_dict = {
    "Left": "05",
    "Right": "06"
}

subject_name = os.listdir(src_seg)
for s in subject_name:
    cls_name = os.listdir(os.path.join(src_seg, s))
    for c in cls_name:
        cls_num = cls_dict[c]
        dst_seg_c = os.path.join(dst_seg, cls_num)
        if not os.path.exists(dst_seg_c):
            os.mkdir(dst_seg_c)
        dst_seg_s = os.path.join(dst_seg_c, s)
        if not os.path.exists(dst_seg_s):
            os.mkdir(dst_seg_s)
        src_seg_c = os.path.join(src_seg, s, c)
        image_name = os.listdir(src_seg_c)
        for i in image_name:
            shutil.copy(os.path.join(src_seg_c, i), os.path.join(dst_seg_s, i))






