import os
import cv2

src_dir = r"F:\finger_knuckle_2018\finger_knuckle_2018\FingerKnukcleDatabase\Finger-knuckle\left-yolov5s-crop-feature-detection\left-index-crop"
dst_dir = r"F:\finger_knuckle_2018\finger_knuckle_2018\FingerKnukcleDatabase\Finger-knuckle\left-yolov5s-crop-feature-detection\left-index-crop-184"

if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

# (width, height)
# dst_size = (66, 64)
# dst_size = (100, 96)
# dst_size = (133, 128)
dst_size = (208, 184)

subject_name = os.listdir(src_dir)
for s in subject_name:
    subject_path = os.path.join(src_dir, s)
    dst_subject_path = os.path.join(dst_dir, s)
    if not os.path.exists(dst_subject_path):
        os.mkdir(dst_subject_path)
    image_name = os.listdir(subject_path)
    sample_id = 1
    for i in image_name:
        image_path = os.path.join(subject_path, i)
        dst_image_path = os.path.join(dst_subject_path, str(sample_id)+'.jpg')
        image = cv2.imread(image_path)
        # image = cv2.rotate(image, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        image = image[8:-8, :, :]

        # ration = h/w; dst_size(w, h)
        ratio = dst_size[1] / dst_size[0]
        h, w, c = image.shape
        dest_w = h / ratio
        dest_h = w * ratio
        if dest_w > w:
            crop_h = int((h - dest_h) / 2)
            if crop_h == 0:
                crop_h = 1
            crop_image = image[crop_h - 1:crop_h + int(dest_h), :, :]
        elif dest_h > h:
            crop_w = int((w - dest_w) / 2)
            if crop_w == 0:
                crop_w = 1
            crop_image = image[:, crop_w - 1:crop_w + int(dest_w), :]
        else:
            crop_image = image

        resize_image = cv2.resize(crop_image, dsize=dst_size)
        cv2.imwrite(dst_image_path, resize_image)
        sample_id += 1

