# ========== Post-processing for detected knuckle regions (bounding boxes)
# there are four finger knuckle in a slap finger knuckle image,
# we should use the post-processing algorithm to select the little, ring, middle, and index finger knuckle
# based on their location on the slap finger knuckle images
# ==========
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


# get the third element
def takeConfidence(elem):
    return elem[5]


def takeX(elem):
    return elem[0]


def takeY(elem):
    return elem[1]


def statistic(top_four, p1, p2, p3, p4, p5, image_h):
    if len(top_four) == 4:
        top_four.sort(key=takeY)
        ymin = top_four[0][1]
        ymax = top_four[-1][1]
        disy = ymax - ymin
        top_four.sort(key=takeX)
        avg_14 = (top_four[0][1] + top_four[-1][1]) / 2
        disy_2_14 = abs(top_four[1][1] - avg_14)
        disy_3_14 = abs(top_four[2][1] - avg_14)
        disy_14 = abs(top_four[0][1] - top_four[-1][1])
        disy_23 = abs(top_four[1][1] - top_four[2][1])
        disyl = abs(top_four[1][1] - top_four[0][1])
        disxl = abs(top_four[1][0] - top_four[0][0])
        disyr = abs(top_four[3][1] - top_four[2][1])
        disxr = abs(top_four[3][0] - top_four[2][0])
        # constraints
        if disy_2_14 > p1 * disy_3_14 and disy_23 > p1 * disy_14 and disy > p2:
            top_four.pop(1)
        elif disy_3_14 > p1 * disy_2_14 and disy_23 > p1 * disy_14 and disy > p2:
            top_four.pop(2)
        else:
            if disyl > p3 and disxl < p4:
                top_four.pop(0)
            elif disyr > p3 and disxr < p4:
                top_four.pop(-1)
            else:
                if top_four[0][1] * p5 < top_four[1][1]:
                    top_four.pop(0)
                elif top_four[-1][1] * p5 < top_four[2][1]:
                    top_four.pop(-1)
                else:
                    # directly output four bounding boxes sorted by x
                    # convert list numpy to torch.tensor
                    # top_four = torch.from_numpy(np.array(top_four)).to(device)
                    return top_four
    else:
        top_four.sort(key=takeY)
        ymin = top_four[0][1]
        ymax = top_four[-1][1]
        disy = ymax - ymin
        top_four.sort(key=takeX)
        y1 = top_four[0][1]
        y2 = top_four[1][1]

        # constraints
        if y1 < image_h / 2 and disy > 100:
            top_four.pop(0)
        elif y2 < image_h and disy > 100:
            top_four.pop(1)
        else:
            return top_four

    return top_four


def post_processing(bboxes, image_w, image_h, p1=1.2, p2=0.23, p3=0.25, p4=0.05, p5=1.02):
    """
    processing little, ring, middle, and index finger knuckle
    bboxes:-> tensor
    bboxes.shape():-> [num_bounding, (x, y, w, h, angle, confidence, class)]
    ratio:-> p1, p5
    distance:-> p2, p3, p4
    """
    p2 = p2 * image_h
    p3 = p3 * image_h
    p4 = p4 * image_w

    # convert tensor to numpy
    device = bboxes.device
    bboxes = bboxes.cpu().numpy()
    bboxes = bboxes.tolist()
    # sort bounding boxes by confidence score

    bboxes.sort(key=takeConfidence)

    if len(bboxes) <= 4:
        bboxes.sort(key=takeX)
        return torch.from_numpy(np.array(bboxes)).to(device)
    else:
        # extract the top four bounding boxes
        top_four = []
        for i in range(4):
            top_four.append(bboxes.pop(-1))

        top_four = statistic(top_four, p1, p2, p3, p4, p5)
        if len(top_four) == 4:
            top_four.sort(key=takeX)
            return torch.from_numpy(np.array(top_four)).to(device)

        while len(bboxes) != 0:
            top_four.append(bboxes.pop(-1))
            statistic(top_four, p1, p2, p3, p4, p5)
            if len(top_four) == 4:
                top_four.sort(key=takeX)
                return torch.from_numpy(np.array(top_four)).to(device)
    top_four.sort(key=takeX)
    return torch.from_numpy(np.array(top_four)).to(device)


def thumb_processing(bboxes, image_w, image_h, p1=1.2, p2=0.23, p3=0.25, p4=0.05, p5=1.02):
    """
    processing thumb
    bboxes:-> tensor
    bboxes.shape():-> [num_bounding, (x, y, w, h, angle, confidence, class)]
    ratio:-> p1, p5
    distance:-> p2, p3, p4
    """
    p2 = p2 * image_h
    p3 = p3 * image_h
    p4 = p4 * image_w

    # convert tensor to numpy
    device = bboxes.device
    bboxes = bboxes.cpu().numpy()
    bboxes = bboxes.tolist()
    # sort bounding boxes by confidence score

    bboxes.sort(key=takeConfidence)

    if len(bboxes) <= 1:
        bboxes.sort(key=takeX)
        return torch.from_numpy(np.array(bboxes)).to(device)
    else:
        # extract the top four bounding boxes
        top_two = []
        for i in range(2):
            top_two.append(bboxes.pop(-1))

        top_two = statistic(top_two, p1, p2, p3, p4, p5, image_h)
        if len(top_two) == 2:
            top_two.sort(key=takeX)
            return torch.from_numpy(np.array(top_two)).to(device)

        while len(bboxes) != 0:
            top_two.append(bboxes.pop(-1))
            statistic(top_two, p1, p2, p3, p4, p5, image_h)
            if len(top_two) == 2:
                top_two.sort(key=takeX)
                return torch.from_numpy(np.array(top_two)).to(device)
    top_two.sort(key=takeX)
    return torch.from_numpy(np.array(top_two)).to(device)


def normalization(image_path):
    # log equalization
    # image = cv2.imread(image_path)
    # c = 255 / np.log(1 + np.max(image))
    # log_image = c * (np.log(image + 1))
    # log_image = np.array(log_image, dtype=np.uint8)
    # plt.subplot(1, 2, 1)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.imshow(image)
    # plt.subplot(1, 2, 2)
    # log_image = cv2.cvtColor(log_image, cv2.COLOR_BGR2RGB)
    # plt.imshow(log_image)
    # plt.show()
    # ========== histogram equalization
    # image = cv2.imread(image_path)
    # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(image_hsv)
    # eq_v = cv2.equalizeHist(v)
    # image_hsv = cv2.merge([h, s, eq_v])
    # img_output = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    # plt.subplot(1, 2, 1)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.imshow(image)
    # plt.subplot(1, 2, 2)
    # img_output = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)
    # plt.imshow(img_output)
    # plt.show()
    # ========== gamma transform
    gamma = 3
    image = cv2.imread(image_path)
    invgamma = 1 / gamma
    img_gamma = np.array(np.power((image / 255), invgamma) * 255, dtype=np.uint8)
    plt.subplot(1, 2, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    img_gamma = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2RGB)
    plt.imshow(img_gamma)
    plt.show()


if __name__ == "__main__":
    with open(
            "/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/yolov5-right/detection"
            "/bboxes_smaller_4.txt",
            'r') as f:
        lines = f.readlines()
        for line in lines:
            normalization(line.strip('\n'))
