# ========== Post-processing for detected knuckle regions (bounding boxes)
# there are four finger knuckle in a slap finger knuckle image,
# we should use the post-processing algorithm to select the little, ring, middle, and index finger knuckle
# based on their location on the slap finger knuckle images
# ==========

import numpy as np
import torch


# get the third element
def takeConfidence(elem):
    return elem[5]


def takeX(elem):
    return elem[0]


def takeY(elem):
    return elem[1]


def statistic(top_four, p1, p2, p3, p4, p5):
    ymin = top_four.sort(key=takeY)[0][1]
    ymax = top_four.sort(key=takeY)[-1][1]
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

    return top_four


def post_processing(bboxes, image_w, image_h, p1=1.2, p2=0.23, p3=0.25, p4=0.05, p5=1.02):
    """
    bboxes:-> tensor
    bboxes.shape():-> [num_bounding, (x, y, w, h, angle, confidence, class)]
    ratio:-> p1, p5
    distance:-> p2, p3, p4
    """
    p2 = p2 * image_h
    p3 = p3 * image_h
    p4 = p4 * image_w

    # convert tensor to numpy
    device = bboxes.device()
    bboxes = bboxes.cpu().numpy()
    bboxes = list(bboxes).tolist()
    # sort bounding boxes by confidence score

    bboxes.sort(key=takeConfidence)

    if len(bboxes) <= 4:
        bboxes.sort(key=takeX)
        return bboxes
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
