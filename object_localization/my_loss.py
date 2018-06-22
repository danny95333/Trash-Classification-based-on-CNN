import torch

def overlapped_area(a, b):  # returns None if rectangles don't intersect
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if (dx>=0) and (dy>=0):
        return dx*dy
    
def IntersectionOfUnion(a, b):

    size_a = (a[2]-a[0])*(a[3]-a[1])
    size_b = (b[2]-b[0])*(b[3]-b[1])
    intersection = overlapped_area(a, b)
    if intersection == None:
        intersection = 0
    union = size_a + size_b - intersection
    return intersection / union

def L2_loss(output, groundtruth):
    loss_sum = 0
    for (a, b) in zip(output, groundtruth): 
        loss_sum += torch.norm(a - b, 2)
    return loss_sum/len(output)
    
# def loss_bbox_cal(output, groundtruth):
#     loss_sum = 0
#     for (a, b) in zip(output, groundtruth):
#         loss_sum += IntersectionOfUnion(a, b)

#     return loss_sum/len(output)