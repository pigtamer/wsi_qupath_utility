#%%%%%%%%%
import json, glob
import numpy as np
from numpy.core.defchararray import rfind
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D #--- For 3D plot
import cv2 as cv
# %%
def surf(matIn, name="fig", div = (10, 10), SIZE = (8, 6)):
    x = np.arange(0, matIn.shape[0])
    y = np.arange(0, matIn.shape[1])
    x, y = np.meshgrid(y, x)
    fig = plt.figure(figsize = SIZE)
    ax = Axes3D(fig)
    ax.plot_surface(x, y, matIn, rstride=div[0], cstride=div[1], cmap='jet')
    plt.title(name)
    plt.show()

# %%
def get_iou(boxA, boxB, mode="Delta"):
    if mode=="Delta":
        boxA[2] += boxA[0]
        boxA[3] += boxA[1]
        boxB[2] += boxB[0]
        boxB[3] += boxB[1]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# json polygon dets file
WORK_PATH = "/Users/cunyuan/DATA/Kimura/qupath-proj/dets-json/eval/"
FLAG_ONLY_TRUE = 0
for JSON_PATH in glob.glob(WORK_PATH + "*DIG*json"):
    id_strsecpoint = str.find(JSON_PATH, "Ki67")
    thisid = JSON_PATH[id_strsecpoint-5: id_strsecpoint-1]
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    with open(JSON_PATH, 'r') as f:
            dets_list = json.load(f)

    polyf = [None]*len(dets_list)
    lblf = [None]*len(dets_list)
    k=0
    for itf in dets_list:
        polyf[k] = dict(itf)[ 'nucleusGeometry']['coordinates'][0]
        lblf[k] = dict(itf)['properties']['classification']['name'] == 'Positive'
        k+=1
    polyf = np.array(polyf)
    f.close()

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    with open(JSON_PATH.replace('DIG', 'IHC'), 'r') as g:
            lbls_list = json.load(g)

    polyg = [None]*len(lbls_list)
    lblg = [None]*len(lbls_list)
    k=0
    for itg in lbls_list:
        polyg[k] = dict(itg)[ 'nucleusGeometry']['coordinates'][0]
        lblg[k] = dict(itg)['properties']['classification']['name'] == 'Positive'
        k+=1
    polyg = np.array(polyg)
    g.close()
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Now we have (polyf, lblf), (polyg, lblg)
    """
    (1)	Calculate the IoU of the bounding boxes and perform non-maximum suppression with a threshold, e.g., IoU > 0.75.
    (2)	Label assignment: assign each label bounding box with the prediction having greatest IoU
    (3)	Remove duplications: for each assigned prediction, only preserve the label with greatest IoU if it overlaps with more than one labels
    (4)	All other label bounding boxes that overlaps with each prediction are marked as false positives.
    """
    if FLAG_ONLY_TRUE==1:
        polyf = polyf[np.array(lblf) == True]
        polyg = polyg[np.array(lblg) == True]
    elif FLAG_ONLY_TRUE==2:
        polyf = polyf[np.array(lblf) == False]
        polyg = polyg[np.array(lblg) == False]
    # %%
    rf = [None]*len(polyf)
    rg = [None]*len(polyg)
    k = 0
    for pf in polyf:
        rf[k] = cv.boundingRect(np.array(pf).astype(np.float32))
        k+=1
    k = 0
    for pg in polyg:
        rg[k] = cv.boundingRect(np.array(pg).astype(np.float32))
        k+=1
    # %%
    # plt.figure(figsize=(10,10), dpi=300)
    # for item in rf:
    #     plt.scatter(item[0], item[1])

    #%%%%%%%%%%
    # set a hard thresh of iou for assigning TP
    HARD_THRESH = 0.5
    iou_table = np.zeros((len(rf), len(rg)))
    for k in range(len(rf)):
        # calc and sort iou, take max
        for j in range(len(rg)):
            iou_table[k,j] = get_iou(list(rf[k]),
                                    list(rg[j]))
    iou_table = np.array(iou_table)
    #%%%%%%%%
    iou_table *= np.double(iou_table> HARD_THRESH)
    # surf(iou_table)
    # #%%%%%%%%%%
    # plt.figure(figsize=(4,4), dpi=300)
    # plt.imshow(iou_table, cmap="gray")
    # plt.title(thisid)
    # plt.tight_layout();plt.axis('off')
    # plt.show()

    # surf(iou_table)
    # %%
    assign_lbl = np.zeros((iou_table.shape[0],))
    while iou_table.max() != 0:
        thisx, thisy = np.unravel_index(iou_table.argmax(), iou_table.shape)
        assign_lbl[thisx] = thisy
        iou_table[thisx, :] = 0
        iou_table[:, thisy] = 0
    # %%
    
    print(thisid, "\t", 
    (assign_lbl>0).sum()/len(assign_lbl), "\t",
    (assign_lbl>0).sum()/iou_table.shape[1], "\t", 
    2/((len(assign_lbl) + iou_table.shape[1])/(assign_lbl>0).sum()), "\t", 
    iou_table.shape[0], "\t", 
    iou_table.shape[1], "\t", 
    )
    
    # %%