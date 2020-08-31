""" Generates labels of Ki-67 positive regions 
options: 
1) Probablity map
2) Binary mask
 """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.axes3d import Axes3D #--- For 3D plot

import cv2 as cv
from cv2 import ximgproc

import os,glob
import argparse
import shutil as shu

from skimage.color import rgb2hed, hed2rgb,separate_stains, combine_stains
from skimage import data
from skimage.exposure import rescale_intensity

from pathlib import Path

def reject_outliers(data, m=3):
    dt = data.copy()
    dt[(abs(data - np.mean(data)) > m * np.std(data))] = np.mean(data)
    return dt


def in_range(d):
    return (0, np.max(cv.GaussianBlur(d.copy(), (3, 3), 0)))


def flip8(imname):
    imSrc = cv.imread(imname)
    for kr in range(4):
        for pr in range(2):
            if kr+pr == 0:
                continue
            im = imSrc
            for k in range(kr):
                im = cv.rotate(im, cv.ROTATE_90_CLOCKWISE)
            if pr != 0:
                im = cv.flip(im, flipCode=1)
            # plt.imshow(im)
            # plt.title("r%dp%d"%(kr*90, pr))
            # plt.show()
            cv.imwrite(imname[:-4] + "[r%dp%d].tif" % (kr*90, pr), im)
    return 0


def norm_by_row(M):
    """normalize the rows of a matrix

    Args:
        M ([ndarry]): [---]

    Returns:
        [type]: [description]
    """
    for k in range(M.shape[1]):
        M[k, :] /= np.sqrt(np.sum(np.power(M[k, :], 2)))
    return M


def showbychan(im_ihc):
    """show the stains' intensities of a deconvoluted image

    Args:
        im_ihc ([ndarry]): [I am tired]
    """
    for k in range(3):
        plt.figure()
        plt.imshow(im_ihc[:, :, k], cmap="gray")


def rgbdeconv(rgb, conv_matrix, C=0):
    """yield the H-R-D stain from RGB input

    Args:
        rgb ([numpy matrix]): input 3-channel RGB image tile
        conv_matrix ([ndarray]): [OD matrix 3x3]
        C (int, optional): constant in the log transform. Defaults to 0.

    Returns:
        [ndarray]: [H-R-D stain from RGB input]
    """
    rgb = rgb.copy().astype(float)
    rgb += C
    stains = np.reshape(-np.log10(rgb), (-1, 3)) @ conv_matrix
    return np.reshape(stains, rgb.shape)


def hecconv(stains, conv_matrix, C=0):
    #     from skimage.exposure import rescale_intensity
    stains = stains.astype(float)
    logrgb2 = -np.reshape(stains, (-1, 3)) @ conv_matrix
    rgb2 = np.power(10, logrgb2)
    return np.reshape(rgb2 - C, stains.shape)


def surf(matIn, name="fig", div=(50, 50), SIZE=(8, 6)):
    """custom surface plot function

    Args:
        matIn ([numpy matrix]): 2-D data to plot
        name (str, optional): [name of the plot]. Defaults to "fig".
        div (tuple, optional): [resolution of the plot grid]. Defaults to (50, 50).
        SIZE (tuple, optional): [size of the plot]. Defaults to (8, 6).
    """
    x = np.arange(0, matIn.shape[0])
    y = np.arange(0, matIn.shape[1])
    x, y = np.meshgrid(y, x)
    fig = plt.figure(figsize=SIZE)
    ax = Axes3D(fig)
    ax.plot_surface(x, y, matIn, rstride=div[0], cstride=div[1], cmap='jet')
    plt.title(name)
    plt.show()


def gfLabel(img_r, mode=-1, hard_thresh=0.5, geps=1000, grad=5,
            MOP_SIZE=3, MCL_SIZE=3, MORPH_ITER=2):
    """Creates binary label from input tile

    Args:
        img_r ([type]): [description]
        mode (int, optional): [description]. Defaults to -1.
        hard_thresh (float, optional): [description]. Defaults to 0.5.
        geps (int, optional): [description]. Defaults to 1000.
        grad (int, optional): [description]. Defaults to 5.
        MOP_SIZE (int, optional): [description]. Defaults to 3.
        MCL_SIZE (int, optional): [description]. Defaults to 3.
        MORPH_ITER (int, optional): [description]. Defaults to 2.
    """
    img_r = rescale_intensity(img_r, in_range=in_range(img_r))
    d1 = img_r
    d2 = (d1*255).astype(uint8)

    d2 = cv.morphologyEx(d2, op=cv.MORPH_OPEN, kernel=np.ones(
        (MOP_SIZE, MOP_SIZE), uint8), iterations=MORPH_ITER)
#     d2 = cv.morphologyEx(d2, op=cv.MORPH_CLOSE, kernel=np.ones((MCL_SIZE, MCL_SIZE), uint8), iterations=MORPH_ITER)

    guidedDAB = ximgproc.guidedFilter(
        guide=d2, src=d2, radius=grad, eps=geps, dDepth=-1)

    if mode == -1:
        d1 *= (guidedDAB < (hard_thresh*255))
        # _, _d1 = cv.threshold(guidedDAB,25,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        # d1 = (guidedDAB/255)*(_d1/255)
        d2 = (d1*255).astype(uint8)
    else:
        d1 *= (guidedDAB > (hard_thresh*255))
        d2 = (d1*255).astype(uint8)
    guide = d2

    guidedDAB = ximgproc.guidedFilter(
        guide=guide, src=d2, radius=grad, eps=geps, dDepth=-1)

    _, guidedDAB = cv.threshold((guidedDAB).astype(
        uint8), 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    guidedDAB = ((guidedDAB) > 0).astype(uint8)
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.imshow(guidedDAB, cmap="gray")
    plt.axis('off')

    return guidedDAB


def buildTree(phys_path, label_path="Prob"):
    """[summary]
    Copy dir struct of Tiles

    Tiles
        Blood
        Tumor
        ...

    ==>

    Probs
        Blood
        Tumor
        ...
    """
    def ig_f(dir, files):
        return [f for f in files if (os.path.isfile(os.path.join(dir, f)) and (not (".txt" in os.path.join(dir, f))))]

    if not os.path.exists(phys_path):
        shu.copytree(phys_path, label_path, ignore=ig_f)

    """ get a list of last subdiretories in tiles. e.g. subdirs = [blood, tumor...]
    and a string that gives the absolute path of corresponding origin directory, 
        e.g.    basedir="./Kimura/DATA/Tiles"
                target_dir = "./Kimura/DATA/Probs"
    
    for subsir in subdirs:
        # get a image list
        l_images = scandir(basedir + "/" + subsir)
        for image in l_images:
            imread(image)
            # process image
            .... some process
            # copy image to label directory
            ???.cp(image, target_dir + "/") # 如果要改文件名，在此前替换。如IHC_(1,2).tif ==> Mask_(1,2).tif
    """

    """ 
    TILES
    └── HE
        ├── 01_14-3768_Ki67_HE
        │   ├── Annotations
        │   │   └── annotations-01_14-3768_Ki67_HE.txt
        │   └── Tiles
        │       ├── Healthy Tissue [519 entries exceeds filelimit, not opening dir]
        │       └── Tumor [1662 entries exceeds filelimit, not opening dir]
        IHC
        └── 01_14-3768_Ki67_IHC
            ├── Annotations
            │   └── annotations-01_14-3768_Ki67_IHC.txt
            └── Tiles
                ├── Healthy Tissue [519 entries exceeds filelimit, not opening dir]
                └── Tumor [1662 entries exceeds filelimit, not opening dir]
    """
def parse_args():
    """parse input arguments
    """
    return


def main():
    """main entry of script.
    """
    return


if __name__ == "__main__":
    """waibiwabi
    waibibabo
    rugudmalaysia
    """
    main()
