""" 
Divide large tiles to smaller tiles and save smaller ones separately
将大 tile 分割成小的并另外保存
 """
import cv2 as cv
import numpy as np
import os
import shutil as shu
from glob import glob
from tqdm import tqdm

print("-*" * 20)
target_shape = {"x": 256, "y": 256}
orig_shape = {"x": 2048, "y": 2048}

# dataset_name = "TILES_(%s, %s)" % (orig_shape['x'], orig_shape['y'])
dataset_name = "TILES_FULL_2048"
orig_path = "/wd_0/ji/%s/" % dataset_name
print(orig_path)
# target_path = orig_path[:-1].replace("(%s, %s)" % (orig_shape['x'], orig_shape['y']), "(%s, %s)" % (target_shape['x'], target_shape['y']))+"/"
target_path = "/wd_0/ji/%s/" % (
    dataset_name + "(%s, %s)" % (target_shape["x"], target_shape["y"])
)

# if not (os.path.exists(target_path)):
#     os.mkdir(target_path)
# Ignoring pattern


def ig_f(dir, files):
    return [
        f
        for f in files
        if (
            os.path.isfile(os.path.join(dir, f))
            and (not (".txt" in os.path.join(dir, f)))
        )
    ]


if not (os.path.exists(target_path)):
    shu.copytree(orig_path, target_path, ignore=ig_f)

case_list = [""]
# case_list = ["7015", "1052", "3768", "5256", "6747", "8107", "7885", "2502", "7930"]
for k, case in enumerate(case_list):
    if k == 0:
        phl, pil, pml = tuple(
            [
                sorted(glob(orig_path + imtype + "/*" + case + "*.png"))
                for imtype in ["HE", "IHC", "Mask"]
            ]
        )
        print(orig_path + "HE" + "/*" + case + "*.png")
    else:
        phl = phl + sorted(glob(orig_path + "HE/" + case + "*.png"))
        pil = pil + sorted(glob(orig_path + "IHC/" + case + "*.png"))
        pml = pml + sorted(glob(orig_path + "Mask/" + case + "*.png"))
# print(np.array(filepath_list))
# print(np.array(filepath_ihc_list))
# print
for path_h, path_i, path_m in tqdm(zip(phl, pil, pml), total=len(phl)):
    # print(path_h)
    # if (".tif" in path_h or ".png" in path_h):
    imh = cv.imread(path_h)
    imi = cv.imread(path_i)
    imm = cv.imread(path_m)
    # orig_shape = {'x': im_large.shape[0], 'y': im_large.shape[1]} // use true image size instead
    w_num = orig_shape["x"] // target_shape["x"]
    h_num = orig_shape["y"] // target_shape["y"]
    for i in range(w_num):
        for j in range(h_num):
            chh = imh[
                i * target_shape["x"] : (i + 1) * target_shape["x"],
                j * target_shape["y"] : (j + 1) * target_shape["y"],
                :,
            ]
            # print(chh)
            th_blank = 200
            margin_pp = 4
            if (
                chh[: target_shape["x"] // margin_pp, :, :].mean() > th_blank
                or chh[
                    int((margin_pp - 1) * target_shape["x"] // margin_pp) :, :, :
                ].mean()
                > th_blank
                or chh[:, : target_shape["x"] // margin_pp, :].mean() > th_blank
                or chh[
                    :, int((margin_pp - 1) * target_shape["x"] // margin_pp) :, :
                ].mean()
                > th_blank
            ):
                continue
            chi = imi[
                i * target_shape["x"] : (i + 1) * target_shape["x"],
                j * target_shape["y"] : (j + 1) * target_shape["y"],
                :,
            ]
            chm = imm[
                i * target_shape["x"] : (i + 1) * target_shape["x"],
                j * target_shape["y"] : (j + 1) * target_shape["y"],
                :,
            ]

            ph_new = path_h.replace(orig_path, target_path).replace(
                ".png", "(%d, %d)" % (i, j) + ".png"
            )
            # print(ph_new)
            pi_new = path_i.replace(orig_path, target_path).replace(
                ".png", "(%d, %d)" % (i, j) + ".png"
            )
            pm_new = path_m.replace(orig_path, target_path).replace(
                ".png", "(%d, %d)" % (i, j) + ".png"
            )

            cv.imwrite(ph_new, chh)
            cv.imwrite(pi_new, chi)
            cv.imwrite(pm_new, chm)