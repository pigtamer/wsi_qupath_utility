""" 
Divide large tiles to smaller tiles and save smaller ones separately
将大 tile 分割成小的并另外保存
 """
import cv2 as cv
import os
import shutil as shu
from glob import glob
from tqdm import tqdm

print("-*"*20)
target_shape = {'x': 256, 'y': 256}
orig_shape = {'x': 2048, 'y': 2048}

# dataset_name = "TILES_(%s, %s)" % (orig_shape['x'], orig_shape['y'])
dataset_name = "TILES_PAD"
orig_path = "/wd_0/ji/%s/" % dataset_name
print(orig_path)
# target_path = orig_path[:-1].replace("(%s, %s)" % (orig_shape['x'], orig_shape['y']), "(%s, %s)" % (target_shape['x'], target_shape['y']))+"/"
target_path ="/wd_0/ji/%s/" % (dataset_name  + "(%s, %s)" % (target_shape['x'], target_shape['y']))
print(orig_path)

# if not (os.path.exists(target_path)):
#     os.mkdir(target_path)
# Ignoring pattern
print(orig_path)

def ig_f(dir, files):
    return [f for f in files if (os.path.isfile(os.path.join(dir, f)) and (not (".txt" in os.path.join(dir, f))))]

if not (os.path.exists(target_path)):
    shu.copytree(orig_path, target_path, ignore=ig_f)

# for imtype in [ "HE", "Mask", "IHC"]:
#     dir_path  = orig_path + imtype + "/"
#     print("Processing " + imtype + "\n")
#     for roots, dirs, files in os.walk(dir_path):
#         for file in tqdm(files):
#             filepath = os.path.join(roots, file)
#             # if not ("Tumor" in filepath): continue 
#             # print("2 "+orig_path + imtype + "/")
#             # print(roots)
#             if (".tif" in filepath or ".png" in filepath):
#                 im_large = cv.imread(filepath)
#                 # ! important: a lower version of 
#                 # orig_shape = {'x': im_large.shape[0], 'y': im_large.shape[1]} // use true image size instead
#                 w_num = orig_shape['x'] // target_shape['x']
#                 h_num = orig_shape['y'] // target_shape['y']
#                 for i in range(w_num):
#                     for j in range(h_num):
#                         chip = im_large[i * target_shape['x']:(i + 1) * target_shape['x'],
#                                         j * target_shape['y']:(j + 1) * target_shape['y'],
#                                         :]

#                         # new_path = roots.replace("(%s, %s)" % (orig_shape['x'], orig_shape['y']), "(%s, %s)" % (target_shape['x'], target_shape['y'])) +\
#                         #             "/" + os.path.splitext(file)[0] + "(%d, %d)" % (i, j) + ".png"
#                         new_path = target_path + imtype + "/" + os.path.splitext(file)[0] + "(%d, %d)" % (i, j) + ".png"
#                         # print(target_path + imtype + "/" + os.path.splitext(file)[0] + "(%d, %d)" % (i, j) + ".png")
#                         # print(new_path)
#                         # print(roots)
#                         if not (os.path.exists(new_path)):
#                             cv.imwrite(new_path, chip)

phl,pil,pml = tuple([sorted(glob(orig_path+imtype+"/G1*2189*.png")) for imtype in ["HE", "IHC", "Mask"]])
for path_h, path_i, path_m in tqdm(zip(phl, pil, pml), total=len(phl)):
    # print(path_h)
    # if (".tif" in path_h or ".png" in path_h):
    imh = cv.imread(path_h)
    imi = cv.imread(path_i)
    imm = cv.imread(path_m)
    # ! important: a lower version of 
    # orig_shape = {'x': im_large.shape[0], 'y': im_large.shape[1]} // use true image size instead
    w_num = orig_shape['x'] // target_shape['x']
    h_num = orig_shape['y'] // target_shape['y']
    for i in range(w_num):
        for j in range(h_num):
            chh = imh[i * target_shape['x']:(i + 1) * target_shape['x'],
                            j * target_shape['y']:(j + 1) * target_shape['y'],
                            :]
            # print(chh)
            th_blank = 200
            margin_pp = 4
            if chh[:target_shape['x']//margin_pp, :, :].mean() > th_blank or \
                chh[int((margin_pp - 1)*target_shape['x']//margin_pp):, :, :].mean() > th_blank or \
                chh[:, :target_shape['x']//margin_pp,  :].mean() > th_blank or \
                chh[:, int((margin_pp - 1)*target_shape['x']//margin_pp):,  :].mean() > th_blank:
                continue
            chi = imi[i * target_shape['x']:(i + 1) * target_shape['x'],
                            j * target_shape['y']:(j + 1) * target_shape['y'],
                            :]
            chm = imm[i * target_shape['x']:(i + 1) * target_shape['x'],
                            j * target_shape['y']:(j + 1) * target_shape['y'],
                            :]

            ph_new = path_h.replace(orig_path, target_path).replace(".png", "(%d, %d)" % (i, j) + ".png")
            # print(ph_new)
            pi_new = path_i.replace(orig_path, target_path).replace(".png", "(%d, %d)" % (i, j) + ".png")
            pm_new = path_m.replace(orig_path, target_path).replace(".png", "(%d, %d)" % (i, j) + ".png")

            cv.imwrite(ph_new, chh)
            cv.imwrite(pi_new, chi)
            cv.imwrite(pm_new, chm)