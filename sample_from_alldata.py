#%%
# This script samples a part of data from the whole dataset
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import os
import shutil as shu
#%%
print("-*"*20)
sample_rate = 10
orig_shape = {'x': 256, 'y': 256}

dataset_name = "TILES_(%s, %s)" % (orig_shape['x'], orig_shape['y'])
orig_path = "/raid/ji/DATA/%s/" % dataset_name
print(orig_path)

target_path = orig_path.replace("(%s, %s)" % (orig_shape['x'], orig_shape['y']), "%s(1 in %s)" % (orig_shape['x'], sample_rate))
print(target_path)

#%%
# 1. Copy directory tree
# copy the txt labels to the new position. skip the images
def ig_f(dir, files):
    return [f for f in files if (os.path.isfile(os.path.join(dir, f)) and (not (".txt" in os.path.join(dir, f))))]

if not (os.path.exists(target_path)):
    shu.copytree(orig_path, target_path, ignore=ig_f)
#%%
for imtype in [ "HE", "Mask"]:
    for roots, dirs, files in os.walk(orig_path + imtype + "/"):
        k = 0
        for file in sorted(files):
            k+=1
            filepath = os.path.join(roots, file)
            if not ("Tumor" in filepath): 
                continue 
            new_path = roots.replace("(%s, %s)" % (orig_shape['x'], orig_shape['y']), "%s(1 in %s)" % (orig_shape['x'], sample_rate))+\
                        "/" + os.path.splitext(file)[0] + ".png"
            if not (os.path.exists(new_path)):
                if k % sample_rate == 0:
                    # cv.imwrite(new_path, chip)
                    shu.copyfile(filepath, new_path)

# %%
