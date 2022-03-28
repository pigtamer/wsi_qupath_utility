#%%
import cv2 as cv
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob, shutil as shu

th = 0.16
size =256

basepath = "/raid/ji/DATA/"

dpl = {
       "HE": basepath + "/TILES_(%d, %d)/HE/*/*/*/*/" % (size, size),
       "IHC": basepath + "/TILES_(%d, %d)/IHC/*/*/*/*/" % (size, size),
       "Mask": basepath + "/TILES_(%d, %d)/Mask/*/*/*/*/" % (size, size),
       "DAB": basepath+ "/TILES_(%d, %d)/DAB/*/*/*/*/" % (size, size),
       }
# dpl = {
#        "HE": basepath + "/TILES_(%d, %d)/HE/*/*/Tumor/Tumor" % (size, size),
#        "IHC": basepath + "/TILES_(%d, %d)/IHC/*/*/Tumor/Tumor" % (size, size),
#        "Mask": basepath + "/TILES_(%d, %d)/Mask/*/*/Tumor/Tumor" % (size, size),
#        "DAB": basepath+ "/TILES_(%d, %d)/DAB/*/*/Tumor/Tumor" % (size, size),
#        }

# for dp in ["HE", "IHC", "Mask", "DAB"]:
#     print(sort(glob.glob(dpl[dp])))
    
#     for f in sort(glob.glob(dpl[dp])):
#         if not "only" in str(f):
#             os.remove(f)
#%%
""" basepath = "/Users/cunyuan/Downloads"
dpl = {"IHC": basepath+ "/DAB/*/*/Tumor/*/",
       "Mask": basepath + "/Mask/*/*/Tumor/*/"}
 """
#%%
def ig_f(dir, files):
    return [
        f
        for f in files
        if (
            (
            os.path.isfile(os.path.join(dir, f)) # is a file
            and (not ("txt" in os.path.join(dir, f))) # but not the annot
            )
            or  ("only" in os.path.join(dir, f)) # 如果之前运行了create_links，在拷贝目录树的时候会把tumor_only作为文件夹拎过来，因此添加这一规则
        )
    ]

size0 = 2048
if not os.path.exists(basepath+ "/TILES_(%d, %d)_%s"% (size, size, th)):
    shu.copytree(basepath+ "/TILES_(%d, %d)"% (size0, size0), basepath+ "/TILES_(%d, %d)_%s"% (size, size, th), ignore=ig_f)
k=0
print(">>> Start.")
#%%
for c, d, m, n in zip(
    *[
        sort(glob.glob(dpl[dp] + "/*.png") + glob.glob(dpl[dp] + "/*.tif"))
        for dp in ["HE", "IHC", "Mask", "DAB"]
    ]
):
    # print(str(m))
    if ("only" in str(m)):
        print("ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("ERROR!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        # print("XXXXXX----", c,"\n", d, "\n", m, "\n", n)
        k += 1
        imm = cv.imread(m).astype(np.uint8)
        imm = cv.cvtColor(imm, cv.COLOR_BGR2GRAY) / 255
        w, h = imm.shape[0], imm.shape[1]
        nw, nh = w // size, h // size

        pr = np.sum(imm) / (w * h)
        if pr >= th:
            print(str(m) + "[VALID]")
            # print(">>>\n"*3)         
            # plt.imshow(imm, cmap='gray');plt.show()
            for x in [c, d, m, n]:
                # print(str(x).replace("TILES_(%d, %d)/" % (size, size),
                # "TILES_(%d, %d)_%s/"% (size, size, th)))
                shu.copyfile(str(x), 
                            str(x).replace("TILES_(%d, %d)/" % (size, size),
                                            "TILES_(%d, %d)_%s/"% (size, size, th))
                            )
            print("--- Transferred.")