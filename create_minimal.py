#%%
import cv2 as cv
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob

th = 0.41
w_tar, h_tar = 2048, 2048
size = w_tar

basepath = "/home/cunyuan/4tb/Kimura/DATA"

dpl = {
    "IHC": basepath + "/TILES_(%d, %d)/DAB/*/*/*/*/" % (size, size),
    "Mask": basepath + "/TILES_(%d, %d)/Mask/*/*/*/*/" % (size, size),
}

""" basepath = "/Users/cunyuan/Downloads"
dpl = {"IHC": basepath+ "/DAB/*/*/Tumor/*/",
       "Mask": basepath + "/Mask/*/*/Tumor/*/"}
 """

#%%
for c, d, m, n in zip(
    *[
        sort(glob.glob(dpl[dp] + "/*.png") + glob.glob(dpl[dp] + "/*.tif"))
        for dp in dpl.keys()
    ]
):
    print(c,"\n", d, "\n", m, "\n", n)
    k += 1
    imm = cv.imread(m).astype(np.uint8)
    imm = cv.cvtColor(imm, cv.COLOR_BGR2GRAY) / 255

    w, h = imm.shape[0], imm.shape[1]
    nw, nh = w // w_tar, h // h_tar

    pr = np.sum(imm) / (w * h)
    if pr >= 0.41:
        for x in [c, d, m, n]:
            print(str(x).replace("TILES_(%d, %d)/" % (size, size),
            "TILES_(%d, %d)_%s/"% (size, size, th)))
#%%
#%%
fig = plt.figure(figsize=(8, 6), dpi=300)
ax = fig.add_subplot(1, 1, 1)

# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 1000, 100)
minor_ticks = np.arange(0, 1000, 10)

hl = ax.hist(l, bins=128, rwidth=0.5)

xlim = np.max(l)
ylim = np.max(hl[0])

ax.set_xticks(major_ticks * xlim / 1000)
ax.set_xticks(minor_ticks * xlim / 1000, minor=True)
ax.set_yticks(major_ticks * ylim / 1000)
ax.set_yticks(minor_ticks * ylim / 1000, minor=True)

ax.grid(which="minor", alpha=0.2)
ax.grid(which="major", alpha=0.5)

plt.tight_layout()
plt.title("Histogram of + pixel ratio in the Masks")
plt.show()

print(k)
# %%
