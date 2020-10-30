#%%
import cv2 as cv
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob

w_tar, h_tar = 2048, 2048
size = w_tar

basepath = "/home/cunyuan/4tb/Kimura/DATA"

dpl = {"IHC": basepath+ "/TILES_(%d, %d)/DAB/*/*/*/*/" % (size, size),
       "Mask": basepath + "/TILES_(%d, %d)/Mask/*/*/*/*/" % (size, size)}

basepath = "/Users/cunyuan/Downloads"
dpl = {"IHC": basepath+ "/DAB/*/*/Tumor/*/",
       "Mask": basepath + "/Mask/*/*/Tumor/*/"}
# dpl = [for x in dpl]

k = 0
l = []
dp = "Mask"

#%%
for m in glob.glob(dpl[dp] + "/*.png") + glob.glob(dpl[dp] + "/*.tif"):
    # print(c,d,m,n)
    k+=1
    imm = cv.imread(m).astype(np.uint8)
    imm = cv.cvtColor(imm, cv.COLOR_BGR2GRAY)/255

    w, h = imm.shape[0], imm.shape[1]
    nw, nh = w//w_tar, h//h_tar
    
    l.append(np.sum(imm) / (w*h))
#%%
fig=plt.figure(figsize=(8,6), dpi=300)
ax = fig.add_subplot(1, 1, 1)

# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 1000, 100)
minor_ticks = np.arange(0, 1000, 10)

hl = ax.hist(l, bins=128, rwidth=0.5)

xlim = np.max(l)
ylim = np.max(hl[0])

ax.set_xticks(major_ticks*xlim/1000)
ax.set_xticks(minor_ticks*xlim/1000, minor=True)
ax.set_yticks(major_ticks*ylim/1000)
ax.set_yticks(minor_ticks*ylim/1000, minor=True)


# Or if you want different settings for the grids:
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
# And a corresponding grid
# ax.grid(which='both')

plt.tight_layout()
plt.title("Histogram of + pixel ratio in the Masks")
plt.show()
print(k)

#%%
fig=plt.figure(figsize=(8,6), dpi=300)
ax = fig.add_subplot(1, 1, 1)

# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 1000, 100)
minor_ticks = np.arange(0, 1000, 10)

hl = ax.hist(l, bins=128, rwidth=0.5)

xlim = np.max(l)
ylim = np.max(hl[0])

ax.set_xticks(major_ticks*xlim/1000)
ax.set_xticks(minor_ticks*xlim/1000, minor=True)
ax.set_yticks(major_ticks*ylim/1000)
ax.set_yticks(minor_ticks*ylim/1000, minor=True)


ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

plt.tight_layout()
plt.title("Histogram of + pixel ratio in the Masks")
plt.show()

print(k)
# %%
""" 
# Threshold analysis
L = np.load("/Users/cunyuan/Downloads/pratio256.npy")
Ln = np.log10([len(L[L>x]) for x in np.linspace(0,1,1000)])
Lm = np.array([L[L>x].mean() for x in np.linspace(0, 1, 1000)]) 

fig=plt.figure(figsize=(8,8), dpi=300) 
ax1 = fig.subplots() 
ax2  =ax1.twinx() 
# Major ticks every 20, minor ticks every 5 
major_ticks = np.arange(0, 1000, 100) 
minor_ticks = np.arange(0, 1000, 10) 

ax1.plot(np.linspace(0,1,1000), np.log10([len(L[L>x]) for x in np.linspace(0,1,1000)]), 'r') 

xlim = np.max(1.01) 
ylim = np.log10([len(L[L>x]) for x in np.linspace(0,1,1000)])[0] 
ylim += ylim *0.1 

# ax1.set_xticks(major_ticks*xlim/1010) 
ax1.set_xticks(minor_ticks*xlim/1010, minor=True) 
# ax1.set_yticks(major_ticks*ylim/1100) 
ax1.set_yticks(minor_ticks*ylim/1100, minor=True) 

ax2.plot(np.linspace(0,1,1000), Lm) 
# ax1.plot([0.41, 0.41], [0, 4.46], 'k--') 
# ax2.plot([0.41, 1], [0.5, 0.5], 'b--') 
# ax1.plot([0,0.41],[4.46, 4.46], 'r--') 
xlim = np.max(1.01) 
ylim = 1 
ylim += ylim*0.1 

ax2.set_xticks(major_ticks*xlim/1010) 
ax2.set_xticks(minor_ticks*xlim/1010, minor=True) 
ax2.set_yticks(major_ticks*ylim/1100) 
ax2.set_yticks(minor_ticks*ylim/1100, minor=True) 

# Or if you want different settings for the grids: 
ax1.grid(which='minor', alpha=0.2) 
ax1.grid(which='major', alpha=0.5) 
# And a corresponding grid 
# ax.grid(which='both') 
# Or if you want different settings for the grids: 
ax2.grid(which='minor', alpha=0.2) 
ax2.grid(which='major', alpha=0.5) 

ax1.set_ylabel("Log10 of sample #", color = "r") 
ax2.set_ylabel("Mean of (+) ratio", color="b") 
ax1.set_xlabel("(+) pixel threshold") 
plt.tight_layout() 
# plt.title("# of samples remaining vs. balancing threshold") 
"""
#%%
fig = plt.figure(figsize=(16,16), dpi=300) 
ax = plt.subplot(2,2,1) 
bm1 = sqrt(Lm *(1-Lm))*Lm*10**Ln 
plt.plot(bm1) 
plt.grid() 
plt.tight_layout()
plt.xlabel("(+) ratio thresh") 
plt.ylabel("sqrt(Lm *(1-Lm))*Lm*10**Ln") 
bm1[np.isnan(bm1)] = 0 
print(bm1.argmax())

ax = plt.subplot(2,2,2) 
bm1 = sqrt(Lm *(1-Lm))*np.log10(Lm*10**Ln) 
plt.plot(bm1) 
plt.grid() 
plt.tight_layout()
plt.xlabel("(+) ratio thresh") 
plt.ylabel("sqrt(Lm *(1-Lm))*(Ln + log10(Lm))") 
bm1[np.isnan(bm1)] = 0 
print(bm1.argmax())

ax = plt.subplot(2,2,3) 
bm1 = Lm *(1-Lm)
plt.plot(bm1)
plt.grid() 
plt.tight_layout()
plt.xlabel("(+) ratio thresh") 
plt.ylabel("sqrt(Lm *(1-Lm))*Lm*Ln") 
bm1[np.isnan(bm1)] = 0 
print(bm1.argmax())

ax = plt.subplot(2,2,4)
bm1 = Lm*10**Ln 
plt.plot(bm1) 
plt.tight_layout()
plt.grid() 
plt.xlabel("(+) ratio thresh") 
plt.ylabel("(+) amount)") 
bm1[np.isnan(bm1)] = 0 
print(bm1.argmax())

# %%
