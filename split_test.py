import cv2 as cv
import numpy as np
import os
import shutil as shu
from glob import glob
from tqdm import tqdm

num_select = 6

print("-*" * 20)
orig_shape = {"x": 2048, "y": 2048}

dataset_name = "TILES_FULL_2048"
orig_path = "/wd_0/ji/%s/" % dataset_name
print(orig_path)
target_path = orig_path + "/test/"

if not (os.path.exists(target_path)):
    os.mkdir(target_path)
    for imtype in ["HE", "Mask", "IHC"]:
        os.mkdir(target_path +imtype)

fold = {'G1': ['7015','1052','3768','7553','5425','3951','2189','3135','3315','4863','4565','2670','3006','3574','3597','3944','1508','0669','1115'],
'G2': ['5256','6747','8107','1295','2072','2204','3433','7144','1590','2400','6897','1963','2118','4013','4498','0003','2943','3525','2839'],
'G3': ['2502','7930','7885','0790','1904','3235','2730','7883','3316','4640','0003','1883','2913','1559','2280','6018','2124','8132','2850']}
case_list = np.ravel(np.vstack([fold[key] for key in fold.keys()]))

for k, case in enumerate(case_list):
    phl, pil, pml = tuple(
        [
            sorted(glob(orig_path + imtype + "/*" + case + "*.png"))
            for imtype in ["HE", "IHC", "Mask"]
        ]
    )
    print(orig_path + "HE" + "/*" + case + "*.png")
    assert len(phl) == len(pml) == len(pil)
    idx_range = len(phl)
    idx_select = np.random.choice(idx_range, 10, replace=False)
    tlh, tli, tlm = np.array(phl)[idx_select], np.array(pil)[idx_select],np.array(pml)[idx_select]
    tlh = list(tlh)
    tlm = list(tlm)
    tli = list(tli)
    for t in tlh+tli+tlm:
        os.system("mv %s %s"%(t, t.replace(orig_path, target_path)))