import shutil as shu
import os, glob
from pathlib import Path

""" 
! 目的：在所有“TIles” 文件夹下面创建面向“TIles/Tumor"的软连接
! The purpose of this script is to create soft links for "Tiles/Tumor" on every "Tiles" folder in the WSIs
"""
size = 256
HOME_PATH = str(Path.home())
glob_keyword = HOME_PATH + "/4tb/Kimura/DATA/TILES_(%d, %d)/*/*" % (size, size)

for WSIs in glob.glob(glob_keyword):
    print(WSIs)
    # print("ln -s %s %s"%(WSIs + "/Tiles/Tumor", WSIs + "_tumoronly"))
    os.system("ln -s \"%s\" \"%s\""%(WSIs + "/Tiles/Tumor", WSIs + "_tumoronly"))