""" 
Divide large tiles to smaller tiles and save smaller ones separately
将大 tile 分割成小的并另外保存
 """
import cv2 as cv
import os
import shutil as shu

print("-*"*20)
target_shape = {'x': 256, 'y': 256}
orig_shape = {'x': 2048, 'y': 2048}

dataset_name = "TILES_(%s, %s)" % (orig_shape['x'], orig_shape['y'])
orig_path = "/home/cunyuan/4tb/Kimura/DATA/%s/HE/" % dataset_name
print(orig_path)
target_path = orig_path.replace("(%s, %s)" % (orig_shape['x'], orig_shape['y']), "(%s, %s)" % (target_shape['x'], target_shape['y']))

# Ignoring pattern

def ig_f(dir, files):
    return [f for f in files if (os.path.isfile(os.path.join(dir, f)) and (not (".txt" in os.path.join(dir, f))))]

if not (os.path.exists(target_path)):
    shu.copytree(orig_path, target_path, ignore=ig_f)

for roots, dirs, files in os.walk(orig_path):
    for file in files:
        filepath = os.path.join(roots, file)

        if (".tif" in filepath or ".png" in filepath):
            im_large = cv.imread(filepath)
            # orig_shape = {'x': im_large.shape[0], 'y': im_large.shape[1]} // use true image size instead
            w_num = orig_shape['x'] // target_shape['x']
            h_num = orig_shape['y'] // target_shape['y']
            for i in range(w_num):
                for j in range(h_num):
                    chip = im_large[i * target_shape['x']:(i + 1) * target_shape['x'],
                                    j * target_shape['y']:(j + 1) * target_shape['y'],
                                    :]

                    new_path = roots.replace("(%s, %s)" % (orig_shape['x'], orig_shape['y']), "(%s, %s)" % (target_shape['x'], target_shape['y'])) +\
                                "/" + os.path.splitext(file)[0] + "(%d, %d)" % (i, j) + ".tif"
                    # print(new_path)
                    # print(roots)
                    if not (os.path.exists(new_path)):
                        cv.imwrite(new_path, chip)

