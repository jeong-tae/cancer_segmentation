import glob
import cv2
import numpy as np
root = 'data/CT_PNG100/NSCLC-Radiomics'
patients = glob.glob(root + '/*')

lines = []
for p in patients:
    image_paths = glob.glob(p + '/*')
    gt_paths = [path for path in image_paths if 'gt' in path]
    f_paths = [path for path in image_paths if 'gt' not in path]

    gt_paths = sorted(gt_paths)
    f_paths = sorted(f_paths)

    if len(gt_paths) != len(f_paths):
        raise ValueError("Size is not same")

    for i in range(len(gt_paths)):
        img = cv2.imread(gt_paths[i], cv2.IMREAD_GRAYSCALE)
        labeled = np.sum(img > 1)
        if labeled == 0:
            continue
        lines.append(f_paths[i] + '\t' + gt_paths[i])

fout = open('frames2.txt', 'w')
for line in lines:
    fout.write(line + '\n')
fout.close()
