import glob
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
        lines.append(f_paths[i] + '\t' + gt_paths[i])

fout = open('frames.txt', 'w')
for line in lines:
    fout.write(line + '\n')
fout.close()
