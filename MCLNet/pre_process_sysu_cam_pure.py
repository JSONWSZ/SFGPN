"""
Generate camid npy files for the SYSU-MM01 "pure" dataset.
Run this once before training MCLNet + FRL/SCG.

The camid is extracted from the original image path:
  cam1 -> 1, cam2 -> 2, cam3 -> 3, cam4 -> 4, cam5 -> 5, cam6 -> 6

Output:
  train_rgb_resized_camid.npy
  train_ir_resized_camid.npy
"""

import numpy as np
import os

# ====== CONFIG — adjust paths as needed ======
ORI_PATH = '../Datasets/SYSU-MM01/'
PURE_PATH = '../Datasets/SYSU-MM01-Pure/'
NPY_PATH = '../Datasets/SYSU-MM01-npy/'

rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
ir_cameras = ['cam3', 'cam6']

# Load id info
file_path_train = os.path.join(ORI_PATH, 'exp/train_id.txt')
file_path_val = os.path.join(ORI_PATH, 'exp/val_id.txt')

with open(file_path_train, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_train = ["%04d" % x for x in ids]

with open(file_path_val, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_val = ["%04d" % x for x in ids]

id_train.extend(id_val)

# Collect files from pure dataset (same as SFGPN pre_process)
files_rgb = []
files_ir = []
for id in sorted(id_train):
    for cam in rgb_cameras:
        img_dir = os.path.join(PURE_PATH, cam, id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            files_rgb.extend(new_files)
    for cam in ir_cameras:
        img_dir = os.path.join(PURE_PATH, cam, id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            files_ir.extend(new_files)

# Extract raw camid from image path (e.g., ".../cam1/0001/xxx.jpg" -> raw=1)
rgb_camids_raw = []
for img_path in files_rgb:
    camid = int(img_path[-15])  # camera digit in path
    rgb_camids_raw.append(camid)

ir_camids_raw = []
for img_path in files_ir:
    camid = int(img_path[-15])
    ir_camids_raw.append(camid)

# Map raw camid (1,2,3,4,5,6) to 0-indexed labels (0,1,2,3,4,5)
# This is required by MCCA_cam_loss which uses num_classes=6
all_camids = set(rgb_camids_raw + ir_camids_raw)
camid2label = {cid: i for i, cid in enumerate(sorted(all_camids))}
print(f"Camid mapping: {camid2label}")

rgb_camids = [camid2label[c] for c in rgb_camids_raw]
ir_camids = [camid2label[c] for c in ir_camids_raw]

# Save
np.save(os.path.join(NPY_PATH, 'train_rgb_resized_camid.npy'), np.array(rgb_camids))
np.save(os.path.join(NPY_PATH, 'train_ir_resized_camid.npy'), np.array(ir_camids))

print(f"RGB camids: {len(rgb_camids)} images, unique: {set(rgb_camids)}")
print(f"IR camids:  {len(ir_camids)} images, unique: {set(ir_camids)}")
print("Done! Camid files saved to:", NPY_PATH)
