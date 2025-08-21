import os
import h5py
import pandas as pd
import numpy as np
import gc
import scipy.io
from matplotlib import pyplot as plt

exclude_ous = [110]
exclude_maastro = [5]
ous_image_path = 'P:/REALTEK-HeadNeck-Project/Head-and-Neck/HNPrepped_Data/Dataset_3D_U_NET/imdata/'
maastro_image_path = 'P:/REALTEK-HeadNeck-Project/Head-and-Neck/HNPrepped_Data/HN_MAASTRICT_29062020_Matlab/imdata_cropped_new/'

for fn in os.listdir(ous_image_path):
    with h5py.File(ous_image_path + fn, 'r') as patient_file:
        print(fn)
        tumour = patient_file['imdata']['tumour'][:]
        node = patient_file['imdata']['nodes'][:]
        if tumour.shape != (173, 191, 265):
            print('wrong shape', fn, tumour.shape)


for fn in os.listdir(maastro_image_path):
    with h5py.File(maastro_image_path + fn, 'r') as patient_file:
        print(fn)
        tumour = patient_file['imdata']['tumour'][:]
        node = patient_file['imdata']['nodes'][:]
        if tumour.shape != (173, 191, 265):
            print('wrong shape', fn, tumour.shape)


def find_bounding_box(mask):
    # Find indices of non-zero elements
    non_zero_indices = np.argwhere(mask > 0)

    # Get min and max along each axis
    x_min, y_min, z_min = non_zero_indices.min(axis=0)
    x_max, y_max, z_max = non_zero_indices.max(axis=0)

    # Return bounding box coordinates (inclusive of min, exclusive of max)
    return (x_min, x_max + 1, y_min, y_max + 1, z_min, z_max + 1)


data = []
for fn in os.listdir(ous_image_path):
    with h5py.File(ous_image_path + fn, 'r') as patient_file:
        print(fn)
        tumour = patient_file['imdata']['tumour'][:]
        node = patient_file['imdata']['nodes'][:]
        x1, x2, y1, y2, z1, z2 = find_bounding_box(tumour + node)
        data.append({
            'pid': fn,
            'x1': x1,
            'x2': x2,
            'y1': y1,
            'y2': y2,
            'z1': z1,
            'z2': z2,
        })
for fn in os.listdir(maastro_image_path):
    with h5py.File(maastro_image_path + fn, 'r') as patient_file:
        print(fn)
        tumour = patient_file['imdata']['tumour'][:]
        node = patient_file['imdata']['nodes'][:]
        x1, x2, y1, y2, z1, z2 = find_bounding_box(tumour + node)
        data.append({
            'pid': fn,
            'x1': x1,
            'x2': x2,
            'y1': y1,
            'y2': y2,
            'z1': z1,
            'z2': z2,
        })

info_df = pd.DataFrame(data)
info_df.to_csv('data_info/bounding_boxes.csv', index=False)
gc.collect()
info_df['x'] = info_df.x2 - info_df.x1
info_df['y'] = info_df.y2 - info_df.y1
info_df['z'] = info_df.z2 - info_df.z1

info_df.x.max()  # 165
info_df.y.max()  # 132
info_df.z.max()  # 124

# 160-176, 176-192, 256-272
info_df.x2.max() - info_df.x1.min()  # 165
info_df.y2.max() - info_df.y1.min()  # 165
info_df.z2.max() - info_df.z1.min()  # 170

info_df[info_df.z1 == 25]
with h5py.File(maastro_image_path + 'PMA016.mat', 'r') as patient_file:
    tumour = patient_file['imdata']['tumour'][:]
    node = patient_file['imdata']['nodes'][:]
    ct = patient_file['imdata']['CT'][:]

idx = 125
plt.imshow(ct[idx], 'gray')
plt.contour(tumour[idx], levels=[0.5], colors='red')
plt.contour(node[idx], levels=[0.5], colors='yellow')
plt.show()

check = np.argwhere(tumour > 0)
check[[check[..., 2].argmin()]]


info_df[info_df.z2 == 195]

with h5py.File(ous_image_path + 'imdata_P008.mat', 'r') as patient_file:
    tumour = patient_file['imdata']['tumour'][:]
    node = patient_file['imdata']['nodes'][:]
    ct = patient_file['imdata']['CT'][:]
check = np.argwhere(tumour + node > 0)
check[[check[..., 2].argmax()]]

idx = 172
plt.imshow(ct[idx], 'gray')
plt.contour(tumour[idx], levels=[0.5], colors='red')
plt.contour(node[idx], levels=[0.5], colors='yellow')
plt.show()


# official size 176-192-256
# patch size 176-144-128
