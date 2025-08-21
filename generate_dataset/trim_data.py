import os
import h5py
import pandas as pd
import numpy as np
import gc
import scipy.io
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit


exclude_ous = [110]
exclude_maastro = [5]
ous_image_path = 'P:/REALTEK-HeadNeck-Project/Head-and-Neck/HNPrepped_Data/Dataset_3D_U_NET/imdata/'
maastro_image_path = 'P:/REALTEK-HeadNeck-Project/Head-and-Neck/HNPrepped_Data/HN_MAASTRICT_29062020_Matlab/imdata_cropped_new/'
ous_info_df = pd.read_csv(
    'P:/REALTEK-HeadNeck-Project/Head-and-Neck/Outcome_prediction/data/clinical_ous.csv', delimiter=';')
maastro_info_df = pd.read_csv(
    'P:/REALTEK-HeadNeck-Project/Head-and-Neck/Outcome_prediction/maastro/clinical_maastro_full.csv', delimiter=';')

ous_info_df = ous_info_df[~ous_info_df['patient_id'].isin(exclude_ous)].reset_index(drop=True)
maastro_info_df = maastro_info_df[~maastro_info_df['patient_id'].isin(
    exclude_maastro)].reset_index(drop=True)

# Create a stratification key by combining the encoded columns and uicc8_III-IV
strat_cols = ['cavum_oris', 'oropharynx', 'hypopharynx', 'larynx', 'uicc8_III-IV', 'hpv_related']
ous_info_df['strat_key'] = ous_info_df[strat_cols].astype(str).agg('-'.join, axis=1)

ous_test_idx = []
rng = np.random.default_rng(seed=42)  # Set random seed for reproducibility
for strat_key in ous_info_df['strat_key'].unique():
    strat_df = ous_info_df[ous_info_df['strat_key'] == strat_key]
    if len(strat_df) > 20:
        # randomly select 3 patients for the test set
        test_patient = rng.choice(strat_df['patient_id'].values, size=3)
        ous_test_idx.extend(test_patient)
    elif len(strat_df) > 5:
        # randomly select two patients for the test set
        test_patient = rng.choice(strat_df['patient_id'].values, size=2)
        ous_test_idx.extend(test_patient)
    else:
        # raandomly select one patient for the test set
        test_patient = rng.choice(strat_df['patient_id'].values, size=1)
        ous_test_idx.extend(test_patient)


def describe_columns(df):
    for col in df.columns:
        vals = df[col]
        if pd.api.types.is_numeric_dtype(vals):
            unique_vals = vals.dropna().unique()
            if set(unique_vals).issubset({0, 1}):
                sum_1 = vals.sum()
                ratio_1 = (vals == 1).mean()
                ratio_nan = vals.isna().mean()
                print(f"{col}: binary, sum 1 = {sum_1}, ratio 1 = {ratio_1:.2f}, ratio NaN = {ratio_nan:.2f}")
            else:
                mean = vals.mean()
                std = vals.std()
                ratio_nan = vals.isna().mean()
                print(f"{col}: mean = {mean:.2f}, SD = {std:.2f}, ratio NaN = {ratio_nan:.2f}")
        else:
            ratio_nan = vals.isna().mean()
            print(f"{col}: non-numeric, ratio NaN = {ratio_nan:.2f}")


print("OUS info columns summary:")
describe_columns(ous_info_df[~ous_info_df['patient_id'].isin(ous_test_idx)])
describe_columns(ous_info_df[ous_info_df['patient_id'].isin(ous_test_idx)])
# test_idx
# [13, 105, 90, 5, 16, 88, 38, 177,
#  40, 24, 182, 249,
#  205, 141, 141, 222,
#  168, 100, 242, 196]

ous_train_idx = ous_info_df[~ous_info_df['patient_id'].isin(ous_test_idx)]['patient_id'].values
rng.shuffle(ous_train_idx)
rng.shuffle(ous_test_idx)
# ous_train_idx
# 209,  35,  42, 250,  61,  86, 203, 166, 159, 206,  15, 128,  62,
#        207, 103, 143, 225, 171, 138, 228,  43, 129, 102, 216,  98, 150,
#        104, 162,  21, 149, 189, 126,  60,  55,  10,  49, 243,  73, 152,
#         18,  72, 117,  81, 147, 215,  45, 121, 231, 246, 247, 120,  54,
#         11, 198,  89, 142, 124,  78,  87, 253,   8,  82, 107, 176,  48,
#        213,  25, 210, 240,   2, 185, 118,  96, 125, 113,  64, 180, 229,
#         34,  26, 218, 133,  12, 184,  74, 244, 154, 165, 194, 201, 178,
#        153, 239, 195, 123,  71, 130, 167,  32, 191, 116, 161, 158, 252,
#        109,  27, 131, 146, 111, 157,  36, 169, 136, 139, 127, 230,  93,
#         99, 172, 144, 108, 223, 204,  44, 173, 248, 140,  37, 164, 112,
#        187,  22, 163, 200,  97,   4,  67, 170,  31, 254,  83, 114, 220,
#         65, 106, 232, 181, 212, 151, 217,  23, 202,  68, 241,  95,  91,
#        224,  39, 233,  29,  94, 155, 197,  52,  30,  70, 156,  56,  50,
#         92,  57, 115,  77,  14, 148, 199,  66
# ous_test_idx
# 90, 24, 100, 40, 5, 205, 38, 242, 141, 88, 182, 16,
#     196, 177, 249, 105, 13, 222, 141, 168


def pad_crop_3d(array, target_shape):
    """
    Pads and crops a 3D array to match the target shape.

    Parameters:
    - array: np.ndarray of shape (D, H, W)
    - target_shape: tuple (target_D, target_H, target_W)

    Returns:
    - np.ndarray of shape target_shape
    """
    target_D, target_H, target_W = target_shape

    # Pad axis 0 (depth) before
    pad_d = max(target_D - array.shape[0], 0)
    array = np.pad(array, ((pad_d, 0), (0, 0), (0, 0)), mode='constant')

    # Pad axis 1 (height) after
    pad_h = max(target_H - array.shape[1], 0)
    array = np.pad(array, ((0, 0), (0, pad_h), (0, 0)), mode='constant')

    # Crop axis 2 (width) after
    if array.shape[2] > target_W:
        array = array[:, :, :target_W]

    return array


# official size 176-192-256
# patch size 176-144-128

target_shape = (176, 192, 256)
num_patients = [len(ous_train_idx), len(ous_test_idx)]

with h5py.File('../../ous_hnc_3d.h5', 'w') as f:
    f.create_group('fold_0')
    f.create_group('fold_1')
for i in range(2):
    with h5py.File('../../ous_hnc_3d.h5', 'a') as f:
        f[f'fold_{i}'].create_dataset(
            'input', shape=(num_patients[i], *target_shape, 2), dtype='f4',
            chunks=(1, 176, 192, 256, 2), compression='lzf')
        f[f'fold_{i}'].create_dataset(
            'target', shape=(num_patients[i], *target_shape, 2), dtype='f4',
            chunks=(1, 176, 192, 256, 2), compression='lzf')
        f[f'fold_{i}'].create_dataset(
            'patient_idx', shape=(num_patients[i],), dtype='i4',
            compression='lzf')


for i, pid in enumerate(ous_train_idx):
    with h5py.File(ous_image_path + f'imdata_P{pid:03d}.mat', 'r') as patient_file:
        print(f'{i}, imdata_P{pid:03d}.mat')
        tumour = pad_crop_3d(patient_file['imdata']['tumour'][:], target_shape)
        node = pad_crop_3d(patient_file['imdata']['nodes'][:], target_shape)
        pt = pad_crop_3d(patient_file['imdata']['PT'][:], target_shape)
        ct = pad_crop_3d(patient_file['imdata']['CT'][:], target_shape)
        input_image = np.stack([ct, pt], axis=-1).squeeze()
        target_mask = np.stack([tumour, node], axis=-1).squeeze()
    with h5py.File('../../ous_hnc_3d.h5', 'a') as f:
        f[f'fold_0']['input'][i] = input_image
        f[f'fold_0']['target'][i] = target_mask
        f[f'fold_0']['patient_idx'][i] = pid

for i, pid in enumerate(ous_test_idx):
    with h5py.File(ous_image_path + f'imdata_P{pid:03d}.mat', 'r') as patient_file:
        print(f'{i}, imdata_P{pid:03d}.mat')
        tumour = pad_crop_3d(patient_file['imdata']['tumour'][:], target_shape)
        node = pad_crop_3d(patient_file['imdata']['nodes'][:], target_shape)
        pt = pad_crop_3d(patient_file['imdata']['PT'][:], target_shape)
        ct = pad_crop_3d(patient_file['imdata']['CT'][:], target_shape)
        input_image = np.stack([ct, pt], axis=-1).squeeze()
        target_mask = np.stack([tumour, node], axis=-1).squeeze()
    with h5py.File('../../ous_hnc_3d.h5', 'a') as f:
        f[f'fold_1']['input'][i] = input_image
        f[f'fold_1']['target'][i] = target_mask
        f[f'fold_1']['patient_idx'][i] = pid


# for fn in os.listdir(maastro_image_path):
#     with h5py.File(maastro_image_path + fn, 'r') as patient_file:
#         print(fn)
#         tumour = patient_file['imdata']['tumour'][:]
#         node = patient_file['imdata']['nodes'][:]
#         if tumour.shape != (173, 191, 265):
#             print('wrong shape', fn, tumour.shape)

# # get data
# data = []
# for fn in os.listdir(ous_image_path):
#     with h5py.File(ous_image_path + fn, 'r') as patient_file:
#         print(fn)
#         tumour = patient_file['imdata']['tumour'][:]
#         node = patient_file['imdata']['nodes'][:]
# for fn in os.listdir(maastro_image_path):
#     with h5py.File(maastro_image_path + fn, 'r') as patient_file:
#         tumour = patient_file['imdata']['tumour'][:]
#         node = patient_file['imdata']['nodes'][:]
