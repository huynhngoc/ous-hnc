import h5py
import numpy as np
import os


ous_data_path = '../datasets/ous_hnc_3d.h5'
maastro_data_path = '../datasets/maastro_hnc_3d.h5'
output_data_path = '../datasets/all_hnc_3d.h5'


with h5py.File(ous_data_path, 'r') as f:
    ous_train_pids = f['fold_0']['patient_idx'][:]
    ous_test_pids = f['fold_1']['patient_idx'][:]
with h5py.File(maastro_data_path, 'r') as f:
    maastro_train_pids = f['fold_0']['patient_idx'][:]
    maastro_test_pids = f['fold_1']['patient_idx'][:]

ous_train_pids += 1000
ous_test_pids += 1000
maastro_train_pids += 2000
maastro_test_pids += 2000
print('OUS train pids:', ous_train_pids)
print('OUS test pids:', ous_test_pids)
print('Maastro train pids:', maastro_train_pids)
print('Maastro test pids:', maastro_test_pids)

print('Creating file...')
with h5py.File(output_data_path, 'w') as f:
    for i in range(4):
        f.create_group(f'fold_{i}')

# OUS train is in fold 0
with h5py.File(output_data_path, 'a') as f:
    f['fold_0'].create_dataset(
        'input', shape=(len(ous_train_pids), 176, 192, 256, 2), dtype='f4',
        chunks=(1, 176, 192, 256, 2), compression='lzf')
    f['fold_0'].create_dataset(
        'target', shape=(len(ous_train_pids), 176, 192, 256, 2), dtype='f4',
        chunks=(1, 176, 192, 256, 2), compression='lzf')
    f['fold_0'].create_dataset(
        'patient_idx', shape=(len(ous_train_pids),), dtype='i4',
        compression='lzf')

# maastro train is in fold 1&2
for i in range(1, 3):
    with h5py.File(output_data_path, 'a') as f:
        f[f'fold_{i}'].create_dataset(
            'input', shape=(50, 176, 192, 256, 2), dtype='f4',
            chunks=(1, 176, 192, 256, 2), compression='lzf')
        f[f'fold_{i}'].create_dataset(
            'target', shape=(50, 176, 192, 256, 2), dtype='f4',
            chunks=(1, 176, 192, 256, 2), compression='lzf')
        f[f'fold_{i}'].create_dataset(
            'patient_idx', shape=(50,), dtype='i4',
            compression='lzf')

print('Filling validation data...')
# validation data is in fold 3
with h5py.File(output_data_path, 'a') as f:
    f['fold_3'].create_dataset(
        'input', shape=(len(ous_test_pids) + len(maastro_test_pids), 176, 192, 256, 2), dtype='f4',
        chunks=(1, 176, 192, 256, 2), compression='lzf')
    f['fold_3'].create_dataset(
        'target', shape=(len(ous_test_pids) + len(maastro_test_pids), 176, 192, 256, 2), dtype='f4',
        chunks=(1, 176, 192, 256, 2), compression='lzf')
    f['fold_3'].create_dataset(
        'patient_idx', shape=(len(ous_test_pids) + len(maastro_test_pids),), dtype='i4',
        compression='lzf')

# filling validation data

with h5py.File(output_data_path, 'a') as f:
    with h5py.File(ous_data_path, 'r') as ous_f:
        f['fold_3']['input'][:len(ous_test_pids)] = ous_f['fold_1']['input'][:]
        f['fold_3']['target'][:len(ous_test_pids)] = ous_f['fold_1']['target'][:]
        f['fold_3']['patient_idx'][:len(ous_test_pids)] = ous_test_pids
    with h5py.File(maastro_data_path, 'r') as maastro_f:
        f['fold_3']['input'][len(ous_test_pids):] = maastro_f['fold_1']['input'][:]
        f['fold_3']['target'][len(ous_test_pids):] = maastro_f['fold_1']['target'][:]
        f['fold_3']['patient_idx'][len(ous_test_pids):] = maastro_test_pids

print('Validation data created')

print('Filling training data...')
for i in range(len(ous_train_pids)):
    with h5py.File(output_data_path, 'a') as f:
        with h5py.File(ous_data_path, 'r') as ous_f:
            f['fold_0']['input'][i] = ous_f['fold_0']['input'][i]
            f['fold_0']['target'][i] = ous_f['fold_0']['target'][i]
            f['fold_0']['patient_idx'][i] = ous_train_pids[i]
for i in range(50):
    with h5py.File(maastro_data_path, 'r') as maastro_f:
        with h5py.File(output_data_path, 'a') as f:
            f['fold_1']['input'][i] = maastro_f['fold_0']['input'][i]
            f['fold_1']['target'][i] = maastro_f['fold_0']['target'][i]
            f['fold_1']['patient_idx'][i] = maastro_train_pids[i]
            f['fold_2']['input'][i] = maastro_f['fold_0']['input'][i+50]
            f['fold_2']['target'][i] = maastro_f['fold_0']['target'][i+50]
            f['fold_2']['patient_idx'][i] = maastro_train_pids[i+50]
print('Training data created')

# checking the data
with h5py.File(output_data_path, 'r') as f:
    for i in range(6):
        print(f'Fold {i} input shape:', f['fold_{i}']['input'].shape)
        print(f'Fold {i} target shape:', f['fold_{i}']['target'].shape)
        print(f'Fold {i} patient_idx shape:', f['fold_{i}']['patient_idx'].shape)
        print(f'Fold {i} patient_idx:', f['fold_{i}']['patient_idx'][:])
