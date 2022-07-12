import nibabel as nib
import os
from tqdm import tqdm
subject = 'F127'

anat = f'/mnt/HDD12TB/freesurfer/subjects/{subject}/mri/orig/anatomical.nii'
outDir = f'{os.path.dirname(anat)}/slicePNGs'
os.makedirs(outDir, exist_ok=True)
anatShape = nib.load(anat).get_fdata().shape
print(f'subject ID: {subject}')
print(f'getting sagittal slices')
for x in tqdm(range(anatShape[0])):
    os.system(f'slicer {anat} -x -{x} {outDir}/X_{x}.png')

print(f'getting coronal slices')
for y in tqdm(range(anatShape[1])):
    os.system(f'slicer {anat} -y -{y} {outDir}/Y_{y}.png')

print(f'getting axial slices')
for z in tqdm(range(anatShape[2])):
    os.system(f'slicer {anat} -z -{z} {outDir}/Z_{z}.png')

