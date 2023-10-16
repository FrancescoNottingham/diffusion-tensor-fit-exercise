import numpy as np
import nibabel as nib
import diffusion_fitting_functions as dff

bvals = np.loadtxt('/home/lpxfd2/Documents/diffusion-tensor-fitting-task/task-data/bvals')
bvecs = np.loadtxt('/home/lpxfd2/Documents/diffusion-tensor-fitting-task/task-data/bvecs')

diffusion_data_img = nib.load('/home/lpxfd2/Documents/diffusion-tensor-fitting-task/task-data/raw-data.nii.gz')
brain_mask_img = nib.load('/home/lpxfd2/Documents/diffusion-tensor-fitting-task/task-data/brain-volume-0_mask.nii.gz')

brain_mask = brain_mask_img.get_fdata()
diffusion_data = diffusion_data_img.get_fdata()

fractional_anisotropy = dff.diffusion_tensor_fit(diffusion_data,brain_mask,bvals,bvecs)

empty_header = nib.Nifti1Header()

nifti_FA = nib.Nifti1Image(fractional_anisotropy, diffusion_data_img.affine, empty_header)

nib.save(nifti_FA, 'FA_result.nii.gz')