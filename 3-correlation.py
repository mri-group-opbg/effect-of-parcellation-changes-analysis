# Default configuration
import nibabel

from configuration import *

# # Filtering

# ## Comp Corr

from nipype.algorithms import confounds


def compcorr_filter(signal_file, subject_id):
    ccinterface = confounds.TCompCor()
    ccinterface.inputs.realigned_file = signal_file
    ccinterface.inputs.components_file = f"{subjects_dir}/{subject_id}/components_file.txt"
    ccinterface.inputs.num_components = 5
    ccinterface.inputs.pre_filter = 'polynomial'
    ccinterface.inputs.regress_poly_degree = 2
    result = ccinterface.run()
    return genfromtxt(result.outputs.components_file, delimiter='\t', skip_header=1)

# result = compcorr_filter(f"{subjects_dir}/{subject_id}/func/rest.nii",
#                         f"{subjects_dir}/{subject_id}/mri/low.res.ribbon.mgz")

# # McFlirt filtering

from numpy import genfromtxt

def mcflirt_filter(subjects_dir, subject_id):
    return genfromtxt(f"{subjects_dir}/{subject_id}/func/rest.par")

# # Filtering

from numpy.linalg import inv

def Brain4D_RegressOutCovariables(oneAxialSlice, theCovariables):
    
    shape = oneAxialSlice.shape

    # (v, t)
    oneAxialSlice = np.reshape(oneAxialSlice, (shape[0] * shape[1] * shape[2], shape[3])).T
              
    result = np.dot(
        (
             np.eye(shape[3]) - 
             np.dot(
                 np.dot(
                     theCovariables, 
                     inv(np.dot(theCovariables.T, theCovariables))
                 ), 
                 theCovariables.T
             )
         ), 
         oneAxialSlice)

    return np.reshape(result.T, (shape[0], shape[1], shape[2], shape[3]))

import nibabel as nib
import scipy.signal as signal

def filtered_functional(subjects_dir, subject_id):

    # Load functional data
    data = nibabel.nifti1.load(f"{subjects_dir}/{subject_id}/func/rest.nii")
    
    functional_no_filter = data.get_fdata()
    
    #detrend filter
    functional_detrend = signal.detrend(functional_no_filter)
    
    # Functional data without polynomial noise
    mcflirt = mcflirt_filter(subjects_dir, subject_id)
    
    #print(f"McFlirt (Filter) shape: {mcflirt.shape}")

    compcorr = compcorr_filter(f"{subjects_dir}/{subject_id}/func/rest.nii", 
                               subject_id)
    #print(f"CompCorr (Filter) shape: {compcorr.shape}")
    final = np.concatenate((mcflirt, compcorr), axis=1)    
    #print(f"Final (Filter) shape: {final.shape}")
    filt_functional = Brain4D_RegressOutCovariables(functional_detrend, final)
    #print (filt_functional)
    return (filt_functional, data.header)


# # Processing
# https://nilearn.github.io/manipulating_images/manipulating_images.html#computing-and-applying-spatial-masks

"""
This method apply a mask file (a 0/1 volumetric matrix that define where a parcel is) 
to a functional file rest.nii.gz (4D: volume x time)
On the result matrix is computed the mean over the volume resulting in an array of mean value
for the parcel in time.
"""
def get_functional_volumetric_mean_value(functional, mask_file):

    # Apply volumetric mask for each timeframe: matrix(Time Frames, Voxels of the mask volume)
    # So the second dimension if the number of voxels that are 1 in the mask_file.
    # Different mask volume lead to different lenght in the second dimension.

    mask = np.array(nib.load(mask_file).dataobj)

    (x, y, z) = np.where(mask == 1)

    return functional[x, y, z, :].mean(axis=0)

import multiprocessing
from multiprocessing import Pool
import numpy as np

def get_functional_mean_values(functional, subjects_dir, subject_id, index):

    # Pool for parallel execution
    # p = Pool(multiprocessing.cpu_count())
    
    # Pattern file
    file_pattern = f"{subjects_dir}/{subject_id}/{DIR_VOLUMES}/{index}/filtered/*.*.nii.gz"

    # Matrix with the mean weight of voxels for parcel in a timeframe
    files = [f for f in glob.glob(file_pattern, recursive=False)]
    files.sort()
    
    # functional_and_mask_files = list(map(lambda x : (functional, x), files))

    results = []
    for mask_file in files:
        results.append(get_functional_volumetric_mean_value(functional, mask_file))

    return (
        [os.path.basename(f).replace(".nii.gz", "") for f in files], 
        np.array(results)
    )

    # result = (
    #     [os.path.basename(f).replace(".nii.gz", "") for f in files], 
    #    np.matrix(p.map(get_functional_volumetric_mean_value, functional_and_mask_files))
    #)
   
    # p.join() 
    # p.close()
    
    # return result 

# TODO make parallel
# TODO ripristinate funcional diltered

import pickle

def write_correlation_matrix(subjects_dir, subject_id, indexes):
    
    (filt_functional,header) = filtered_functional(subjects_dir, subject_id)

    os.makedirs(f"{subjects_dir}/{subject_id}/{DIR_CORRELATION}", exist_ok=True)

    for index in indexes:
        
        print(f"Working on index: {index}")
        # Label and matrix with labels x timeseries mean value of signals (one row for each parcel)
        (labels, matrix) = get_functional_mean_values(filt_functional, subjects_dir, subject_id, index)
    
        #print(matrix)
        #print(np.corrcoef(matrix))
        # Computation of correlation matrix
        correlation_matrix = np.corrcoef(matrix)
        #print(correlation_matrix)
        # print(correlation_matrix.shape)
       # print(f"Correlation matrix for {subjects_dir}/{subject_id}/{index} is {correlation_matrix.shape}")

    # Save
    # with open(f"{subjects_dir}/{subject_id}/{DIR_CORRELATION}/{index}.pickle", "wb") as f:
        # pickle.dump(correlation_matrix, f)
    #print(correlation_matrix)
    #print(f"Saved correlation matrix: {subjects_dir}/{subject_id}/{DIR_CORRELATION}/{index}.pickle")
    
    ##savefrancesca
    #import numpy
    #a = numpy.asarray(correlation_matrix)
    #numpy.savetxt('/data/comparison-schizofrenia/mount/recon_all/controllo/test33.csv', a, delimiter=",")
    # Save
        with open(f"{subjects_dir}/{subject_id}/{DIR_CORRELATION}/{index}.pickle", "wb") as f:
            pickle.dump(correlation_matrix, f)
            print(f"Saved correlation matrix: {subjects_dir}/{subject_id}/{DIR_CORRELATION}/{index}.pickle")
        
import glob
import os

def get_processing_labels(subject_id, index, hemi):
    return [label
            for label in [
                os.path.basename(f).replace(".label", "").replace(hemi + ".", "") 
                for f in glob.glob(f"{subjects_dir}/{subject_id}/{DIR_LABELS}/{index}/{hemi}.*", recursive=False)]]

def check_phase2(subject_id):
    error = False
    missing_labels = []
    missing_volumes = []
    for index in indexes:
#        if not os.path.exists(f"{subjects_dir}/{subject_id}/{DIR_CORRELATION}/{index}.pickle"):
#            error = True
#            break
        for hemi in hemis:
            labels = get_processing_labels(subject_id, index, hemi)
            if len(labels) == 0:
                missing_labels.append(f"Subject {subject_id} missing labels {index} {hemi}")
                error = True
            else:
                for label in labels:
                    path = f"{subjects_dir}/{subject_id}/{DIR_VOLUMES}/{index}/filtered/{hemi}.{label}.nii.gz"
                    if not os.path.exists(path):
                        missing_volumes.append(f"Subject {subject_id} missing volume {index} / {hemi} / {label}: {path}\n")
                        error = True

    if not error:
        print("Subject", subject_id, "is well formed")
        return True
    else:
        print("Missing labels:", len(missing_labels))
        if len(missing_labels) == 0:
            print("Missing volumes:", len(missing_volumes))
            print(missing_volumes)
        return False

if subject_id != "fsaverage" and check_phase2(subject_id):
    write_correlation_matrix(subjects_dir, subject_id, indexes)

