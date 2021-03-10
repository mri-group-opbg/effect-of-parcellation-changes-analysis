
import subprocess

# Default configuration
from configuration import *

class Fsl:
    
    def __init__(self, fsl_home, verbose=True):
        self.fsl_home = fsl_home
        self.verbose = verbose
                
    def run(self, cmdline):
        print("Executing: " + self.fsl_home + "/bin/" + cmdline)
        MyOut = subprocess.Popen(
            self.fsl_home + "/bin/" + cmdline, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            env={
                "FSLOUTPUTTYPE": "NIFTI"
            })
        stdout, stderr = MyOut.communicate()
        if self.verbose:
            if stdout:
                print(stdout.decode("utf-8"))
            if stderr:
                print(stderr.decode("utf-8"))

class FreeSurfer:
    
    def __init__(self, subjects_dir, freesurfer_home, verbose=True):
        self.freesurfer_home = freesurfer_home
        self.subjects_dir = subjects_dir
        self.verbose = verbose
        
    def set_verbose(self, verbose):
        self.verbose = verbose
        
    def run(self, cmdline):
        print("Executing: " + self.freesurfer_home + "/bin/" + cmdline)
        MyOut = subprocess.Popen(
            self.freesurfer_home + "/bin/" + cmdline, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            env={
                "SUBJECTS_DIR": self.subjects_dir,
                "FREESURFER_HOME": self.freesurfer_home,
                "LD_LIBRARY_PATH": f"{self.freesurfer_home}/lib/tcltktixblt/lib:/usr/local/fsl/fslpython/envs/fslpython/lib/" #tkregister2
            })
        stdout, stderr = MyOut.communicate()
        if self.verbose:
            if stdout:
                print(stdout.decode("utf-8"))
            if stderr:
                print(stderr.decode("utf-8"))

from nipype.interfaces.freesurfer import SurfaceTransform

def surface_transform_avg_to_subject(launcher, hemisphere, subject_id, source_annot, target_annot):
    sxfm = SurfaceTransform()

    sxfm.inputs.hemi = hemisphere                     # --hemi
            
    # --tval
    sxfm.inputs.out_file = target_annot
    sxfm.inputs.source_subject = "fsaverage"          # --srcsubject
    sxfm.inputs.target_subject = subject_id           # --trgsubject
    sxfm.inputs.source_annot_file = source_annot      # subjects_dir + "/fsaverage/label/lh.aparc.a2005s.annot" # --sval-annot

    # sxfm.cmdline
    # mri_surf2surf --hemi lh --tval lh.pippo.aparc.annot --sval-annot ../data/subjects/fsaverage/label/lh.aparc.a2005s.annot --srcsubject fsaverage --trgsubject bert

    launcher.run(sxfm.cmdline)

import os

def annotation_to_label(launcher, subjects_dir, hemisphere, subject_id, index, source_annot):
    
    os.makedirs(f"{subjects_dir}/{subject_id}/{DIR_LABELS}/{index}", exist_ok=True)
    
    base_name = os.path.basename(source_annot)
    annotation = base_name.replace('.annot', '')
    annotation = annotation.replace(".lh", "").replace(".rh", "")
    
    command = "mri_annotation2label"
    command += f" --annotation {source_annot}"
    command += f" --subject {subject_id}"
    command += f" --hemi {hemisphere}"
    command += f" --outdir {subjects_dir}/{subject_id}/{DIR_LABELS}/{index}"
    
    launcher.run(command)

# Launched on subject data once at begin
# Time frame will be probably all the same, we will use only the first splitted frame (see next method)

from nipype.interfaces import fsl as flsi

def mcflirt(launcher, in_file, out_file):
    mcflt = flsi.MCFLIRT()
    mcflt.inputs.in_file = in_file          # Low Space Resolution, High time resolution 64x64x32x150
    mcflt.inputs.cost = 'mutualinfo'
    mcflt.inputs.out_file = out_file
    mcflt.inputs.save_plots = True
    mcflt.cmdline

    # 'mcflirt -in rest.nii.gz -plot -cost mutualinfo -out rest'
    # >>> res = mcflt.run()  # doctest: +SKIP
    launcher.run(mcflt.cmdline)

from nipype.interfaces import fsl as flsi

def splitting(launcher, in_file, base_name):
    split = flsi.Split()
    split.inputs.in_file = in_file
    split.inputs.out_base_name = base_name
    split.inputs.dimension = 't'
    split.inputs.environ = {
        'FSLOUTPUTTYPE': 'NIFTI'
    }

    # 'mcflirt -in functional.nii -cost mutualinfo -out moco.nii'
    # >>> res = mcflt.run()  # doctest: +SKIP
    launcher.run(split.cmdline)

from nipype.interfaces.freesurfer import MRIConvert

def mri_convert(launcher, in_file, out_file):
    mc = MRIConvert()
    mc.inputs.in_file = in_file
    mc.inputs.out_file = out_file
    mc.inputs.out_type = 'mgz'
    launcher.run(mc.cmdline.replace("mri_convert", "mri_convert.bin"))

    # 'mri_convert --out_type mgz --input_volume structural.nii --output_volume outfile.mgz'

from nipype.interfaces.freesurfer import MRIConvert

def mri_convert_with_reslice(launcher, in_file, out_file, slice_file):
    mc = MRIConvert()
    mc.inputs.in_file = in_file
    mc.inputs.out_file = out_file
    mc.inputs.out_type = 'mgz'
    mc.inputs.reslice_like = slice_file
    mc.inputs.resample_type = 'nearest'

    launcher.run(mc.cmdline.replace("mri_convert", "mri_convert.bin"))

    # 'mri_convert --out_type mgz --input_volume structural.nii --output_volume outfile.mgz'

from nipype.interfaces.freesurfer import VolumeMask

def volume_mask(launcher, subject_id, lh_pial, rh_pial, lh_white, rh_white):
    volmask = VolumeMask()
    volmask.inputs.left_whitelabel = 2
    volmask.inputs.left_ribbonlabel = 3
    volmask.inputs.right_whitelabel = 41
    volmask.inputs.right_ribbonlabel = 42
    volmask.inputs.lh_pial = lh_pial
    volmask.inputs.rh_pial = rh_pial
    volmask.inputs.lh_white = lh_white
    volmask.inputs.rh_white = rh_white
    volmask.inputs.subject_id = subject_id
    # volmask.inputs.subjects_dir = subjects_dir
    volmask.inputs.save_ribbon = True

    launcher.run(volmask.cmdline.replace("mris_volmask", "mris_volmask_novtk"))


# In[ ]:


from nipype.interfaces.freesurfer import Tkregister2

def tkregister2(launcher, subject_id, reg_file, moving_image, target_file):
    command = "../tktools/tkregister2.bin"
    command += " --mov " + moving_image
    command += " --noedit"
    command += f" --s {subject_id}"
    command += " --reg " + reg_file
    command += " --regheader"
    command += " --targ " + target_file
    
    launcher.run(command)

    # 'tkregister2 --mov T1.mgz --noedit --reg T1_to_native.dat --regheader --targ structural.nii'
    # tk2.run() 


# In[ ]:


import nibabel as nib

def check_mgz(mgz_file):
    # Show the size of the brain resolution
    img = nib.load(mgz_file)
    print("Shape", img.shape)
    print("Data Type:", img.get_data_dtype())

import seaborn as sns
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt

def plot_mgz(mgz_file):
    sns.set(color_codes=True)
    plotting.plot_img(mgz_file)
    img = nib.load(mgz_file)
    p = np.array(img.dataobj).flatten()
    sns.distplot(p[p != 0])

from nipype.interfaces.freesurfer import Label2Vol

def label_to_volume(launcher, subject_id, hemi, label_file, template_file, reg_file, volume_label_file):
    binvol = Label2Vol(
        label_file=label_file, 
        template_file=template_file, 
        reg_file=reg_file, 
        fill_thresh=0.5, 
        subject_id=subject_id,
        hemi=hemi,
        # surf="white",
        proj=("frac", 0, 1, 0.01),
        vol_label_file=volume_label_file)
    launcher.run(binvol.cmdline)
    
# 'mri_label2vol --fillthresh 0 --label cortex.label --reg register.dat --temp structural.nii --o foo_out.nii'

from nipype.interfaces.freesurfer import Binarize

def binarize(launcher, in_file, binary_file):
    binvol = Binarize()
    binvol.inputs.in_file=in_file
    binvol.inputs.min=1
    binvol.inputs.binary_file=binary_file
    binvol.inputs.dilate=1
    binvol.inputs.erode=1

    launcher.run(binvol.cmdline.replace("mri_binarize", "mri_binarize.bin"))
    # 'mri_binarize --o foo_out.nii --i structural.nii --min 10.000000'

from nipype.interfaces.freesurfer import MRIsCalc

def mri_calc(launcher, label_volume_file, ribbon_volume_file, output_file):
    example = MRIsCalc()
    example.inputs.in_file1 = label_volume_file
    example.inputs.in_file2 = ribbon_volume_file
    example.inputs.action = 'mul'
    example.inputs.out_file = output_file

    launcher.run(example.cmdline)

import nibabel as nib

def assert_resolution(image_path, resolution):
    image = np.array(nib.load(image_path).dataobj)
    image_file = os.path.basename(image_path)
    assert (image.shape == resolution), f"We espect that {image_file} is of shape {resolution} but is {image.shape}"

# # Main software
freeSurfer = FreeSurfer(subjects_dir, "/usr/local/freesurfer", verbose=True)
fsl = Fsl("/usr/local/fsl", verbose=False)

# ## Preparation (register)
# 
# This section will prepare data for a set of subject defined in the subjects variable

# input: {subjects_dir}/0040013/func/rest.nii.gz
# output: {subjects_dir}/0040013/func/motion.correction.rest.nii

os.makedirs(f"{subjects_dir}/{subject_id}/func", exist_ok=True)

assert_resolution(f"{dataset_dir}/{subject_id}/session_1/rest_1/rest.nii.gz", resolution)

mcflirt(fsl, 
	f"{dataset_dir}/{subject_id}/session_1/rest_1/rest.nii.gz",
	f"{subjects_dir}/{subject_id}/func/rest")

assert_resolution(f"{subjects_dir}/{subject_id}/func/rest.nii", resolution)

os.makedirs(f"{subjects_dir}/{subject_id}/{DIR_SPLITTED}", exist_ok=True)

# input: data/subjects/0040013/func/motion.correction.rest.nii
# output: data/subjects/0040013/func/splitted/splitted_<T>.nii
# T = 0 to 150 time frame
splitting(fsl, 
          f"{subjects_dir}/{subject_id}/func/rest.nii", 
          f"{subjects_dir}/{subject_id}/{DIR_SPLITTED}/splitted_")


assert_resolution(f'{subjects_dir}/{subject_id}/{DIR_SPLITTED}/splitted_0000.nii', resolution[:3])

mri_convert(freeSurfer, 
            f'{subjects_dir}/{subject_id}/{DIR_SPLITTED}/splitted_0000.nii',
            f'{subjects_dir}/{subject_id}/{DIR_SPLITTED}/splitted_0000.mgz')

os.makedirs(f"{subjects_dir}/{subject_id}/{DIR_REGISTER}", exist_ok=True)

assert_resolution(f'{subjects_dir}/{subject_id}/mri/T1.mgz', (256, 256, 256))

tkregister2(freeSurfer, 
            subject_id,
            f'{subjects_dir}/{subject_id}/{DIR_REGISTER}/register-functional.dat',
            f'{subjects_dir}/{subject_id}/{DIR_SPLITTED}/splitted_0000.mgz',
            f'{subjects_dir}/{subject_id}/mri/T1.mgz')

# ## Preparation (ribbon)

aseg = np.array(nib.load(f"{subjects_dir}/{subject_id}/mri/aseg.mgz").dataobj)

print("aseg.mgz is:", aseg.shape)
# print("orig.mgz is:", np.array(nib.load(f"{subjects_dir}/{subject_id}/mri/orig.aseg.mgz").dataobj).shape)
# print("low.resolution.aseg.mgz is:", np.array(nib.load(f"{subjects_dir}/{subject_id}/mri/low.resolution.aseg.mgz").dataobj).shape)

os.rename(f"{subjects_dir}/{subject_id}/mri/aseg.mgz", f"{subjects_dir}/{subject_id}/mri/orig.aseg.mgz")

assert_resolution(f"{subjects_dir}/{subject_id}/mri/orig.aseg.mgz", (256, 256, 256))

# input: data/subjects/0040013/mri/brain.mgz
# input.slice: data/subjects/0040013/func/splitted/splitted_0000.mgz
# output: data/subjects/0040013/mri/low.resolution.brain.mgz
    
mri_convert_with_reslice(freeSurfer,
                        f'{subjects_dir}/{subject_id}/mri/orig.aseg.mgz', # 256 x 256 x 256
                        f'{subjects_dir}/{subject_id}/mri/low.resolution.aseg.mgz', # 64 x 64 x 33
                        f'{subjects_dir}/{subject_id}/{DIR_SPLITTED}/splitted_0000.mgz') # 64 x 64 x 33

assert_resolution(f"{subjects_dir}/{subject_id}/mri/orig.aseg.mgz", (256, 256, 256))
assert_resolution(f"{subjects_dir}/{subject_id}/mri/low.resolution.aseg.mgz", resolution[:3])
assert_resolution(f"{subjects_dir}/{subject_id}/{DIR_SPLITTED}/splitted_0000.mgz", resolution[:3])

from shutil import copyfile

assert_resolution(f"{subjects_dir}/{subject_id}/mri/low.resolution.aseg.mgz", resolution[:3])

copyfile(f"{subjects_dir}/{subject_id}/mri/low.resolution.aseg.mgz", f"{subjects_dir}/{subject_id}/mri/aseg.mgz")

# input: [hemi]_pial, [hemi]_white
# output: 
#    data/subjects/0040013/mri/lh.ribbon.mgz
#    data/subjects/0040013/mri/rh.ribbon.mgz
#    data/subjects/0040013/mri/ribbon.mgz
volume_mask(freeSurfer, 
            subject_id,
            f"{subjects_dir}/{subject_id}/surf/lh.pial",
            f"{subjects_dir}/{subject_id}/surf/rh.pial",
            f"{subjects_dir}/{subject_id}/surf/lh.white",
            f"{subjects_dir}/{subject_id}/surf/rh.white")

assert_resolution(f"{subjects_dir}/{subject_id}/mri/ribbon.mgz", resolution[:3])
assert_resolution(f"{subjects_dir}/{subject_id}/mri/lh.ribbon.mgz", resolution[:3])
assert_resolution(f"{subjects_dir}/{subject_id}/mri/rh.ribbon.mgz", resolution[:3])

assert_resolution(f"{subjects_dir}/{subject_id}/mri/orig.aseg.mgz", (256, 256, 256))

copyfile(f"{subjects_dir}/{subject_id}/mri/orig.aseg.mgz", f"{subjects_dir}/{subject_id}/mri/aseg.mgz")

assert_resolution(f"{subjects_dir}/{subject_id}/mri/ribbon.mgz", resolution[:3])
assert_resolution(f"{subjects_dir}/{subject_id}/mri/lh.ribbon.mgz", resolution[:3])
assert_resolution(f"{subjects_dir}/{subject_id}/mri/rh.ribbon.mgz", resolution[:3])

os.rename(f"{subjects_dir}/{subject_id}/mri/ribbon.mgz", f"{subjects_dir}/{subject_id}/mri/low.res.ribbon.mgz")
os.rename(f"{subjects_dir}/{subject_id}/mri/lh.ribbon.mgz", f"{subjects_dir}/{subject_id}/mri/low.res.lh.ribbon.mgz")
os.rename(f"{subjects_dir}/{subject_id}/mri/rh.ribbon.mgz", f"{subjects_dir}/{subject_id}/mri/low.res.rh.ribbon.mgz")

# # Parcels processing

# All volumetric parcels are maintained in the directory data/parcels with the following naming convention:
#     
# [subject].[index].[hemi].[label].nii.gz
# 
# Where [index] is "original" or a number for the trial set of a subjects

# ## Volume processing

# input: parcels/lh.parcels.0.annot
# output: data/subjects/0040013/label/parcels/lh.parcels.0.annot

os.makedirs(f"{subjects_dir}/{subject_id}/{DIR_PARCELS}", exist_ok=True)

for index in indexes:
    
    for hemi in hemis:

        if index == 'origin':
            annot_file = f"{fsaverage_dir}/label/{hemi}.{atlas}.annot"
        else:
            annot_file = f"{subjects_dir}/{subject_id}/{DIR_RANDOM_PARCELS}/{index}.{hemi}.annot"

        surface_transform_avg_to_subject(
            freeSurfer, 
            hemi, 
            subject_id,
            annot_file,
            f"{subjects_dir}/{subject_id}/{DIR_PARCELS}/{index}.{hemi}.annot")


for index in indexes:

    for hemi in hemis:

        # This will produce <subject_id>/labels
        annotation_to_label(
            freeSurfer, 
            subjects_dir,
            hemi, 
            subject_id,
            index,
            f"{subjects_dir}/{subject_id}/{DIR_PARCELS}/{index}.{hemi}.annot")

"""
Produces a volume starting from a label:
it invokes the methods:
- label_to_volume: to map surface into volume
- binarize: to binarize the volume
- mri_calc against ribbon: to reduce the volume to the ribbon volume
"""

def process_label(item):

    (subject_id, index, hemi, label) = item
        
    print(f"* processing: {subject_id}, {index}, {hemi}, {label}")
    
    # {volumes_dir}/{subject_id}/{index}/...
    os.makedirs(f"{subjects_dir}/{subject_id}/{DIR_VOLUMES}/{index}", exist_ok=True)
    os.makedirs(f"{subjects_dir}/{subject_id}/{DIR_VOLUMES}/{index}/origin", exist_ok=True)
    os.makedirs(f"{subjects_dir}/{subject_id}/{DIR_VOLUMES}/{index}/binarized", exist_ok=True)
    os.makedirs(f"{subjects_dir}/{subject_id}/{DIR_VOLUMES}/{index}/filtered", exist_ok=True)
    
    label_to_volume(
        freeSurfer, 
        subject_id,
        hemi,
        f"{subjects_dir}/{subject_id}/{DIR_LABELS}/{index}/{hemi}.{label}.label",
        f"{subjects_dir}/{subject_id}/{DIR_SPLITTED}/splitted_0000.mgz",
        f"{subjects_dir}/{subject_id}/{DIR_REGISTER}/register-functional.dat",
        f"{subjects_dir}/{subject_id}/{DIR_VOLUMES}/{index}/origin/{hemi}.{label}.nii.gz")
    
    binarize(
        freeSurfer, 
        f"{subjects_dir}/{subject_id}/{DIR_VOLUMES}/{index}/origin/{hemi}.{label}.nii.gz",
        f"{subjects_dir}/{subject_id}/{DIR_VOLUMES}/{index}/binarized/{hemi}.{label}.nii.gz")
    
    mri_calc(freeSurfer,
        f"{subjects_dir}/{subject_id}/{DIR_VOLUMES}/{index}/binarized/{hemi}.{label}.nii.gz",
        f"{subjects_dir}/{subject_id}/mri/low.res.{hemi}.ribbon.mgz", 
        f"{subjects_dir}/{subject_id}/{DIR_VOLUMES}/{index}/filtered/{hemi}.{label}.nii.gz")

    return 0

import os
import glob

"""
Return the list of items to process given a set of subjects, indexes, hemispheres
"""
def get_processing_labels(subjects, indexes, hemi):
    return [(subject_id, index, hemi, label) 
            for index in indexes 
            for hemi in hemis 
            for label in [
                os.path.basename(f).replace(".label", "").replace(hemi + ".", "") 
                for f in glob.glob(f"{subjects_dir}/{subject_id}/{DIR_LABELS}/{index}/{hemi}.*", recursive=False)]]

for item in get_processing_labels([subject_id], indexes, hemis):
    process_label(item)

# import multiprocessing
# from multiprocessing import Pool, Value

# processing_pool = Pool(processes=multiprocessing.cpu_count())
# processing_pool.map_async(process_label, get_processing_labels([subject_id], indexes, hemis))

# processing_pool.close()
# processing_pool.join()

print("Bye")

# import sys
# sys.exit()

