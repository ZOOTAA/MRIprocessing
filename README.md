# MRIprocessing

## Background
This is a script for MRI image processing. It loads the image data, then do the selected processing, finally save the processed image data.

### Supported image file type
- .nii / .nii.gz

### Supported procssing function
- Denoising with non local means
- Edge filtering with Sobel filter
- Gaussian blur with Gaussian filter
- Thresholding with adaptive thresholdinig
- Resample with interploation
- Intensity normalization with min-max normalization

### Script Dependence
The script is written with Python 3.10.9. Please refer to requirements.txt for required libraries.

## User Manual
The script is executed with command line input. The following steps help setup the Python environment.
1. clone the git repository
2. create a virtual environment of Python in the folder and activate it (Optional)
   - python -m venv .venv
   - .venv\Scripts\activate (Windows)
3. install all the packages in requirements.txt with pip
   - pip install -r \path\to\requirements.txt
4. execute the script with command, enter '-h' will show help message with information of supported input parameters and options
   - python MRIprocessing.py -h
   - python MRIprocessing.py -i \path\to\image_data.nii.gz -o \path\to\store_processed_data.nii.gz -p denoise

### Examples
The example image data is the anatomical scan from Subject 2 in the MEG-BIDS dataset (https://s3.amazonaws.com/openneuro.org/ds000247/sub-0002/ses-01/anat/sub-0002_ses-01_T1w.nii.gz?versionId=71.XAnuxtjw6ITyFLSPZeH_lAayTeyvq). The slices which plotted below are the 101 slice along the left-right axis, showing the posterior-anterior and inferior-superior axises.

### Gaussian blur
The command is as follow,  
python MRIprocessing.py -i \path\to\sub-0002_ses-01_T1w.nii.gz -o \path\to\processout.nii.gz -a 0 -p gaussianblur --sigma 2 --plot 1 --plotindex 100 --plotaxis 0  
'-a': process along slices on which axis index (default: 0)  
'-p': image processing to do  
'--sigma': sigma parameter of gaussian blur  
<img src="https://github.com/ZOOTAA/MRIprocessing/blob/main/processedImage/sub-0002_ses-01_T1w_gaussblur_sigma2.png" width="600" height="300">

### Thresholding
The command is as follow,  
python MRIprocessing.py -i \path\to\sub-0002_ses-01_T1w.nii.gz -o \path\to\processout.nii.gz -a 0 -p thresholding --blocksize 21 --plot 1 --plotindex 100 --plotaxis 0  
'-a': process along slices on which axis index (default: 0)  
'-p': image processing to do  
'--blocksize': odd number block size of adaptive thresholding  
<img src="https://github.com/ZOOTAA/MRIprocessing/blob/main/processedImage/sub-0002_ses-01_T1w_threshold_bsize21.png" width="600" height="300">

### Resample
The command is as follow,  
python MRIprocessing.py -i \path\to\sub-0002_ses-01_T1w.nii.gz -o \path\to\processout.nii.gz -a 0 -p resample --scalefactors 0.8 0.8 0.8 --plot 1 --plotindex 100 --plotaxis 0  
'-a': process along slices on which axis index (default: 0)  
'-p': image processing to do  
'--scalefactors': list of scale factors of axis 0, 1, 2 (default: [0.5, 0.5, 0.5])  
<img src="https://github.com/ZOOTAA/MRIprocessing/blob/main/processedImage/sub-0002_ses-01_T1w_resample_sfactor08.png" width="600" height="300">

### Denoise -> Edge filtering -> Intensity normalization
The command is as follow,  
python MRIprocessing.py -i \path\to\sub-0002_ses-01_T1w.nii.gz -o \path\to\processout.nii.gz -a 0 -p denoise edgefilter norm --plot 1 --plotindex 100 --plotaxis 0  
'-a': process along slices on which axis index (default: 0)  
'-p': image processing functions to do, follow user input order, different order may turn into different result
<img src="https://github.com/ZOOTAA/MRIprocessing/blob/main/processedImage/sub-0002_ses-01_T1w_denoise_edgefilter_norm.png" width="600" height="300">

#### Commands for plotting figure
'--plot': plotting orignal and processed slices figure (default: 0)  
'--plotindex': index of slice to be plotted  
'--plotaxis': plot slice with index on which axis 

