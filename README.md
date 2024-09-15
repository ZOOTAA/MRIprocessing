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
   - pip install -r /path/to/requirements.txt
4. execute the script with command
   - python MRIprocessing.py -i /path/to/image_data.nii.gz -o /path/to/store_processed_data.nii.gz -p denoise

