import sys
import os
import argparse
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means,  estimate_sigma
from skimage import filters

def denoiseSlice(in_slice):
    """
    Denoise image with scikit image Non Local Means (NLM)
    Input:
        in_slice: 2d array, input image
    Output:
        denoise_img: 2d array
    """
    if np.mean(in_slice) == 0:
        # RuntimeWarning: Mean of empty slice
        denoise_img = in_slice
    else:
        sigma_est = np.mean(estimate_sigma(in_slice))
        denoise_img = denoise_nl_means(
            in_slice,
            h = 1.15*sigma_est,
            fast_mode = True,
            patch_size = 9,
            patch_distance = 5
        )
    return denoise_img

def edgefiltering(in_slice):
    """
    Edge filtering with scikit image Sobel filter
    Input:
        in_slice: 2d array, input image
    Output:
        edgedetect_img: 2d array
    """
    sobel_image = filters.sobel(in_slice)
    return sobel_image

def gaussianblur(in_slice, in_sigma=1):
    """
    
    """
    # Apply Gaussian Blur
    blurred_image = filters.gaussian(in_slice, sigma=in_sigma)  # sigma defines the standard deviation
    pass

def plotSlice(in_slice, fsize=(5,5)):
    """
    Plot 2D array
    Input:
        in_slice: 2d array, input image
        fsize: (width, height), set figure size
    Output:
        fig: matplotlib figure
    """
    fig = plt.figure(figsize=fsize)
    plt.imshow(in_slice, cmap='gray')
    return fig

if __name__ == '__main__':
    # Command line parsing
    parser = argparse.ArgumentParser(description='MRI image processing')
    parser.add_argument('-i', '--input', type=str, help='file path of input data')
    parser.add_argument('-o', '--output', type=str, help='file path of output data')
    parser.add_argument('-a', '--axis', type=int, default=0, help='file path of output data')
    args = parser.parse_args()
    print(args)

    # Check arguments
    if args.input is None:
        print('Error: Please give input data file path with \"-i\" or \"--input\"')
        sys.exit()
    elif args.output is None:
        print('Error: Please give output data file path with \"-o\" or \"--output\"')
        sys.exit()
    elif not os.path.exists(args.input):
        # input file path not exist
        print("Error: Input data file path not exist")
        sys.exit()
    elif not os.path.exists(os.path.dirname(args.output)):
        # output file directory not exist
        print("Error: Ouput file directory not exist")
        sys.exit()
    
    # Load input data
    fext = args.input.split('.')[-2:] # get last two extension
    if fext[1] == "nii" or (fext[1] == 'gz' and fext[0] == 'nii'):
        # .nii medical image file
        print("Load .nii medical image file")
        img_nib = nib.load(args.input)
        # array info
        img_data = img_nib.get_fdata()
        print('\timage shape:', img_nib.shape)
        print('\tdata shape:', img_data.shape)
        print('\tdata type:', type(img_data))
        # volume info
        zooms = img_nib.header.get_zooms() # spacing between voxel
        print('\tzooms of the voxel:', zooms)
        # R: right, L: left, A: anterior, P: posteria, I: inferior, S: superior
        axs_code = nio.ornt2axcodes(nio.io_orientation(img_nib.affine))
        print('\timage orientation code:', axs_code) # axis positive direction towards
        # global info
        # print('\taffine matrix:', img_nib.affine)
    else:
        # not supported file type
        print("Input file type is not supported")
        sys.exit()

    # Plot slices
    psidx = 100
    slice0 = img_data[psidx, :, :]
    s0fig = plotSlice(slice0)
    
    # Shift axis for process image along axis 0
    if args.axis > len(img_data.shape)-1:
        print('Error: Selected axis index is out of input data dimension')
        sys.exit()
    img_data = np.swapaxes(img_data, 0, args.axis)

    # Loop through each slice
    print("Start processing ......")
    for ss in range(img_data.shape[0]):
        ppslice = img_data[ss, :, :]        
        # Image processing
        # Denoise
        ppslice = denoiseSlice(ppslice)
        # Edge detection
        ppslice = edgefiltering(ppslice)
        # Append output
        img_data[ss, :, :] = ppslice
    print("Processing complete")

    # Unshift axis of axis 0 and target axis
    img_data = np.swapaxes(img_data, 0, args.axis)

    # Plot
    final_s0 = img_data[psidx, :, :]
    final_s0fig = plotSlice(final_s0)
    plt.show()
