import sys
import os
import argparse
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means,  estimate_sigma
from skimage import filters
import cv2

def denoiseSlice(in_slice, param):
    """
    Denoise image with scikit image Non Local Means (NLM)
    Input:
        in_slice: 2d array, input image
        param: argparse arg
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

def edgefiltering(in_slice, param):
    """
    Edge filtering with scikit image Sobel filter
    Input:
        in_slice: 2d array, input image
        param: argparse arg
    Output:
        edgedetect_img: 2d array
    """
    sobel_image = filters.sobel(in_slice)
    return sobel_image

def gaussianblur(in_slice, param):
    """
    Gaussian Blur with scikit image Gaussian filter
    Input:
        in_slice: 2d array, input image
        param: argparse arg
    Output:
        blurred_image: 2d array
    """
    blurred_image = filters.gaussian(in_slice, sigma=param.sigma)  # sigma defines the standard deviation
    return blurred_image

def thresholding(in_slice, param):
    """
    Thresholding of image with opencv adaptive thresholding
    Input:
        in_slice: 2d array, input image
        param: argparse arg
    Output:
        binary_image: 2d array
    """
    # Normalize the image to the 0-255 range
    normalized_image = (in_slice - param.imgmin) / (param.imgmax - param.imgmin) * 255
    # Convert to unsigned 8-bit (uint8)
    image_uint8 = normalized_image.astype(np.uint8)
    binary_image = cv2.adaptiveThreshold(image_uint8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, param.blocksize, 2)
    return binary_image

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
    parser.add_argument(
        '-p',
        '--process',
        nargs='+',
        type=str,
        help='Image processing to do in order, support denoise, edgefilter, gaussianblur'
    )
    parser.add_argument('--sigma', type=float, default=1, help='sigma parameter of gaussian blur')
    parser.add_argument('--blocksize', type=int, default=11, help='odd number block size of adaptive thresholding')
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
    
    # Processing functions
    profunc_dict = {
        'denoise': denoiseSlice,
        'edgefilter': edgefiltering,
        'gaussianblur': gaussianblur,
        'thresholding': thresholding
    }
    profunc = []
    if args.process is not None:
        for pf in args.process:
            profunc.append(profunc_dict[pf])
    
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

    # max, min of image
    image_min = np.min(img_data)
    image_max = np.max(img_data)
    # add to args
    args.imgmin = image_min
    args.imgmax = image_max
    print('\timage minimum:', image_min, ", image maximum:", image_max)
    
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
        for pfunc in profunc:
            ppslice = pfunc(ppslice, args)
        # Append output
        img_data[ss, :, :] = ppslice
    print("Processing complete")

    # Unshift axis of axis 0 and target axis
    img_data = np.swapaxes(img_data, 0, args.axis)

    # Plot
    final_s0 = img_data[psidx, :, :]
    final_s0fig = plotSlice(final_s0)
    plt.show()
