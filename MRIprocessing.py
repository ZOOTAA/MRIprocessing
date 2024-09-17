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
from scipy.ndimage import zoom

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
            patch_size = param.denoisesize, # 9
            patch_distance = param.denoisedist # 5
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
    # Convert back to float32
    binary_image = binary_image.astype(np.float32)
    return binary_image

def resampleSlice(in_tensor, param):
    """
    Resample / rescaling image with scipy.ndimage zoom
    Input:
        in_tensor: 3d tensor, input image
        param: argparse arg
    Output:
        zoom_tensor: 3d tensor
    """
    # Apply zoom with interpolation (e.g., order=3 for cubic interpolation)
    zoom_tensor = zoom(in_tensor, param.scalefactors, order=3)
    return zoom_tensor

def intensitynorm(in_tensor, param):
    """
    Intensity Normalisation with Min-Max Normalization to 1
    Resample / rescaling image with scipy.ndimage zoom
    Input:
        in_tensor: 3d tensor, input image
        param: argparse arg
    Output:
        norm_tensor: 3d tensor
    """
    tensor_min = np.min(in_tensor)
    tensor_max = np.max(in_tensor)
    norm_tensor = (in_tensor - tensor_min) / (tensor_max - tensor_min)
    return norm_tensor

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

def plotOriProSlices(ori_data, pro_data, paxis, pindex, data_orientation, resamfact=1, fsize=(10,5)):
    """
    Plot original slice and processed slice for comparison
    Input:
        ori_data: 3d tensor, orignal data
        pro_data: 3d tensor, processed data
        paxis: int, plot slice with index on which axis
        pindex: int, index of slice to be plotted
        data_orientation: list of str, data orientation to plot image x, y axis
        resamfact: float, resample factor of processed data, default is 1 means no resample
        fsize: (width, height), set figure size
    Output:
        fig: matplotlib figure
    """
    # fix pindex range
    pindex = min(max(pindex, 0), img_data.shape[paxis]-1)
    # arrange plot axis to axis 0
    ori_data = np.moveaxis(ori_data, paxis, 0)
    pro_data = np.moveaxis(pro_data, paxis, 0)
    # get slice
    pslice = []
    pslice.append(ori_data[pindex, :, :])
    pslice.append(pro_data[int(pindex*resamfact), :, :])
    # adjust x, y axis
    data_orientation = list(data_orientation) # tuple to list
    del data_orientation[paxis]
    flip = 0 
    if 'S' in data_orientation and data_orientation[0] != 'S': # case with 'S'
        flip = 1
    elif data_orientation[1] != 'R': # 'R', 'A' case
        flip = 1
    if flip:
        data_orientation = data_orientation[::-1] # reverse list
        for pidx in range(len(pslice)):
            pslice[pidx] = pslice[pidx].transpose()
    # plot slices
    ptitle = ['Original', 'Processed']
    fig, axs = plt.subplots(1, len(pslice), figsize=fsize)
    for pidx in range(len(pslice)):
        axs[pidx].imshow(pslice[pidx], cmap='gray', origin='lower')
        axs[pidx].set_title(ptitle[pidx])
        axs[pidx].set_xlabel(data_orientation[1])
    axs[0].set_ylabel(data_orientation[0])
    return fig

if __name__ == '__main__':
    # Command line parsing
    parser = argparse.ArgumentParser(description='MRI image processing')
    parser.add_argument('-i', '--input', type=str, help='file path of input data')
    parser.add_argument('-o', '--output', type=str, help='file path of output data')
    parser.add_argument('-a', '--axis', type=int, default=0, help='process along slices on which axis index (default: 0)')
    parser.add_argument(
        '-p',
        '--process',
        nargs='+',
        type=str,
        help='image processing to do, support denoise, edgefilter, gaussianblur, thresholding, resample, norm. \
        Resample is always the second last process and normalization is always the last process. \
        Others follow user input order.'
    )
    parser.add_argument('--denoisesize', type=int, default=9, help='patch size of non local mean denoising (default: 9)')
    parser.add_argument('--denoisedist', type=int, default=5, help='patch distance of non local mean denoising  (default: 5)')
    parser.add_argument('--sigma', type=float, default=1, help='sigma parameter of gaussian blur')
    parser.add_argument('--blocksize', type=int, default=11, help='odd number block size of adaptive thresholding')
    parser.add_argument('--scalefactors', type=float, nargs=3, default=[0.5, 0.5, 0.5],
                        help='list of scale factors of axis 0, 1, 2 (default: [0.5, 0.5, 0.5])')
    parser.add_argument('--plot', type=bool, default=0, help='switch for plotting orignal and processed slices (default: 0)')
    parser.add_argument('--plotaxis', type=int, default=0, help='plot slice with index on which axis (default: 0)')
    parser.add_argument('--plotindex', type=int, default=0, help='index of slice to be plotted (default: 0)')
    args = parser.parse_args()
    # print(args)

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
        # neglect resample, norm
        pflist = args.process.copy()
        if 'resample' in pflist:
            pflist.remove('resample')
        if 'norm' in pflist:
            pflist.remove('norm')
        for pf in pflist:
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
        print('\tdata type:', type(img_data), type(img_data[0,0,0]))
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
    
    # copy initial image for plotting if require
    if args.plot:
        initial_img = img_data.copy()

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
    
    # Unshift axis of axis 0 and target axis
    img_data = np.swapaxes(img_data, 0, args.axis)
    # always put resample size at second last operation after unshift
    plot_resamfact = 1 # scale factor for plotting
    if 'resample' in args.process:
        img_data = resampleSlice(img_data, args)
        plot_resamfact = args.scalefactors[args.plotaxis]
    # always put normalization at last operation
    if 'norm' in args.process:
        img_data = intensitynorm(img_data, args)
    print("Processing complete")

    # Create a new NIfTI image with the modified data and the original affine
    new_img = nib.Nifti1Image(img_data, img_nib.affine, img_nib.header)
    # Save processed data
    nib.save(new_img, args.output)
    print('Data is saved to', args.output)

    # plotting orignal and processed slices
    if args.plot:
        fig = plotOriProSlices(initial_img, img_data, args.plotaxis, args.plotindex, axs_code, resamfact=plot_resamfact)
        plt.show()
