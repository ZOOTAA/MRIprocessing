import sys
import os
import argparse
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
import matplotlib as plt

if __name__ == '__main__':
    # Command line parsing
    parser = argparse.ArgumentParser(description='MRI image processing')
    parser.add_argument('-i', '--input', type=str, help='file path of input data')
    parser.add_argument('-o', '--output', type=str, help='file path of output data')
    args = parser.parse_args()
    print(args.input, args.output, os.path.dirname(args.output))

    # Check arguments
    if not os.path.exists(args.input):
        # input file path not exist
        print("Input data file path not exist")
        sys.exit()
    elif not os.path.exists(os.path.dirname(args.output)):
        # output file directory not exist
        print("Ouput file directory not exist")
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

