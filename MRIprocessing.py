import sys
import os
import argparse


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
    else:
        # not supported file type
        print("Input file type is not supported")
