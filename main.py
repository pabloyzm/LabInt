from SIFT.SIFT_gen_and_utils import SIFTFeatures
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_directory', type=str)
    parser.add_argument('--n_descriptors', default = 100, type=int)
    parser.add_argument('--subsample_ratio',default = 1, type=float)

    args = parser.parse_args()

    try:
        assert args.image_directory != '' 

        col_size = os.get_terminal_size().columns

        print(col_size*'-')
        dg = len('descriptor generation')
        dgd = len('description generation details') 
        print((col_size-dgd)//2*' ' + 'Descriptor Generation Details' + (col_size-dgd)//2*' ')
        print(col_size*'-')
        print(f"- Directory: {args.image_directory}")
        print(f"- Number of Descriptors: {args.n_descriptors}")
        print(f"- subsample_ratio: {args.subsample_ratio}")
        print(col_size*'-')
        print((col_size-dg)//2*' ' + 'Descriptor Generation' + (col_size-dg)//2*' ')
        print(col_size*'-')


        SIFTFeatures(image_folder_path=args.image_directory, 
                    n_features=args.n_descriptors, 
                    subsample_ratio = args.subsample_ratio)
        print(col_size*'-')
        
    except AssertionError as e:
        print('You must provide an image directory')
        print('Exiting execution')
