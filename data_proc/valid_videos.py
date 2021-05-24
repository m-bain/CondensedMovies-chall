import os
import argparse
import numpy as np

"""
This script outputs a list of valid videos given the target features.
"""



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_directory', required=True, type=str,
                        help='path to directory containing videos.')
    parser.add_argument('--features_list_txt', required=True, type=str,
                        help='path to txt file containing directories of target features')
    parser.add_argument('--filter_by', default='all', choices=['all', 'one'],
                        help='valid is either all features are present or "one"')
