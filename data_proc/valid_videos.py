import os
import argparse
import numpy as np
import pandas as pd
import glob
"""
This script outputs a list of valid videos given the target features.
"""
def main(args):
    df = pd.read_csv(args.features_list_csv)

    exp_arr = {}
    for idx, row in df.iterrows():
        expert_dir = row['dir']
        expert_ext = row['ext']
        found = glob.glob(os.path.join(expert_dir, '*', '*.' + expert_ext), recursive=True)
        found = [x.replace(expert_dir, '').strip('/') for x in found]
        exp_arr[expert_dir] = pd.Series(found)

    if args.filter_by == 'one':
        fdf = pd.concat(list(exp_arr.values()))
        fdf = fdf.drop_duplicates()
    elif args.filter_by == 'all':
        raise NotImplementedError

    fdf.to_csv(args.features_list_csv.replace('.csv', f'_valid_{args.filter_by}.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--video_directory', required=True, type=str,
    #                    help='path to directory containing videos.')
    parser.add_argument('--features_list_csv', required=True, type=str,
                        help='path to csv file containing directories of target features')
    parser.add_argument('--filter_by', default='all', choices=['all', 'one'],
                        help='valid is either all features are present or "one"')
    args = parser.parse_args()
    main(args)