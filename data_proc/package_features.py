import argparse
import numpy as np
import glob
import os
import multiprocessing as mp

def package_ftr(import_fp,save_fp, downsample):
    datum = np.load(import_fp, allow_pickle=True).item()
    datum['raw_feats'] = datum['raw_feats'][::args.downsample]
    datum['frames'] = datum['frames'][::downsample]
    datum['logits'] = datum['logits'][::downsample]

    save_dir = '/'.join(save_fp.split('/')[:-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(save_fp, datum)


def package_features(args):

    ftrs_list = glob.glob(os.path.join(args.src_dir, args.expert, '**', '*.npy'), recursive=True)
    ctr = 0
    for ftr_fp in ftrs_list:
        ctr += 1
        if ctr % 1 == 0:
            print(ctr)
        save_fp = ftr_fp.replace(args.src_dir, args.exp_dir)
        datum = np.load(ftr_fp, allow_pickle=True).item()
        datum['raw_feats'] = datum['raw_feats'][::args.downsample]
        datum['frames'] = datum['frames'][::args.downsample]
        datum['logits'] = datum['logits'][::args.downsample]

        save_dir = '/'.join(save_fp.split('/')[:-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(save_fp, datum)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, default='/scratch/shared/beegfs/maxbain/datasets/CondensedMovies/high-quality/processing')
    parser.add_argument('--exp_dir', type=str, default='/scratch/shared/beegfs/maxbain/datasets/CondensedMovies/challenge/features')
    parser.add_argument('--expert', type=str, required=True)
    parser.add_argument('--downsample', type=int)

    args = parser.parse_args()
    package_features(args)
