from torch.utils.data import Dataset
import os
import pandas as pd
from os.path import join as osj
import numpy as np
import random


class CondensedMovies(Dataset):
    def __init__(self,
                 data_dir,
                 experts,
                 split='train'):

        self.data_dir = data_dir
        self.metadata_dir = osj(self.data_dir, 'challenge', 'metadata')
        self.experts = experts
        self.experts_used = [exp for exp, params in self.experts.items() if params['use']]
        self.split = split
        self.load_metadata()

    def load_metadata(self):

        if self.split in ['train', 'val']:
            df = pd.read_csv(osj(self.metadata_dir, 'train_val_challf0.csv'))
        elif self.split == 'test':
            raise ValueError("Test set not yet available")
        else:
            raise ValueError("Split should be either train, val or test")

        df = df[df['split'] == self.split]
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data.iloc[item]

        datum = {}
        video_data = {}
        for expert in self.experts_used:
            ftr_fp = osj(self.data_dir, 'challenge', 'features', self.experts[expert]['src'], str(sample['upload_year']),
                         sample['videoid'] + '.npy')
            ftr = np.load(ftr_fp, allow_pickle=True).item()['raw_feats']
            ftr, toks = self._pad_to_max_tokens(ftr, self.experts[expert]['max_tokens'])
            video_data[expert] = {'ftr': ftr, 'n_tokens': toks,}

        datum['text'] = sample['caption']
        datum['video'] = video_data

        return datum

    def _pad_to_max_tokens(self, ftr, max_tokens):
        """
        Pads or truncates features to max tokens.
        For truncation, at test time use center, at training use random.
        """
        output_shape = list(ftr.shape)
        output_shape[0] = max_tokens
        output_arr = np.ones(output_shape)

        if ftr.shape[0] <= max_tokens:
            output_arr[:ftr.shape[0]] = ftr
            n_tokens = ftr.shape[0]
        else:
            if self.split == 'train':
                start_idx = random.randint(0, ftr.shape[0] - max_tokens)
            else:
                start_idx = int((ftr.shape[0] / 2) - (max_tokens / 2))
            n_tokens = max_tokens
            output_arr = ftr[start_idx:start_idx+max_tokens]

        return output_arr, n_tokens


def make_new_splits():
    metadata_dir = '/scratch/shared/beegfs/maxbain/datasets/CondensedMovies/metadata'
    df = pd.read_csv(osj(metadata_dir, 'clips.csv'))
    split_data = pd.read_csv(osj(metadata_dir, 'split.csv')).set_index('imdbid')

    # put val in new train
    df.set_index('imdbid', inplace=True)
    df['split'] = split_data
    df['split'] = np.where(df['split'] == 'val', 'train', df['split'])
    df['split'] = np.where(df['split'] == 'test', 'val', df['split'])

    valdf = df[df['split'] == 'val']
    valdf['imdbid'] = valdf.index
    # reduce val to 1k? 2k?
    valimdbids = valdf['imdbid'].unique()
    val_to_keep = valimdbids[:len(valimdbids)//3]
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    #make_new_splits()
    ds = CondensedMovies('/scratch/shared/beegfs/maxbain/datasets/CondensedMovies/',
                         {
                             "resnext101": {
                                 "src": "pred_imagenet_25fps_256px_stride25_offset0/resnext101_32x48d",
                                 "max_tokens": 128,
                                 "use": True
                             },
                             "senet154": {
                                 "src": "pred_imagenet_25fps_256px_stride25_offset0/senet154",
                                 "max_tokens": 128,
                                 "use": True
                             },
                             "i3d": {
                                 "src": "pred_i3d_25fps_256px_stride8_offset0_inner_stride1/i3d",
                                 "max_tokens": 128,
                                 "use": True
                             },
                             "vggish": {
                                 "src": "pred_audio/vggish",
                                 "max_tokens": 128,
                                 "use": True
                             },
                             "densenet161": {
                                 "src": "pred_scene_25fps_256px_stride25_offset0/densenet161",
                                 "max_tokens": 128,
                                 "use": True
                             },
                             "r2p1d-ig65m": {
                                 "src": "pred_r2p1d_30fps_256px_stride16_offset0_inner_stride1/r2p1d-ig65m",
                                 "max_tokens": 128,
                                 "use": True
                             }
                         },
                         split='train'
                         )
    res = ds.__getitem__(0)
    import pdb; pdb.set_trace()
    print(res)
