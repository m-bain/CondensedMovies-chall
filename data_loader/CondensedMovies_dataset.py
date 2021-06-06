from torch.utils.data import Dataset
import os
import pandas as pd
from os.path import join as osj
import numpy as np

class CondensedMovies(Dataset):
    def __init__(self,
                 data_dir,
                 experts_used,
                 experts,
                 split='train'):

        self.data_dir = data_dir
        self.metadata_dir = osj(self.data_dir, 'metadata')
        self.experts_used = experts_used
        self.experts = experts
        self.split = split
        self.load_metadata()

    def load_metadata(self):
        data = {
            'movies': pd.read_csv(osj(self.metadata_dir, 'movies.csv')).set_index('imdbid'),
            'casts': pd.read_csv(osj(self.metadata_dir, 'casts.csv')).set_index('imdbid'),
            'clips': pd.read_csv(osj(self.metadata_dir, 'clips.csv')).set_index('videoid'),
            'descs': pd.read_csv(osj(self.metadata_dir, 'descriptions.csv')).set_index('videoid'),
        }

        # filter by split {'train', 'val', 'test'}
        split_data = pd.read_csv(osj(self.metadata_dir, 'split.csv')).set_index('imdbid')



        if self.split == 'train_val':
            ids = split_data[split_data['split'].isin(['train', 'val'])].index
        else:
            ids = split_data[split_data['split'] == self.split].index

        for key in data:
            if 'imdbid' in data[key]:
                filter = data[key]['imdbid'].isin(ids)
            else:
                filter = data[key].index.isin(ids)
            data[key] = data[key][filter]        # filter by split {'train', 'val', 'test'}
        split_data = pd.read_csv(osj(self.metadata_dir, 'split.csv')).set_index('imdbid')
        if self.split == 'train_val':
            ids = split_data[split_data['split'].isin(['train', 'val'])].index
        else:
            ids = split_data[split_data['split'] == self.split].index
        for key in data:
            if 'imdbid' in data[key]:
                filter = data[key]['imdbid'].isin(ids)
            else:
                filter = data[key].index.isin(ids)
            data[key] = data[key][filter]

        # filter only videos containing features
        valid_ids = pd.read_csv('data_proc/simple_rgb_valid_one.csv')['0'].str.split('/').str[1].str.split('.').str[0]

        filter = data['clips']['videoid'].isin(valid_ids)
        data['clips'] = data['clips'][filter]


        self.data = data
        import pdb; pdb.set_trace()

    def __len__(self):
        return len(self.data['clips'])

    def __getitem__(self, item):
        sample = self.data['clips'].iloc[item]

        datum = {}
        for expert in self.experts_used:
            ftr_fp = osj(self.data_dir, 'challenge', 'features', self.experts[expert], str(sample['upload_year']), sample.name + '.npy')
            ftr = np.load(ftr_fp, allow_pickle=True)
            import pdb; pdb.set_trace()
if __name__ == "__main__":
    ds = CondensedMovies('/scratch/shared/beegfs/maxbain/datasets/CondensedMovies/',
                         ['rgb'],
                         {
                             'rgb': 'pred_imagenet_25fps_256px_stride1_offset0/resnext101_32x48d'
                         },
                         )

    ds.__getitem__(0)