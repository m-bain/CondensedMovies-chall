from base import BaseDataLoader, BaseDataLoaderExplicitSplit
from torchvision import transforms
from data_loader.CondensedMovies_dataset import CondensedMovies

class CondensedMoviesDataLoader(BaseDataLoaderExplicitSplit):
    """
    CondensedeMovies DataLoader.
    """

    def __init__(self, data_dir, experts, batch_size, split='train', shuffle=True, num_workers=4):
        self.data_dir = data_dir
        self.dataset = CondensedMovies(data_dir, experts, split)
        self.dataset_name = 'CondensedMovies'
        # batch size of entire val test set. change this for intra-movie
        if split in ['train', 'val']:
            drop_last = True
        else:
            drop_last = False
        #    batch_size = len(self.dataset.data'clips')
        super().__init__(self.dataset, batch_size, shuffle, num_workers, drop_last=drop_last)