from base import BaseDataLoader, BaseDataLoaderExplicitSplit
from torchvision import transforms
from Cond

class CondensedMoviesDataLoader(BaseDataLoader):
    """
    CondensedeMovies DataLoader.
    """

    def __init__(self, data_dir, metadata_dir, label, experts_used, experts, max_tokens, batch_size, split='train', shuffle=True, num_workers=4):
        self.data_dir = data_dir
        self.dataset = CondensedMovies(data_dir, metadata_dir, label, experts_used, experts, max_tokens, split)

        # batch size of entire val test set. change this for intra-movie
        if split in ['val', 'test']:
            batch_size = len(self.dataset.data['clips'])
        super().__init__(self.dataset, batch_size, shuffle, split, num_workers)