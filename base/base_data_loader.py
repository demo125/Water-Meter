import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import ntpath

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def get_train_val_idxs(self, split):
        #all augmented images from single original image and original image must belong to to same set
        val_img_names = set()
        val_img_idxs = []
        train_img_names = set()
        train_img_idxs = []
        for idx, img_path in enumerate(self.dataset.data):
            img_basename = ntpath.basename(img_path)
            extension = img_basename.split('.')[-1]

            if 'samples' in img_path:
                train_img_idxs.append(idx)
                continue

            if '__' in img_basename:
                img_non_aug_name = ''.join(img_basename.split('__')[:-1]) +'.'+extension
            else:
                img_non_aug_name = img_basename

            if img_non_aug_name in val_img_names:
                val_img_names.add(img_non_aug_name)
                val_img_idxs.append(idx)
            elif img_non_aug_name in train_img_names:
                train_img_names.add(img_non_aug_name)
                train_img_idxs.append(idx)
            else:
                if np.random.rand() < split:
                    val_img_names.add(img_non_aug_name)
                    val_img_idxs.append(idx)
                else:
                    train_img_names.add(img_non_aug_name)
                    train_img_idxs.append(idx)
        
        return train_img_idxs, val_img_idxs

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)
        assert isinstance(split, float)
        
        train_idx, valid_idx = self.get_train_val_idxs(split)

        # if isinstance(split, int):
        #     assert split > 0
        #     assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
        #     len_valid = split
        # else:
        #     len_valid = int(self.n_samples * split)

        # valid_idx = idx_full[0:len_valid]
        # train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)