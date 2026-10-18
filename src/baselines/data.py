"""MRI-only view of the unchanged, approved preprocessing pipeline."""
from torch.utils.data import Dataset
from src.data.dataset import BrainTumorDataset


class MRIOnlyBaselineDataset(Dataset):
    """Expose no auxiliary supervision to the baseline or classification loss.

    User-approved policy: training preprocessing may use the mask internally for
    tumor-clipping rejection. The frozen pipeline also computes auxiliary arrays;
    these are immediately discarded. Validation/test request no mask or targets.
    Seed, epoch, image augmentation, normalization, and split checks are inherited.
    """
    def __init__(self, csv_path, mode, config=None, seed=42):
        self.dataset = BrainTumorDataset(csv_path, mode, config=config, seed=seed,
                                         include_targets=(mode == 'train'))

    @property
    def frame(self):
        return self.dataset.frame

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        return {key: sample[key] for key in ('image', 'label', 'pid', 'sample_id')}
