"""Train/validation-only data adapter; construction rejects the Test partition."""
from torch.utils.data import Dataset
from src.data.dataset import BrainTumorDataset

class AblationDataset(Dataset):
    def __init__(self, csv_path, mode, morphology, config, seed=42):
        if mode not in ('train','val'):
            raise ValueError('Test data are locked during Phase 12')
        self.mode=mode; self.morphology=morphology
        self.dataset=BrainTumorDataset(csv_path,mode,config=config,seed=seed,include_targets=morphology or mode=='train')
    @property
    def frame(self): return self.dataset.frame
    def set_epoch(self,epoch): self.dataset.set_epoch(epoch)
    def __len__(self): return len(self.dataset)
    def __getitem__(self,index):
        sample=self.dataset[index]
        keys=['image','label','pid','sample_id']+(['region','boundary','distance'] if self.morphology else [])
        return {key:sample[key] for key in keys}
