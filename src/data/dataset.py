"""Lazy CPU Dataset for immutable Figshare split CSV files."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .inspection import read_record, decode_pid
from .preprocessing import DEFAULT_CONFIG, preprocess, validate_config

ORIGINAL_TO_INDEX = {1: 0, 2: 1, 3: 2}
INDEX_TO_CLASS = {0: 'Meningioma', 1: 'Glioma', 2: 'Pituitary'}


def verify_frozen_splits(project_root):
    root = Path(project_root)
    metadata = json.loads((root / 'results/kfold/split_metadata.json').read_text(encoding='utf-8'))
    observed = {}
    for name, expected in metadata['split_sha256'].items():
        path = root / 'splits' / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f'Frozen split changed: {path}')
        observed[name] = digest
    if len(observed) != 15:
        raise ValueError('Expected 15 canonical split files')
    if hashlib.sha256((root / 'results/dataset_statistics.csv').read_bytes()).hexdigest() != metadata['manifest_sha256']:
        raise ValueError('Phase 1 manifest has changed')
    return observed


def load_checked_sample(record, include_mask=True):
    fields, _ = read_record(record)
    if fields is None:
        raise ValueError('Missing cjdata')
    pid = decode_pid(fields['PID'])
    label_array = np.asarray(fields['label']).ravel()
    if label_array.size != 1 or label_array[0] not in ORIGINAL_TO_INDEX:
        raise ValueError('Invalid raw label')
    original_label = int(label_array[0])
    if pid != str(record['PID']) or original_label != int(record['original_label']):
        raise ValueError('Raw PID/label disagrees with frozen Phase 2 metadata')
    if INDEX_TO_CLASS[ORIGINAL_TO_INDEX[original_label]] != record['class_name']:
        raise ValueError('Class mapping inconsistent')
    image = fields['image']
    digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
    if digest != record['image_sha256']:
        raise ValueError('MRI contents changed since Phase 1')
    return image, fields['tumorMask'] if include_mask else None


class BrainTumorDataset(Dataset):
    def __init__(self, csv_path, mode, config=None, seed=42, include_targets=True):
        self.csv_path = Path(csv_path).resolve()
        self.root = self.csv_path.parent.parent
        verify_frozen_splits(self.root)
        if mode not in ('train', 'val', 'test') or self.csv_path.name not in {
                f'fold_{fold}_{mode}.csv' for fold in range(1, 6)}:
            raise ValueError('Explicit mode must match canonical split filename')
        self.mode, self.seed, self.epoch = mode, int(seed), 0
        self.config = deepcopy(DEFAULT_CONFIG if config is None else config)
        validate_config(self.config)
        self.include_targets = include_targets
        if mode == 'train' and self.config['augmentation']['enabled'] and not include_targets:
            raise ValueError('Training geometry requires supervision masks')
        self.frame = pd.read_csv(self.csv_path, dtype={'PID': str, 'sample_id': str}, keep_default_na=False)
        if not self.frame['split'].eq(mode).all() or not self.frame.sample_id.is_unique:
            raise ValueError('Invalid split metadata')
        reference = pd.read_csv(self.root / 'results/dataset_statistics.csv',
                                dtype={'PID': str, 'sample_id': str}, keep_default_na=False).set_index('sample_id')
        for row in self.frame.to_dict('records'):
            canonical = reference.loc[row['sample_id']]
            for key in ('PID', 'class_name', 'source', 'member', 'image_sha256'):
                if row[key] != canonical[key]:
                    raise ValueError(f'Manifest mismatch in {key}')
            if row['original_label'] != canonical['raw_label']:
                raise ValueError('Manifest label mismatch')
        for source, subset in self.frame.groupby('source'):
            if not Path(source).is_file():
                raise FileNotFoundError(source)
            if subset.member.ne('').any():
                with zipfile.ZipFile(source) as archive:
                    missing = set(subset.member) - set(archive.namelist())
                    if missing:
                        raise FileNotFoundError(f'Missing archive members: {sorted(missing)}')

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError('Epoch must be a nonnegative integer')
        self.epoch = epoch

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index].to_dict()
        try:
            image, mask = load_checked_sample(row, include_mask=self.include_targets)
            # Stable per-sample/epoch draws independent of worker scheduling or shuffle order.
            identity = int.from_bytes(hashlib.sha256(row['sample_id'].encode()).digest()[:8], 'little')
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, identity]))
            arrays, _ = preprocess(image, mask, self.mode, self.config, rng)
            result = {key: torch.from_numpy(value) for key, value in arrays.items()}
            result.update(label=torch.tensor(ORIGINAL_TO_INDEX[row['original_label']], dtype=torch.long),
                          pid=row['PID'], sample_id=row['sample_id'])
            return result
        except Exception as exc:
            raise ValueError(f"sample={row['sample_id']} PID={row['PID']} class={row['class_name']} "
                             f"file={row['source']}::{row['member']}: {exc}") from exc
