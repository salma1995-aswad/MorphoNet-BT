"""Deterministic patient-grouped splits and strict on-disk audit."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

LABELS = {1: 'Meningioma', 2: 'Glioma', 3: 'Pituitary'}
PARTS = ('train', 'val', 'test')
IDENTITY = ['sample_id', 'source', 'member', 'PID', 'original_label', 'class_name', 'image_sha256']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_manifest(path):
    frame = pd.read_csv(path, dtype={'PID': str, 'sample_id': str}, keep_default_na=False)
    require(set(['raw_label'] + [c for c in IDENTITY if c != 'original_label']).issubset(frame), 'Missing required columns')
    labels = pd.to_numeric(frame.raw_label, errors='raise')
    require(labels.isin(LABELS).all(), 'Invalid original label')
    frame['original_label'] = labels.astype(int)
    require(frame.PID.str.strip().ne('').all() and frame.PID.eq(frame.PID.str.strip()).all(), 'Invalid PID')
    require(frame.sample_id.ne('').all() and frame.sample_id.is_unique, 'Invalid or duplicate sample ID')
    require(frame.class_name.eq(frame.original_label.map(LABELS)).all(), 'Class mapping mismatch')
    require(frame.groupby('PID').original_label.nunique().eq(1).all(), 'Inconsistent patient labels')
    require(not frame.duplicated(['source', 'member']).any(), 'Duplicate file reference')
    require(frame.groupby('image_sha256').PID.nunique().le(1).all(), 'Exact duplicate image across patients')
    for source in frame.source.unique():
        require(Path(source).is_file(), f'Missing source: {source}')
    frame['file_path'] = frame.source + np.where(frame.member.ne(''), '::' + frame.member, '')
    return frame.sort_values(['PID', 'sample_id'], kind='stable').reset_index(drop=True)


def patient_manifest(frame):
    return frame.groupby('PID', sort=True).agg(
        original_label=('original_label', 'first'), class_name=('class_name', 'first'),
        num_slices=('sample_id', 'size')).reset_index()


def generate_outer_folds(frame, seed=42):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return [(frame.iloc[remaining].copy(), frame.iloc[test].copy())
            for remaining, test in splitter.split(frame, frame.original_label, groups=frame.PID)]


def add_validation(outer_folds, seed=42, validation_fraction=0.2):
    splits = {}
    for fold, (remaining, test) in enumerate(outer_folds, 1):
        patients = patient_manifest(remaining)
        train_pid, val_pid = train_test_split(
            patients.PID.to_numpy(), test_size=validation_fraction, random_state=seed,
            shuffle=True, stratify=patients.original_label.to_numpy())
        parts = {'train': remaining[remaining.PID.isin(train_pid)].copy(),
                 'val': remaining[remaining.PID.isin(val_pid)].copy(), 'test': test.copy()}
        for name, data in parts.items():
            data['fold'], data['split'] = fold, name
            parts[name] = data.sort_values(['PID', 'sample_id'], kind='stable').reset_index(drop=True)
        splits[fold] = parts
    return splits


def verify_patient_separation(train, val, test, expected):
    parts = dict(zip(PARTS, (train, val, test)))
    pids = {name: set(data.PID) for name, data in parts.items()}
    overlaps = {f'{a}_{b}_overlap': len(pids[a] & pids[b])
                for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    require(not any(overlaps.values()), f'PID LEAKAGE: {overlaps}')
    require(set.union(*pids.values()) == set(expected.PID), 'Patient population mismatch')
    combined = pd.concat(parts.values(), ignore_index=True)
    require(combined.sample_id.is_unique, 'Duplicate sample within fold')
    require(set(combined.sample_id) == set(expected.sample_id), 'Sample coverage mismatch')
    for name, data in parts.items():
        require(len(data) > 0, f'Empty {name}')
        require(set(data.original_label) == set(LABELS), f'Missing class in {name}')
        selected = expected[expected.PID.isin(pids[name])]
        require(set(data.sample_id) == set(selected.sample_id), f'Incomplete patient slices in {name}')
        pd.testing.assert_frame_equal(
            data[IDENTITY].sort_values('sample_id').reset_index(drop=True),
            selected[IDENTITY].sort_values('sample_id').reset_index(drop=True), check_dtype=False)
    return {**overlaps, 'num_patients': len(set.union(*pids.values())),
            'num_slices': len(combined), 'status': 'PASS'}


def audit_splits(splits, frame):
    require(set(splits) == set(range(1, 6)), 'Expected exactly folds 1-5')
    audits, test_pids, test_samples = [], Counter(), Counter()
    for fold, parts in sorted(splits.items()):
        for name in PARTS:
            require(parts[name].fold.eq(fold).all() and parts[name]['split'].eq(name).all(), 'Fold/split metadata mismatch')
        audits.append({'fold': fold, **verify_patient_separation(*(parts[p] for p in PARTS), frame)})
        test_pids.update(set(parts['test'].PID))
        test_samples.update(parts['test'].sample_id)
    require(set(test_pids) == set(frame.PID) and set(test_pids.values()) == {1}, 'Outer test PID coverage failure')
    require(set(test_samples) == set(frame.sample_id) and set(test_samples.values()) == {1}, 'Outer test slice coverage failure')
    coverage = {'unique_test_patients': len(test_pids), 'unique_test_samples': len(test_samples),
                'each_patient_test_count': 1, 'each_sample_test_count': 1, 'status': 'PASS'}
    return pd.DataFrame(audits), coverage


def summarize_splits(splits, frame):
    rows, balance = [], []
    global_patients = patient_manifest(frame)
    for fold, parts in sorted(splits.items()):
        for name, data in parts.items():
            patients = patient_manifest(data)
            row = dict(fold=fold, split=name, num_patients=len(patients), num_slices=len(data))
            for label, class_name in LABELS.items():
                prefix = class_name.lower()
                row[f'{prefix}_patients'] = int(patients.original_label.eq(label).sum())
                row[f'{prefix}_slices'] = int(data.original_label.eq(label).sum())
                for level, subset, population in [('patient', patients, global_patients), ('slice', data, frame)]:
                    proportion = subset.original_label.eq(label).mean()
                    global_proportion = population.original_label.eq(label).mean()
                    balance.append(dict(fold=fold, split=name, level=level, class_name=class_name,
                                        proportion=proportion, global_proportion=global_proportion,
                                        signed_deviation_pp=100 * (proportion - global_proportion),
                                        absolute_deviation_pp=100 * abs(proportion - global_proportion)))
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(balance)


def assignments(splits):
    return pd.concat([patient_manifest(p['test']).assign(outer_test_fold=fold)
                      for fold, p in sorted(splits.items())]).sort_values('PID').reset_index(drop=True)


def save_splits(splits, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for fold, parts in sorted(splits.items()):
        for name, data in parts.items():
            data.to_csv(directory / f'fold_{fold}_{name}.csv', index=False, lineterminator='\n')


def reload_splits(directory):
    return {fold: {name: pd.read_csv(Path(directory) / f'fold_{fold}_{name}.csv',
                                    dtype={'PID': str, 'sample_id': str}, keep_default_na=False)
                   for name in PARTS} for fold in range(1, 6)}


def save_metadata(path, manifest_path, splits_dir, seed=42):
    metadata = dict(seed=seed, outer_strategy='StratifiedGroupKFold', n_splits=5, shuffle=True,
                    outer_input='One row per slice; groups=PID; y=original_label',
                    inner_strategy='Stratified train_test_split on one row per PID',
                    inner_validation_fraction=0.2, inner_seed=seed,
                    canonical_order=['PID lexicographic', 'sample_id lexicographic'],
                    label_mapping=LABELS, sklearn_version=sklearn.__version__,
                    numpy_version=np.__version__, pandas_version=pd.__version__,
                    manifest_sha256=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
                    split_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(Path(splits_dir).glob('fold_*.csv'))})
    Path(path).write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata
