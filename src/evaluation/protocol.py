"""Fail-closed verification and single-pass, MRI-only test inference."""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import BrainTumorDataset, verify_frozen_splits
from src.data.kfold import audit_splits, load_manifest, reload_splits
from src.models.morphonet_bt import MorphoNetBT
from src.utils.complexity import fingerprint

CLASSES = ['Meningioma', 'Glioma', 'Pituitary']
PROBS = ['prob_meningioma', 'prob_glioma', 'prob_pituitary']
VERSION = 'MorphoNet-BT Preprocessing v1.0'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def preflight(root):
    """Verify all five folds before any test image is evaluated."""
    root = Path(root).resolve()
    config = yaml.safe_load((root / 'configs/default.yaml').read_text())
    approved = yaml.safe_load((root / 'results/architecture/preprocessing_v1_approved.yaml').read_text())
    require({key: config[key] for key in approved} == approved, 'Preprocessing changed')
    hashes = verify_frozen_splits(root)
    frozen = json.loads((root / 'results/architecture/frozen_input_hashes.json').read_text())
    require(hashes == frozen['splits'], 'Architecture-era split mismatch')
    for name, digest in frozen['preprocessing_sources'].items():
        require(sha256(root / name) == digest, f'Frozen source changed: {name}')
    model = MorphoNetBT(**config['model'])
    observed = fingerprint(model, config['model'], root)
    saved = json.loads((root / 'results/complexity/architecture_fingerprint.json').read_text())
    require(observed == saved, 'Architecture fingerprint mismatch')
    splits = reload_splits(root / 'splits')
    manifest = load_manifest(root / 'results/dataset_statistics.csv')
    split_audit, coverage = audit_splits(splits, manifest)
    rows = []
    expected_epochs = [60, 18, 22, 55, 45]
    expected_scores = [.9641710676, .8508045414, .9400990494, .9634327921, .9444884525]
    for fold in range(1, 6):
        path = root / f'checkpoints/main/fold_{fold}/best_macro_f1.pt'
        marker_path = root / f'results/main/fold_{fold}/training_complete.json'
        marker = json.loads(marker_path.read_text())
        require(Path(marker['checkpoint_path']).resolve() == path, 'Completion checkpoint mismatch')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        require(checkpoint['fold'] == fold, 'Checkpoint fold mismatch')
        require(checkpoint['architecture_fingerprint'] == saved['fingerprint_sha256'], 'Checkpoint architecture mismatch')
        require(checkpoint['configuration'] == config, 'Checkpoint configuration mismatch')
        require(checkpoint['preprocessing_version'] == VERSION, 'Preprocessing version mismatch')
        require(checkpoint['split_hashes'] == hashes, 'Checkpoint split mismatch')
        require(checkpoint['seed'] == 42, 'Checkpoint seed mismatch')
        require(checkpoint['epoch'] == checkpoint['best_epoch'] == marker['best_epoch'] == expected_epochs[fold-1], 'Best epoch mismatch')
        require(abs(checkpoint['validation_macro_f1'] - expected_scores[fold-1]) < 1e-9, 'Confirmed validation score mismatch')
        require(checkpoint['validation_macro_f1'] == marker['best_validation_macro_f1'], 'Completion score mismatch')
        require(marker['total_epochs'] >= marker['best_epoch'], 'Invalid completion epoch')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        dataset = BrainTumorDataset(root / f'splits/fold_{fold}_test.csv', 'test', config=approved, include_targets=False)
        rows.append(dict(Fold=fold, BestEpoch=checkpoint['epoch'], ValidationMacroF1=checkpoint['validation_macro_f1'],
                         TestPatients=dataset.frame.PID.nunique(), TestSlices=len(dataset), Checkpoint=str(path),
                         CheckpointSHA256=sha256(path), CompletionSHA256=sha256(marker_path),
                         TestSplitSHA256=hashes[f'fold_{fold}_test.csv']))
        del checkpoint
    metadata = dict(architecture_fingerprint=saved['fingerprint_sha256'], preprocessing_version=VERSION,
                    preprocessing=approved, evaluation_seed=42, class_mapping=dict(enumerate(CLASSES)),
                    split_sha256=hashes, checkpoints=rows, precision='float32; no autocast; TF32 disabled',
                    batch_size=16, num_workers=0, augmentation=False, model_input='MRI only',
                    bootstrap=dict(unit='patient', iterations=2000, seed=42, method='percentile', confidence=.95),
                    software=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                                  pandas=pd.__version__, sklearn=sklearn.__version__, cuda=torch.version.cuda),
                    device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    return dict(root=root, config=approved, model_config=config['model'], manifest=manifest, splits=splits,
                checkpoints=pd.DataFrame(rows), split_audit=split_audit, coverage=coverage, metadata=metadata)


def validate_predictions(frame, expected, fold):
    """Check identities, class mapping, probabilities and exact split coverage."""
    require(len(frame) == len(expected) and frame.sample_id.is_unique, 'Prediction count/uniqueness mismatch')
    require(set(frame.sample_id) == set(expected.sample_id), 'Prediction sample coverage mismatch')
    actual = frame.set_index('sample_id').loc[expected.sample_id]
    require(np.array_equal(actual.PID.to_numpy(), expected.PID.to_numpy()), 'Prediction PID mismatch')
    require(np.array_equal(actual.true_label.to_numpy(), expected.original_label.to_numpy()-1), 'Prediction labels mismatch')
    require(np.array_equal(actual.file_path.to_numpy(), expected.file_path.to_numpy()), 'Prediction source mismatch')
    require(frame.fold.eq(fold).all(), 'Prediction fold mismatch')
    probabilities = frame[PROBS].to_numpy()
    require(np.isfinite(probabilities).all() and (probabilities >= 0).all() and (probabilities <= 1).all(), 'Invalid probabilities')
    require(np.allclose(probabilities.sum(axis=1), 1, atol=1e-6), 'Softmax sum mismatch')
    require(np.array_equal(probabilities.argmax(axis=1), frame.predicted_label), 'Argmax mismatch')
    for prefix in ['true', 'predicted']:
        require(frame[f'{prefix}_class'].tolist() == [CLASSES[int(i)] for i in frame[f'{prefix}_label']], 'Class mapping mismatch')


def evaluate_fold(context, fold):
    """Run one pass, or reuse a hash-verified completed receipt on reruns.

    A started but incomplete evaluation is deliberately not retried automatically.
    No optimizer or scheduler is constructed; only model weights are consumed.
    """
    root = context['root']
    row = context['checkpoints'].set_index('Fold').loc[fold]
    output = root / f'results/main/fold_{fold}'
    prediction_path = output / 'test_predictions.csv'
    receipt_path = output / 'test_evaluation_receipt.json'
    started_path = output / 'test_evaluation_started.json'
    identity = dict(fold=fold, checkpoint_sha256=row.CheckpointSHA256, test_split_sha256=row.TestSplitSHA256,
                    architecture=context['metadata']['architecture_fingerprint'], preprocessing_version=VERSION,
                    preprocessing_source_sha256={str(p.relative_to(root)): sha256(p) for p in sorted((root/'src/data').glob('*.py'))},
                    inference_source_sha256=sha256(Path(__file__)), batch_size=16, precision='float32')
    require(sha256(row.Checkpoint) == row.CheckpointSHA256, 'Checkpoint changed after preflight')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        require(receipt['identity'] == identity and receipt['status'] == 'complete', 'Cached evaluation protocol mismatch')
        require(sha256(prediction_path) == receipt['predictions_sha256'], 'Cached predictions changed')
        frame = pd.read_csv(prediction_path, dtype={'PID': str, 'sample_id': str}, keep_default_na=False)
        validate_predictions(frame, context['splits'][fold]['test'], fold)
        print(f'Fold {fold}: verified saved predictions; no repeated inference.')
        return frame
    require(not prediction_path.exists() and not started_path.exists(), 'Incomplete/unreceipted evaluation found; stop for review')
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MorphoNetBT(**context['model_config'])
    checkpoint = torch.load(row.Checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint
    model.to(device).eval()
    dataset = BrainTumorDataset(root / f'splits/fold_{fold}_test.csv', 'test', config=context['config'], include_targets=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, drop_last=False)
    save_json(started_path, dict(identity=identity, started_at=datetime.now(timezone.utc).isoformat()))
    probabilities, sample_ids, pids, labels = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            require(set(batch) == {'image', 'label', 'pid', 'sample_id'}, 'Unexpected morphology target in test batch')
            images = batch['image'].to(device)
            require(images.dtype == torch.float32 and tuple(images.shape[1:]) == (1, 256, 256), 'Invalid input tensor')
            require(bool(torch.isfinite(images).all()), 'Nonfinite input')
            probabilities.append(model(images).softmax(dim=1).cpu().numpy())
            sample_ids.extend(batch['sample_id'])
            pids.extend(batch['pid'])
            labels.extend(batch['label'].tolist())
    probability_array = np.concatenate(probabilities)
    frame = pd.DataFrame(dict(fold=fold, PID=pids, sample_id=sample_ids, true_label=labels))
    frame['file_path'] = dataset.frame.set_index('sample_id').loc[sample_ids, 'file_path'].to_numpy()
    frame['predicted_label'] = probability_array.argmax(axis=1)
    for prefix in ['true', 'predicted']:
        frame[f'{prefix}_class'] = [CLASSES[i] for i in frame[f'{prefix}_label']]
    frame[PROBS] = probability_array
    validate_predictions(frame, dataset.frame, fold)
    temporary = prediction_path.with_suffix('.tmp')
    frame.to_csv(temporary, index=False, float_format='%.17g')
    temporary.replace(prediction_path)
    save_json(receipt_path, dict(identity=identity, status='complete', inference_passes=1, samples=len(frame),
                                completed_at=datetime.now(timezone.utc).isoformat(), predictions_sha256=sha256(prediction_path)))
    del model
    print(f'Fold {fold}: one test pass completed; {len(frame)} slices, {frame.PID.nunique()} patients.')
    return frame


def audit_oof(context, frames):
    """Gate final metrics on exact, patient-separated outer-test coverage."""
    require(set(frames) == set(range(1, 6)), 'Five folds required')
    for fold, frame in frames.items():
        validate_predictions(frame, context['splits'][fold]['test'], fold)
    oof = pd.concat(frames.values(), ignore_index=True)
    expected = set(context['manifest'].sample_id)
    audit = dict(predictions=len(oof), unique_samples=oof.sample_id.nunique(), unique_patients=oof.PID.nunique(),
                 duplicate_samples=int(oof.sample_id.duplicated().sum()), missing_samples=len(expected-set(oof.sample_id)),
                 unexpected_samples=len(set(oof.sample_id)-expected),
                 patients_in_multiple_folds=int((oof.groupby('PID').fold.nunique() != 1).sum()))
    require(list(audit.values()) == [3064, 3064, 233, 0, 0, 0, 0], f'OOF audit failed: {audit}')
    require(oof.groupby('PID').true_label.nunique().eq(1).all(), 'Inconsistent patient true labels')
    oof.to_csv(context['root'] / 'results/main/oof_predictions.csv', index=False, float_format='%.17g')
    save_json(context['root'] / 'results/main/oof_integrity_audit.json', audit)
    return oof, audit
