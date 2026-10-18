"""Phase 10: verified single-pass baseline inference; no training entry point."""
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data.dataset import BrainTumorDataset
from src.data.kfold import reload_splits, load_manifest
from src.models.baselines import BASELINES, DISPLAY_NAMES, build_baseline
from src.baselines.training import preflight as training_preflight, run_identity, compatible_identity
from .protocol import CLASSES, PROBS, VERSION, require, sha256, save_json, validate_predictions


def preflight(root):
    root = Path(root).resolve()
    info = training_preflight(root)
    splits = reload_splits(root / 'splits')
    rows = []
    protected = {}
    for directory in ['results/main', 'results/complexity', 'results/baselines/setup', 'results/baselines/training', 'splits', 'configs']:
        for path in (root / directory).rglob('*'):
            if path.is_file():
                protected[path.relative_to(root).as_posix()] = sha256(path)
    for name in BASELINES:
        model = build_baseline(name)
        for fold in range(1, 6):
            path = root / f'checkpoints/baselines/{name}/fold_{fold}/best_macro_f1.pt'
            marker_path = root / f'results/baselines/training/{name}/fold_{fold}/training_complete.json'
            marker = json.loads(marker_path.read_text())
            require(marker['model'] == name and marker['fold'] == fold, 'Completion identity mismatch')
            require(Path(marker['checkpoint_path']).resolve() == path, 'Completion checkpoint mismatch')
            digest = sha256(path)
            require(digest == marker['checkpoint_sha256']['best_macro_f1.pt'], 'Best checkpoint digest mismatch')
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            for identity in [marker['identity'], checkpoint['identity']]:
                require(compatible_identity(identity, run_identity(info, name, fold)), 'Training identity mismatch')
            require(checkpoint['architecture'] == name and checkpoint['fold'] == fold, 'Checkpoint model/fold mismatch')
            require(checkpoint['configuration'] == info['config'], 'Checkpoint configuration mismatch')
            require(checkpoint['preprocessing'] == info['preprocessing'] and checkpoint['preprocessing_version'] == VERSION, 'Preprocessing mismatch')
            require(checkpoint['class_mapping'] == dict(enumerate(CLASSES)) and checkpoint['seed'] == 42, 'Labels/seed mismatch')
            require(checkpoint['epoch'] == checkpoint['best_epoch'] == marker['best_epoch'], 'Best epoch mismatch')
            require(checkpoint['best_validation_macro_f1'] == marker['best_validation_macro_f1'], 'Validation selection mismatch')
            require(checkpoint['validation_metrics']['macro_f1'] == marker['best_validation_macro_f1'], 'Best score mismatch')
            require(marker['total_epochs'] >= marker['best_epoch'] and marker['test_evaluated'] is False, 'Invalid completion state')
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            test = splits[fold]['test']
            for part in ['train', 'val']:
                require(not set(test.PID) & set(splits[fold][part].PID), 'Patient leakage')
            protected[path.relative_to(root).as_posix()] = digest
            rows.append(dict(Model=name, Fold=fold, BestEpoch=checkpoint['epoch'], ValidationMacroF1=marker['best_validation_macro_f1'],
                TestPatients=int(test.PID.nunique()), TestSlices=len(test), Checkpoint=str(path), CheckpointSHA256=digest,
                CompletionSHA256=sha256(marker_path), TestSplitSHA256=info['split_hashes'][f'fold_{fold}_test.csv'],
                TrainerSourceSHA256=checkpoint['identity']['trainer_sources']['src/baselines/training.py']))
            del checkpoint
        del model
    import sklearn, torchvision, platform
    metadata = dict(models=list(BASELINES), checkpoints=rows, class_mapping=dict(enumerate(CLASSES)),
        configuration_fingerprint=info['configuration_fingerprint'], preprocessing=info['preprocessing'], preprocessing_version=VERSION,
        evaluation_seed=42, precision='FP32; no autocast; TF32 disabled', batch_size=16, num_workers=0, augmentation=False,
        model_input='MRI only; no masks or morphology targets', software=dict(python=platform.python_version(), torch=torch.__version__,
        torchvision=torchvision.__version__, numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, cuda=torch.version.cuda),
        GPU=info['runtime']['device'], bootstrap=dict(unit='patient', iterations=2000, seed=42, method='percentile', shared_draws=True),
        training_implementation_note='Historical trainer hashes are accepted only through the audited Phase 9 AMP overflow compatibility rule.')
    output = root / 'results/baselines/evaluation'
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'protected_artifacts_sha256.json', protected)
    pd.DataFrame(rows).to_csv(output / 'checkpoint_inventory.csv', index=False)
    print('All 25 best checkpoints, completion records, frozen splits and patient partitions verified.')
    return dict(root=root, config=info['preprocessing'], splits=splits, manifest=load_manifest(root / 'results/dataset_statistics.csv'),
        checkpoints=pd.DataFrame(rows), metadata=metadata, protected=protected)


def evaluate_fold(context, name, fold):
    """Run one pass, or reuse a hash-verified completed receipt on reruns.

    A started but incomplete evaluation is deliberately not retried automatically.
    No optimizer or scheduler is constructed; only model weights are consumed.
    """
    root = context['root']
    row = context['checkpoints'].set_index(['Model', 'Fold']).loc[(name, fold)]
    output = root / f'results/baselines/evaluation/{name}/fold_{fold}'
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / 'test_predictions.csv'
    receipt_path = output / 'test_evaluation_receipt.json'
    started_path = output / 'test_evaluation_started.json'
    identity = dict(fold=fold, checkpoint_sha256=row.CheckpointSHA256, test_split_sha256=row.TestSplitSHA256,
                    architecture=name, configuration_fingerprint=context['metadata']['configuration_fingerprint'], preprocessing_version=VERSION,
                    preprocessing_source_sha256={str(p.relative_to(root)): sha256(p) for p in sorted((root/'src/data').glob('*.py'))},
                    inference_source_sha256=sha256(Path(__file__)), batch_size=16, precision='float32')
    require(sha256(row.Checkpoint) == row.CheckpointSHA256, 'Checkpoint changed after preflight')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        require(receipt['identity'] == identity and receipt['status'] == 'complete', 'Cached evaluation protocol mismatch')
        require(sha256(prediction_path) == receipt['predictions_sha256'], 'Cached predictions changed')
        frame = pd.read_csv(prediction_path, dtype={'PID': str, 'sample_id': str}, keep_default_na=False, float_precision='round_trip')
        validate_predictions(frame, context['splits'][fold]['test'], fold)
        print(f'{name}, fold {fold}: verified saved predictions; no repeated inference.')
        return frame
    require(not prediction_path.exists() and not started_path.exists(), 'Incomplete/unreceipted evaluation found; stop for review')
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_baseline(name)
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
    frame = pd.DataFrame(dict(model=name, fold=fold, PID=pids, sample_id=sample_ids, true_label=labels))
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
    print(f'{name}, fold {fold}: one test pass completed; {len(frame)} slices, {frame.PID.nunique()} patients.')
    return frame


