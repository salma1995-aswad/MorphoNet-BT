"""Frozen protocol, fail-closed fingerprints and opaque protected-file hashing."""
from pathlib import Path
import hashlib
import json
import platform
import pandas as pd
import torch
import torchvision
import numpy as np
import yaml
from src.models.ablation import VARIANTS
from src.data.dataset import verify_frozen_splits
from src.data.kfold import audit_splits, reload_splits, load_manifest

TRAIN_VARIANTS=('A0','A1','A2','A3','A4')
CLASS_MAPPING={0:'Meningioma',1:'Glioma',2:'Pituitary'}
VERSION='MorphoNet-BT Preprocessing v1.0'
SOURCES=['src/ablation/protocol.py','src/ablation/data.py','src/ablation/training.py','src/ablation/validation.py',
 'src/models/ablation.py','src/models/morphonet_bt.py','src/models/blocks.py','src/models/attention.py','src/models/fusion.py',
 'src/training/losses.py','src/training/scheduler.py','src/utils/checkpoint.py','src/utils/training_metrics.py',
 'src/data/dataset.py','src/data/preprocessing.py','src/data/morphology.py','src/data/inspection.py']

def require(value,message):
    if not value: raise ValueError(message)

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def object_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def preflight(root):
    root=Path(root).resolve()
    setup=json.loads((root/'results/ablation/setup/ablation_setup_metadata.json').read_text())
    for name,digest in setup['sources'].items():require(file_hash(root/name)==digest,f'Phase 11 source/config changed: {name}')
    cfg=yaml.safe_load((root/'configs/ablation.yaml').read_text())
    main=yaml.safe_load((root/'configs/default.yaml').read_text())
    require(cfg['variants']==VARIANTS and tuple(cfg['train_variants'])==TRAIN_VARIANTS,'Variant definitions changed')
    require(cfg['model']==main['model'],'Architecture configuration changed')
    expected={**main['training'],'scheduler':{**main['training']['scheduler'],'T_max':100}}
    require(cfg['training']==expected,'Primary training protocol mismatch')
    require(expected['max_epochs']==100 and expected['batch_size']==32 and expected['num_workers']==2 and not expected['persistent_workers'],'Frozen loader/epoch settings changed')
    require(expected['optimizer']==dict(name='AdamW',learning_rate=.001,weight_decay=.0001),'Optimizer changed')
    require(expected['early_stopping']==dict(monitor='val_macro_f1',mode='max',patience=15,min_delta=0.),'Early stopping changed')
    require(expected['loss']==dict(lambda_region=1.,lambda_boundary=1.,lambda_distance=.5,lambda_morphology=.3),'Loss weights changed')
    require(expected['class_weights'] is None and expected['max_grad_norm'] is None and expected['amp'] is True,'Loss/gradient/AMP policy changed')
    require(cfg['seed']==42 and cfg['class_mapping']==CLASS_MAPPING,'Seed/class mapping changed')
    preprocessing=yaml.safe_load((root/'results/architecture/preprocessing_v1_approved.yaml').read_text())
    require(cfg['preprocessing']==preprocessing=={k:main[k] for k in preprocessing},'Preprocessing changed')
    frozen=json.loads((root/'results/architecture/frozen_input_hashes.json').read_text())
    hashes=verify_frozen_splits(root);require(hashes==frozen['splits'],'Frozen split mismatch')
    for name,digest in frozen['preprocessing_sources'].items():require(file_hash(root/name)==digest,'Preprocessing source changed')
    # Test CSV identities only: no image loader or evaluation result parsing.
    audits,coverage=audit_splits(reload_splits(root/'splits'),load_manifest(root/'results/dataset_statistics.csv'))
    memory=pd.read_csv(root/'results/ablation/setup/ablation_memory_preflight.csv')
    require(set(memory.Variant)==set(VARIANTS) and memory.BatchSize.eq(32).all() and memory.Status.eq('PASS').all(),'Phase 11 readiness missing')
    contract=dict(seed=42,training=expected,preprocessing=preprocessing,class_mapping=CLASS_MAPPING,split_hashes=hashes,
        amp_dtype='float16',gradient_accumulation=False,gradient_clipping=False,numerical_policy='Abort on nonfinite loss/logits/gradients, matching frozen main model',
        augmentation_epoch_index='zero-based epoch-1, matching frozen main model',test_lock=True)
    runtime=dict(python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,cuda=torch.version.cuda,
        GPU=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    return dict(root=root,config=cfg,preprocessing=preprocessing,split_hashes=hashes,split_audit=audits,coverage=coverage,memory=memory,
        contract=contract,configuration_fingerprint=object_hash(contract),sources={p:file_hash(root/p) for p in SOURCES},runtime=runtime,test_lock=True)

def validate_selection(variants,folds):
    if 'A5' in variants:raise ValueError('A5 is the frozen full-model reference and must not be retrained')
    require(variants and len(variants)==len(set(variants)) and all(v in TRAIN_VARIANTS for v in variants),'Select unique A0-A4 variants')
    require(folds and len(folds)==len(set(folds)) and all(type(f) is int and 1<=f<=5 for f in folds),'Select unique integer folds 1-5')

def run_identity(info,variant,fold):
    validate_selection([variant],[fold])
    return dict(variant=variant,fold=fold,flags=VARIANTS[variant],loss_mode='frozen_morphology' if VARIANTS[variant]['use_morphology_supervision'] else 'CE_only',
        configuration_fingerprint=info['configuration_fingerprint'],sources=info['sources'],runtime=info['runtime'],split_hashes=info['split_hashes'])

def protected_inventory(root):
    root=Path(root);paths=set()
    # Evaluation artifacts are opaque bytes here, never parsed for performance.
    for folder in ['checkpoints/main','checkpoints/baselines','results/main','results/baselines','results/ablation/setup','results/architecture','results/complexity','splits','logs/main']:
        for p in (root/folder).rglob('*'):
            if p.is_file():paths.add(p)
    paths.update(root/p for p in ['configs/default.yaml','configs/baselines.yaml','configs/ablation.yaml',*SOURCES])
    return {p.relative_to(root).as_posix():file_hash(p) for p in sorted(paths)}

def check_protection(root,manifest=None):
    root=Path(root)
    if manifest is None:manifest=json.loads((root/'results/ablation/training_setup/protected_artifacts.json').read_text())
    for p,digest in manifest.items():require(file_hash(root/p)==digest,f'Protected artifact changed: {p}')
    return len(manifest)

def require_prepared(info):
    audit=json.loads((info['root']/'results/ablation/training_setup/preparation_audit.json').read_text())
    require(audit['configuration_fingerprint']==info['configuration_fingerprint'] and audit['sources']==info['sources'],'Prepared protocol/source fingerprint mismatch')
    require(audit['runtime']==info['runtime'] and audit['passed'] is True,'Preparation runtime/audit mismatch')
    require(info['test_lock'] is True,'Test lock must be active')
    return check_protection(info['root'])
