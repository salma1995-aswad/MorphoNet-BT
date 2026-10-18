"""Read-only Figshare inventory; no preprocessing, splitting, or training."""
import io
import hashlib
import json
import platform
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

CLASSES = {1: 'Meningioma', 2: 'Glioma', 3: 'Pituitary'}


def environment():
    assert Path(sys.prefix).name.lower() == 'dl', f'Expected dl, got {sys.prefix}'
    info = dict(python_executable=sys.executable, python_version=platform.python_version(),
                environment_prefix=sys.prefix, pytorch_version=torch.__version__,
                pytorch_cuda_version=torch.version.cuda, cuda_available=torch.cuda.is_available())
    if info['cuda_available']:
        info.update(gpu_name=torch.cuda.get_device_name(0),
                    gpu_memory_gib=torch.cuda.get_device_properties(0).total_memory / 2**30)
        info['cuda_tensor_check'] = float((torch.ones(2, device='cuda') + 1).sum().cpu()) == 4
    return info


def discover(root):
    records = []
    for path in sorted(Path(root).rglob('*.mat')):
        records.append(dict(source=str(path.resolve()), member='', sample_id=path.stem))
    for path in sorted(Path(root).rglob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.lower().endswith('.mat'):
                    records.append(dict(source=str(path.resolve()), member=name, sample_id=Path(name).stem))
    return records


def read_record(record):
    if record['member']:
        with zipfile.ZipFile(record['source']) as archive:
            raw = archive.read(record['member'])  # ZIP CRC is verified during read.
    else:
        raw = Path(record['source']).read_bytes()
    stream = io.BytesIO(raw)
    signature_offsets = [0] + [2**power for power in range(9, len(raw).bit_length())]
    if any(raw[offset:offset + 8] == b'\x89HDF\r\n\x1a\n' for offset in signature_offsets):
        stream.seek(0)
        with h5py.File(stream, 'r') as handle:
            if 'cjdata' not in handle:
                return None, {'format': 'HDF5', 'root_keys': list(handle)}
            group = handle['cjdata']
            schema = {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in group.items()}
            # HDF5 dimension order is reversed relative to MATLAB. Apply equally to image/mask.
            fields = {k: np.asarray(v[()]).T for k, v in group.items()}
        return fields, {'format': 'MATLAB v7.3 HDF5', 'fields_on_disk': schema}
    loaded = loadmat(stream, simplify_cells=True)
    if 'cjdata' not in loaded:
        return None, {'format': 'MATLAB legacy', 'root_keys': [k for k in loaded if not k.startswith('__')]}
    fields = {k: np.asarray(v) for k, v in loaded['cjdata'].items()}
    return fields, {'format': 'MATLAB legacy', 'fields_on_disk': {
        k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in fields.items()}}


def decode_pid(value):
    array = np.asarray(value).ravel()
    if array.dtype.kind in 'ui':
        return ''.join(chr(int(x)) for x in array).strip('\x00 ')
    if array.dtype.kind in 'US':
        return ''.join(str(x) for x in array).strip()
    raise ValueError(f'Unsupported PID encoding: {array.dtype}')


def inspect_dataset(records):
    rows, auxiliary, errors, schemas = [], [], [], {}
    for record in records:
        try:
            fields, schema = read_record(record)
            if fields is None:
                auxiliary.append({**record, **schema})
                continue
            key = json.dumps(schema, sort_keys=True)
            schemas.setdefault(key, dict(schema=schema, count=0, example=record))['count'] += 1
            row = {**record, 'fields': ','.join(sorted(fields))}
            for name in ('PID', 'label', 'image', 'tumorMask', 'tumorBorder'):
                row[f'{name}_missing'] = name not in fields
            row['PID'] = decode_pid(fields['PID']) if 'PID' in fields else ''
            lab = np.asarray(fields.get('label', [])).ravel()
            row['raw_label'] = float(lab[0]) if lab.size == 1 else np.nan
            row['valid_label'] = row['raw_label'] in CLASSES
            row['class_name'] = CLASSES.get(row['raw_label'], 'INVALID')
            image = np.asarray(fields.get('image', []))
            row.update(image_shape=str(image.shape), image_dtype=str(image.dtype),
                       image_valid=image.ndim == 2 and image.size > 0 and bool(np.isfinite(image).all()))
            if image.size:
                row.update(image_min=float(image.min()), image_max=float(image.max()),
                           image_mean=float(image.mean()), image_std=float(image.std()),
                           image_p01=float(np.percentile(image, 1)), image_p99=float(np.percentile(image, 99)),
                           image_sha256=hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest())
            mask = np.asarray(fields.get('tumorMask', []))
            row.update(mask_shape=str(mask.shape), mask_dtype=str(mask.dtype),
                       mask_empty=not bool(np.any(mask)), mask_binary=bool(mask.size and np.isin(mask, [0, 1]).all()),
                       mask_shape_matches=bool(mask.size and mask.shape == image.shape), mask_pixels=int(np.count_nonzero(mask)))
            border = np.asarray(fields.get('tumorBorder', [])).ravel()
            row.update(border_values=int(border.size), border_empty=border.size == 0,
                       border_dtype=str(border.dtype), border_finite=bool(np.isfinite(border).all()),
                       border_even=border.size % 2 == 0)
            if border.size and border.size % 2 == 0 and image.ndim == 2:
                xy = border.reshape(-1, 2)
                row['border_in_bounds'] = bool(((xy[:, 0] >= 1) & (xy[:, 0] <= image.shape[1]) &
                                               (xy[:, 1] >= 1) & (xy[:, 1] <= image.shape[0])).all())
            else:
                row['border_in_bounds'] = False
            rows.append(row)
        except Exception as exc:
            errors.append({**record, 'error': f'{type(exc).__name__}: {exc}'})
    return pd.DataFrame(rows), auxiliary, errors, list(schemas.values())


def summarize(frame, records, auxiliary, errors):
    valid = frame[frame.valid_label & frame.PID.ne('')]
    slices = valid.groupby('PID').size()
    inconsistent = valid.groupby('PID').raw_label.nunique()
    hashes = valid.groupby('image_sha256').agg(samples=('PID', 'size'), patients=('PID', 'nunique'))
    return dict(discovered_mat_files=len(records), auxiliary_files=len(auxiliary), readable_samples=len(frame),
                unique_patients=int(valid.PID.nunique()), images_per_class=valid.class_name.value_counts().to_dict(),
                patients_per_class=valid.groupby('class_name').PID.nunique().to_dict(),
                slices_per_patient=slices.describe().to_dict(), missing_masks=int(frame.tumorMask_missing.sum()),
                empty_masks=int(frame.mask_empty.sum()), nonbinary_masks=int((~frame.mask_binary).sum()),
                mask_shape_mismatches=int((~frame.mask_shape_matches).sum()),
                missing_borders=int(frame.tumorBorder_missing.sum()), empty_borders=int(frame.border_empty.sum()),
                invalid_borders=int((~(frame.border_even & frame.border_finite & frame.border_in_bounds)).sum()),
                unreadable_files=len(errors), invalid_labels=int((~frame.valid_label).sum()),
                invalid_images=int((~frame.image_valid).sum()), missing_pids=int(frame.PID.eq('').sum()),
                inconsistent_pids=inconsistent[inconsistent > 1].index.tolist(),
                duplicate_image_groups=int((hashes.samples > 1).sum()),
                cross_patient_duplicate_image_groups=int((hashes.patients > 1).sum()),
                duplicate_sample_ids=int(frame.sample_id.duplicated().sum()),
                image_shapes=frame.image_shape.value_counts().to_dict(), image_dtypes=frame.image_dtype.value_counts().to_dict(),
                global_intensity_range=[float(frame.image_min.min()), float(frame.image_max.max())])
