"""Phase 3 numerical audits; no model, optimization, or split mutation."""
import json
from copy import deepcopy
import cv2
import numpy as np
import pandas as pd
from .inspection import read_record
from .dataset import load_checked_sample
from .preprocessing import DEFAULT_CONFIG, normalize_image, resize_pair, preprocess, affine_pair
from .morphology import binary_mask, morphology_targets, validate_targets


def scan_intensities(frame, output):
    rows = []
    for i, record in enumerate(frame.to_dict('records')):
        try:
            image, mask = load_checked_sample(record)
            image, mask = np.asarray(image), binary_mask(mask)
            if image.shape != mask.shape:
                raise ValueError('Original MRI-mask dimensions differ')
            percentiles = np.percentile(image, [1, 50, 95, 99, 99.5, 99.9])
            resized, _ = resize_pair(image, mask)
            row = {k: record[k] for k in ('sample_id', 'PID', 'class_name', 'original_label')}
            row.update(native_height=image.shape[0], native_width=image.shape[1], minimum=float(image.min()),
                       maximum=float(image.max()), mean=float(image.mean()), std=float(image.std()),
                       zero_fraction=float((image == 0).mean()),
                       **dict(zip(['p01', 'p50', 'p95', 'p99', 'p995', 'p999'], map(float, percentiles))))
            row['max_over_p99'] = row['maximum'] / row['p99'] if row['p99'] else np.nan
            row['max_over_p999'] = row['maximum'] / row['p999'] if row['p999'] else np.nan
            row['fraction_above_p999'] = float((image > row['p999']).mean())
            row['fraction_above_10000'] = float((image > 10000).mean())
            row['p999_excess_variance_fraction'] = float(np.square(np.maximum(image.astype(float) - row['p999'], 0)).sum()
                                                         / np.square(image.astype(float) - row['mean']).sum())
            for method in ('minmax', 'zscore'):
                normalized = normalize_image(resized, method)
                row.update({f'{method}_{name}': float(value) for name, value in {
                    'min': normalized.min(), 'max': normalized.max(), 'mean': normalized.mean(),
                    'std': normalized.std(), 'p99': np.percentile(normalized, 99)}.items()})
            rows.append(row)
        except Exception as exc:
            _save_failure(output, record, 'intensity_scan', exc)
            raise
        if (i + 1) % 500 == 0:
            print(f'Intensity audit: {i + 1}/{len(frame)}')
            (output / 'progress.json').write_text(json.dumps({'stage': 'intensity', 'completed': i + 1, 'total': len(frame)}))
    table = pd.DataFrame(rows)
    table.to_csv(output / 'intensity_statistics.csv', index=False)
    return table


def _save_failure(output, record, stage, exc):
    pd.DataFrame([{k: record[k] for k in ('sample_id', 'PID', 'class_name', 'source', 'member')} |
                  {'stage': stage, 'failure': f'{type(exc).__name__}: {exc}'}]).to_csv(output / 'audit_failures.csv', index=False)


def audit_pipeline(frame, config, output, train_sample_ids):
    rows = []
    for i, record in enumerate(frame.to_dict('records')):
        try:
            image, mask = load_checked_sample(record)
            deterministic, _ = preprocess(image, mask, mode='test', config=config)
            augmentation_tested = record['sample_id'] in train_sample_ids
            augmented, metadata = preprocess(image, mask, mode='train' if augmentation_tested else 'test', config=config,
                                              rng=np.random.default_rng(np.random.SeedSequence([42, int(record['sample_id'])])))
            for arrays in (deterministic, augmented):
                validate_targets({k: arrays[k] for k in ('region', 'boundary', 'distance')}, size=256)
                if arrays['image'].shape != (1, 256, 256) or arrays['image'].dtype != np.float32 or not np.isfinite(arrays['image']).all():
                    raise ValueError('Invalid image tensor')
            r, b, d = (deterministic[k][0] for k in ('region', 'boundary', 'distance'))
            row = {k: record[k] for k in ('sample_id', 'PID', 'class_name')}
            row.update(native_height=image.shape[0], region_pixels=int(r.sum()), boundary_pixels=int(b.sum()),
                       boundary_fraction=float(b.sum() / r.sum()), distance_min=float(d.min()), distance_max=float(d.max()),
                       augmentation_tested=augmentation_tested,
                       geometry_attempts=metadata['attempts'], geometry_identity_fallback=metadata['fallback_identity'],
                       augmented_region_pixels=int(augmented['region'].sum()),
                       augmented_boundary_pixels=int(augmented['boundary'].sum()), status='PASS')
            rows.append(row)
        except Exception as exc:
            _save_failure(output, record, 'morphology_or_augmentation', exc)
            raise
        if (i + 1) % 500 == 0:
            print(f'Preprocessing and target audit: {i + 1}/{len(frame)}')
            (output / 'progress.json').write_text(json.dumps({'stage': 'morphology', 'completed': i + 1, 'total': len(frame)}))
    result = pd.DataFrame(rows)
    result.to_csv(output / 'morphology_integrity.csv', index=False)
    pd.DataFrame(columns=['sample_id', 'PID', 'class_name', 'source', 'member', 'stage', 'failure']).to_csv(output / 'audit_failures.csv', index=False)
    return result


def numerical_checks(config):
    checks = {}
    mask = np.zeros((256, 256), np.uint8)
    mask[70:180, 90:160] = 1
    image = mask.astype(np.float32)
    resized_image, resized_mask = resize_pair(image, mask)
    checks['256_no_interpolation'] = np.array_equal(image, resized_image) and np.array_equal(mask, resized_mask)
    shifted_image, shifted_mask = affine_pair(image, mask, np.array([[1., 0., 7.], [0., 1., -4.]]))
    checks['exact_integer_translation_alignment'] = np.array_equal(shifted_image, shifted_mask)
    matrix = cv2.getRotationMatrix2D((127.5, 127.5), 9., 1.03)
    transformed_image, transformed_mask = affine_pair(image, mask, matrix)
    thresholded = transformed_image >= .5
    dice = 2 * np.logical_and(thresholded, transformed_mask).sum() / (thresholded.sum() + transformed_mask.sum())
    checks['nontrivial_affine_alignment_dice'] = float(dice)
    checks['nontrivial_affine_alignment'] = bool(dice > .99)
    large = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    small_image, small_mask = resize_pair(large.astype(np.float32), large)
    checks['resize_geometry_alignment'] = bool(np.array_equal(small_image >= .5, small_mask))
    checks['mask_255_conversion'] = np.array_equal(binary_mask(mask * 255), mask)
    for name, edge_mask in [('single_pixel', np.pad(np.ones((1, 1), np.uint8), ((0, 255), (0, 255)))),
                            ('full_image', np.ones((256, 256), np.uint8)), ('normal_region', mask)]:
        validate_targets(morphology_targets(edge_mask), size=256)
        checks[f'morphology_{name}'] = True
    for name, bad_mask in [('empty', np.zeros((256, 256))), ('nonbinary', mask.astype(float) * .5)]:
        try:
            morphology_targets(bad_mask)
        except ValueError:
            checks[f'reject_{name}_mask'] = True
        else:
            raise AssertionError(f'{name} mask was not rejected')
    try:
        normalize_image(np.ones((256, 256)))
    except ValueError:
        checks['reject_constant_mri'] = True
    else:
        raise AssertionError('Constant MRI was not rejected')
    try:
        preprocess(image, mask[:128], config=config)
    except ValueError:
        checks['reject_shape_mismatch'] = True
    else:
        raise AssertionError('Shape mismatch was not rejected')
    no_intensity = deepcopy(config)
    no_intensity['augmentation']['gain_probability'] = 0.
    no_intensity['augmentation']['gaussian_noise']['probability'] = 0.
    with_noise, _ = preprocess(image, mask, 'train', config, np.random.default_rng(42))
    without_noise, _ = preprocess(image, mask, 'train', no_intensity, np.random.default_rng(42))
    checks['photometry_does_not_change_targets'] = all(np.array_equal(with_noise[k], without_noise[k]) for k in ('region', 'boundary', 'distance'))
    for mode in ('val', 'test'):
        first, _ = preprocess(image, mask, mode, config, np.random.default_rng(1))
        second, _ = preprocess(image, mask, mode, config, np.random.default_rng(999))
        checks[f'{mode}_deterministic'] = all(np.array_equal(first[k], second[k]) for k in first)
    image_only, _ = preprocess(image, mode='test', config=config)
    checks['mask_free_preprocessing'] = set(image_only) == {'image'}
    if not all(value for key, value in checks.items() if key != 'nontrivial_affine_alignment_dice'):
        raise AssertionError(checks)
    return checks


def border_sanity(record):
    fields, _ = read_record(record)
    mask = binary_mask(fields['tumorMask'])
    points = fields['tumorBorder'].ravel().reshape(-1, 2) - 1
    polygon = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(polygon, [np.rint(points).astype(np.int32)], 1)
    return float(2 * np.logical_and(mask, polygon).sum() / (mask.sum() + polygon.sum()))
