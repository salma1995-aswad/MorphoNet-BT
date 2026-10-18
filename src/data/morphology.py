"""Binary region, one-pixel inner boundary, normalized internal Euclidean depth."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def binary_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.size or not np.isfinite(mask).all():
        raise ValueError('Mask must be a finite nonempty 2-D array')
    values = np.unique(mask)
    if not (np.isin(values, [0, 1]).all() or np.isin(values, [0, 255]).all()):
        raise ValueError(f'Unexpected nonbinary mask values: {values[:10]}')
    result = (mask > 0).astype(np.uint8)
    if not result.any():
        raise ValueError('Empty tumor mask')
    return result


def morphology_targets(mask):
    region = binary_mask(mask).astype(bool)
    # One erosion with an 8-neighbour, 3x3 square; outside image is background.
    eroded = binary_erosion(region, structure=np.ones((3, 3), bool), iterations=1, border_value=0)
    boundary = region & ~eroded
    # Explicit zero padding gives correct distances even when a tumor touches an edge.
    distance = distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
    distance /= distance.max()
    targets = {key: np.ascontiguousarray(value[None], dtype=np.float32)
               for key, value in [('region', region), ('boundary', boundary), ('distance', distance)]}
    validate_targets(targets)
    return targets


def validate_targets(targets, size=None):
    region, boundary, distance = (targets[key] for key in ('region', 'boundary', 'distance'))
    expected = (1, size, size) if size is not None else region.shape
    for value in (region, boundary, distance):
        if value.shape != expected or value.ndim != 3 or value.shape[0] != 1:
            raise ValueError('Invalid morphology shape')
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Nonfinite morphology map or incorrect dtype')
    if not np.isin(region, [0, 1]).all() or not region.any():
        raise ValueError('Invalid region target')
    if not np.isin(boundary, [0, 1]).all() or not boundary.any() or np.any(boundary > region):
        raise ValueError('Invalid inner boundary target')
    if distance.min() < 0 or distance.max() > 1 or not np.isclose(distance.max(), 1):
        raise ValueError('Distance outside [0,1] or missing unit maximum')
    if np.any(distance[region == 0] != 0) or np.any(distance[region == 1] <= 0):
        raise ValueError('Invalid internal distance support')
