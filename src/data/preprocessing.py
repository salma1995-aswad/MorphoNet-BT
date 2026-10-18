"""CPU grayscale preprocessing; shared MRI/mask geometry, image-only photometry."""
from copy import deepcopy
import cv2
import numpy as np
from .morphology import binary_mask, morphology_targets

DEFAULT_CONFIG = {
    'seed': 42, 'image_size': 256,
    'normalization': {'method': 'zscore', 'scope': 'per_image_all_pixels', 'epsilon': 1e-8, 'clipping': False},
    'augmentation': {
        'enabled': True, 'rotation_degrees': 10.0, 'translate': 0.02, 'scale': [0.95, 1.05],
        'geometry_probability': 0.8, 'gain_range': [0.95, 1.05], 'gain_probability': 0.5,
        'gaussian_noise': {'std': 0.01, 'probability': 0.3, 'units': 'normalized_image'},
        'horizontal_flip': False, 'padding': 'constant_zero',
        'max_geometry_attempts': 20, 'reject_tumor_clipping': True},
    'morphology': {'boundary_method': 'region_minus_erosion', 'kernel_size': 3,
                   'erosion_iterations': 1, 'distance_method': 'internal_euclidean',
                   'distance_normalization': 'per_mask_max', 'distance_padding': 1},
    'label_mapping': {1: {'class_name': 'Meningioma', 'class_index': 0},
                      2: {'class_name': 'Glioma', 'class_index': 1},
                      3: {'class_name': 'Pituitary', 'class_index': 2}},
}


def validate_config(config):
    if config['image_size'] != 256:
        raise ValueError('Phase 3 requires 256x256 output')
    norm = config['normalization']
    if norm['method'] not in ('minmax', 'zscore') or norm['scope'] != 'per_image_all_pixels' or norm['clipping']:
        raise ValueError('Unsupported normalization configuration')
    if not 0 < norm['epsilon'] < 1e-3:
        raise ValueError('Invalid normalization epsilon')
    aug = config['augmentation']
    if aug['horizontal_flip'] or aug['padding'] != 'constant_zero':
        raise ValueError('Horizontal flip is disabled; only constant zero padding is supported')
    if not (0 <= aug['rotation_degrees'] <= 10 and 0 <= aug['translate'] <= .02):
        raise ValueError('Geometry exceeds approved conservative limits')
    if not (.95 <= aug['scale'][0] <= aug['scale'][1] <= 1.05):
        raise ValueError('Invalid scale range')
    if not (.95 <= aug['gain_range'][0] <= aug['gain_range'][1] <= 1.05):
        raise ValueError('Invalid gain range')
    for probability in (aug['geometry_probability'], aug['gain_probability'], aug['gaussian_noise']['probability']):
        if not 0 <= probability <= 1:
            raise ValueError('Probability outside [0,1]')
    if not 0 <= aug['gaussian_noise']['std'] <= .01 or aug['gaussian_noise']['units'] != 'normalized_image':
        raise ValueError('Invalid normalized noise setting')
    if not aug['reject_tumor_clipping'] or aug['max_geometry_attempts'] < 1:
        raise ValueError('Tumor-clipping protection is required')
    if config['morphology'] != DEFAULT_CONFIG['morphology']:
        raise ValueError('Unsupported morphology configuration; implementation must match documentation')
    if config['label_mapping'] != DEFAULT_CONFIG['label_mapping']:
        raise ValueError('Label mapping differs from explicit canonical mapping')


def normalize_image(image, method='zscore', epsilon=1e-8):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError('MRI must be a finite 2-D array')
    if method == 'minmax':
        offset, denominator = float(image.min()), float(image.max() - image.min())
    elif method == 'zscore':
        offset, denominator = float(image.mean(dtype=np.float64)), float(image.std(dtype=np.float64, ddof=0))
    else:
        raise ValueError(f'Unknown normalization: {method}')
    if denominator <= epsilon:
        raise ValueError('Constant/degenerate MRI; normalization is not meaningful')
    result = np.ascontiguousarray((image - offset) / denominator, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite normalized MRI')
    return result


def resize_pair(image, mask, size=256):
    image = np.asarray(image, dtype=np.float32)
    mask = binary_mask(mask)
    if image.shape != mask.shape or image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError('MRI and mask cannot be aligned')
    if image.shape != (size, size):
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        # NEAREST_EXACT uses the same pixel-centre resize convention as bilinear.
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST_EXACT)
    return np.ascontiguousarray(image, dtype=np.float32), binary_mask(mask)


def affine_pair(image, mask, matrix):
    if image.shape != mask.shape:
        raise ValueError('MRI-mask shape mismatch before affine transform')
    height, width = image.shape
    out_image = cv2.warpAffine(np.float32(image), matrix, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out_mask = cv2.warpAffine(binary_mask(mask), matrix, (width, height), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out_image, binary_mask(out_mask)


def augment_geometry(image, mask, rng, settings):
    height, width = image.shape
    identity = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float64)
    if rng.random() >= settings['geometry_probability']:
        return image, mask, {'matrix': identity.tolist(), 'attempts': 0, 'fallback_identity': False}
    ys, xs = np.nonzero(mask)
    # Extrema of an affine transform occur on the convex hull of the foreground.
    points = cv2.convexHull(np.column_stack([xs, ys]).astype(np.float32)).reshape(-1, 2)
    for attempt in range(1, settings['max_geometry_attempts'] + 1):
        angle = float(rng.uniform(-settings['rotation_degrees'], settings['rotation_degrees']))
        scale = float(rng.uniform(*settings['scale']))
        matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, scale)
        matrix[:, 2] += rng.uniform(-settings['translate'], settings['translate'], 2) * [width, height]
        transformed = points @ matrix[:, :2].T + matrix[:, 2]
        if (transformed[:, 0].min() < 0 or transformed[:, 0].max() > width - 1 or
                transformed[:, 1].min() < 0 or transformed[:, 1].max() > height - 1):
            continue
        out_image, out_mask = affine_pair(image, mask, matrix)
        return out_image, out_mask, {'matrix': matrix.tolist(), 'attempts': attempt, 'fallback_identity': False}
    # This is an explicit conservative policy, not a repair for malformed data.
    return image, mask, {'matrix': identity.tolist(), 'attempts': settings['max_geometry_attempts'], 'fallback_identity': True}


def preprocess(image, mask=None, mode='test', config=None, rng=None):
    config = deepcopy(DEFAULT_CONFIG if config is None else config)
    validate_config(config)
    if mode not in ('train', 'val', 'test'):
        raise ValueError('Mode must be train, val, or test')
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError('Invalid MRI array')
    if mask is not None:
        mask = binary_mask(mask)
        if image.shape != mask.shape:
            raise ValueError('Original MRI-mask dimension mismatch')
    active = mode == 'train' and config['augmentation']['enabled']
    metadata = {'augmentation_active': active, 'attempts': 0, 'fallback_identity': False}
    if active:
        if mask is None or rng is None:
            raise ValueError('Training augmentation needs a mask and explicit random generator')
        image, mask, geometry = augment_geometry(image, mask, rng, config['augmentation'])
        metadata.update(geometry)
    size = config['image_size']
    if mask is None:
        if image.shape != (size, size):
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        image, mask = resize_pair(image, mask, size)
    image = normalize_image(image, config['normalization']['method'], config['normalization']['epsilon'])
    # Post-normalization gain/noise are not cancelled by per-image normalization.
    if active:
        aug = config['augmentation']
        gain = float(rng.uniform(*aug['gain_range'])) if rng.random() < aug['gain_probability'] else 1.
        image *= gain
        noise_applied = rng.random() < aug['gaussian_noise']['probability']
        if noise_applied:
            image += rng.normal(0, aug['gaussian_noise']['std'], image.shape).astype(np.float32)
        metadata.update(gain=gain, noise_applied=bool(noise_applied))
    result = {'image': np.ascontiguousarray(image[None], dtype=np.float32)}
    if result['image'].shape != (1, size, size) or not np.isfinite(result['image']).all():
        raise ValueError('Invalid preprocessed image')
    if mask is not None:
        result.update(morphology_targets(mask))
    return result, metadata
