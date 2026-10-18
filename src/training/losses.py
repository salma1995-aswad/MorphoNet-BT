"""Stable float32 loss arithmetic with explicitly separated supervised tasks."""
import torch
from torch import nn
from torch.nn import functional as F


class MorphoNetLoss(nn.Module):
    def __init__(self, lambda_region=1., lambda_boundary=1., lambda_distance=.5,
                 lambda_morphology=.3, class_weights=None, epsilon=1e-6):
        super().__init__()
        self.weights = dict(lambda_region=lambda_region, lambda_boundary=lambda_boundary,
                            lambda_distance=lambda_distance, lambda_morphology=lambda_morphology)
        if min(self.weights.values()) < 0 or epsilon <= 0:
            raise ValueError('Loss weights must be nonnegative and epsilon positive')
        weight = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        if weight is not None and (weight.shape != (3,) or not torch.isfinite(weight).all() or not (weight > 0).all()):
            raise ValueError('Expected three finite positive class weights')
        # Optional weighting uses a mean of weighted per-sample CE; epoch means are sample-weighted.
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction='none')
        self.epsilon = epsilon

    def _segmentation(self, logits, target):
        logits, target = logits.float(), target.float()
        if logits.shape != target.shape:
            raise ValueError('Morphology output and target shapes differ')
        probabilities = logits.sigmoid()
        axes = (1, 2, 3)
        dice = 1 - ((2 * (probabilities*target).sum(axes) + self.epsilon) /
                    (probabilities.sum(axes) + target.sum(axes) + self.epsilon)).mean()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return bce, dice

    def forward(self, outputs, batch):
        cls = self.cross_entropy(outputs['logits'].float(), batch['label']).mean()
        rb, rd = self._segmentation(outputs['region_logits'], batch['region'])
        bb, bd = self._segmentation(outputs['boundary_logits'], batch['boundary'])
        if outputs['distance'].shape != batch['distance'].shape:
            raise ValueError('Distance shape mismatch')
        distance = F.smooth_l1_loss(outputs['distance'].float(), batch['distance'].float())
        region, boundary = rb + rd, bb + bd
        w = self.weights
        morphology = w['lambda_region']*region + w['lambda_boundary']*boundary + w['lambda_distance']*distance
        weighted = w['lambda_morphology']*morphology
        result = dict(classification_loss=cls, region_bce=rb, region_dice=rd, region_loss=region,
                      boundary_bce=bb, boundary_dice=bd, boundary_loss=boundary, distance_loss=distance,
                      morphology_loss=morphology, weighted_morphology_loss=weighted, total_loss=cls+weighted)
        if not all(torch.isfinite(v).all() for v in result.values()):
            raise FloatingPointError('Nonfinite loss; stop instead of changing loss weights')
        return result


def train_only_class_weights(train_frame, mode=None):
    if mode is None:
        return None
    if mode != 'balanced_train' or not train_frame['split'].eq('train').all():
        raise ValueError('Only balanced_train weighting on TRAIN metadata is supported')
    counts = train_frame.original_label.value_counts().reindex([1, 2, 3], fill_value=0)
    if (counts == 0).any():
        raise ValueError('TRAIN is missing a class')
    return (len(train_frame)/(3*counts)).tolist()
