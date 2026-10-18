"""Epoch-wise cosine schedule and validation-only early stopping."""
from dataclasses import dataclass, asdict
import math
from torch.optim.lr_scheduler import CosineAnnealingLR


def build_scheduler(optimizer, training_config):
    config = training_config['scheduler']
    if config['name'] != 'CosineAnnealingLR':
        raise ValueError('Only the configured cosine schedule is supported')
    return CosineAnnealingLR(optimizer, T_max=training_config['max_epochs'], eta_min=config.get('eta_min', 0.))


@dataclass
class EarlyStopping:
    patience: int = 15
    min_delta: float = 0.
    best_metric: float = -math.inf
    reference_metric: float = -math.inf
    best_epoch: int = 0
    bad_epochs: int = 0

    def update(self, metric, epoch):
        if not math.isfinite(metric):
            raise FloatingPointError('Validation Macro-F1 is nonfinite')
        is_best = metric > self.best_metric
        if is_best:
            self.best_metric, self.best_epoch = float(metric), epoch
        if metric > self.reference_metric + self.min_delta:
            self.reference_metric, self.bad_epochs = float(metric), 0
        else:
            self.bad_epochs += 1
        return is_best

    @property
    def stopped(self):
        return self.bad_epochs >= self.patience

    def state_dict(self):
        return asdict(self)

    def load_state_dict(self, state):
        for key, value in state.items():
            if key not in asdict(self):
                raise ValueError(f'Unknown early-stopping state: {key}')
            setattr(self, key, value)
