# Frozen scientific protocol: static consistency record

Authority is the frozen source/configuration and recorded aggregate results, not runtime experiments in Phase18. The inspected static tests check source/configuration contracts; tests/data/scientific_sha256.json verifies frozen scientific bytes.

- Cohort:3,064 CE-MRI slices,233 patients, three classes. Meningioma708/82; Glioma1426/89; Pituitary930/62.
- Splits: StratifiedGroupKFold five outer folds, shuffle=True, seed42, groups=PID; inner patient-level stratified20% holdout with seed42. Original saved records are authoritative.
- MRI resize256×256 bilinear, masks nearest-neighbor exact; skip resize if already256. Z-score over all pixels, ddof0, epsilon1e-8; no clipping. Validation/Test deterministic.
- Train geometry probability0.8, rotation±10°, translation±2%, scale0.95–1.05; reject tumor-hull clipping, at most20 attempts then identity. No flips. Post-normalization MRI gain0.95–1.05 probability0.5; Gaussian noise std0.01 probability0.3.
- Region=final mask; boundary=mask minus one3×3 erosion; internal EDT with one-pixel zero padding, per-mask max normalization, zero outside. Loss CE+0.3×(region BCE+Dice + boundary BCE+Dice +0.5×SmoothL1 distance); effective coefficients0.3/0.3/0.15, float32 loss arithmetic, Dice epsilon1e-6.
- A4 uses MSS/MFE, supervised morphology, predicted-boundary BAA, global/local branches and .5*(global+local), no AMGF. Boundary predictor runs when BAA is active, including return_aux=False; forward accepts MRI only. Original default.yaml describes A5; build_ablation applies variant flags.
- AdamW LR0.001, weight_decay0.0001; max100 epochs, batch32, workers2, persistent_workers=False; cosine T_max100/eta_min0; early stopping validation slice macro-F1, patience15/min_delta0. Seed42. CUDA AMP initial scale1024; no clipping/accumulation/class weights.
- Five baselines initialized with weights=None; one-channel input, three classes, CE only. Internal masks preserve identical preprocessing augmentation and are removed before baseline model/loss use. This compares training configurations, not architecture alone.
- A0 minimal; A1 adds MSS; A2 morphology supervision; A3 BAA; A4 dual paths with mean fusion; A5 AMGF. A5 reuses original full-model checkpoints. Sequential contrasts are not isolated component causality.
- Patient probabilities are equal-weight means of slice softmax probabilities in float64, followed by argmax. Patients receive equal weight in patient metrics; pooled slice results weight slices. Fold mean/SD differs from pooled metrics.
- Phase14: exact two-sided McNemar on patient correctness; one primary Holm family of10 unique comparisons (A4/A5, five baselines, four earlier adjacent transitions; A5/A4 is the repeated reverse). Bootstrap10,000, seed42; paired patient draws; slice-level intervals preserve patient clusters. Meningioma82-patient comparisons use a separate exploratory Holm family. Intervals are marginal percentile intervals, not multiplicity-adjusted or retraining uncertainty.
- Grad-CAM primary stage3 maps16×16, resized for comparison; BAA is feature modulation, not an interchangeable attribution. Auxiliary region/boundary/distance metrics have modest quality. Occlusion compares annotated tumor replacement with same-shape/area translated non-tumor controls in an eroded head-interior proxy, not anatomical segmentation; up to10 controls, shared zero/local-median locations, no renormalization after replacement. Primary true-class-logit drop contrast aggregates controls within slice then slices within patient;10,000 patient bootstrap draws, seed42; no formal occlusion p-values or causal claim. Paired eligibility3,049 slices/all233 patients;15 excluded (1meningioma,14glioma).

## Historical discrepancies retained

The original main/ablation trainer aborts nonfinite gradients. The later baseline repair lets enabled GradScaler skip overflow updates; EfficientNet-B0 folds2–5 used that repaired trainer, with1/0/2/1 skipped steps. Existing completed baseline runs were retained. This difference is documented, not silently standardized. Historical notebook Markdown describes preparation-time state; later execution counts/gates sometimes differed. Public counts/outputs are cleared and gates reset, without claiming a rerun. Historical architecture counts579,599 refer to A5, not A4's575,359.

A4 was selected after examining outer Test results; all comparisons are exploratory. External validation, repeat-training variability and prospective clinical evaluation remain absent.


Release clarification: region and distance auxiliary outputs are conditionally skipped during classification inference. The predicted boundary remains active with BAA. Parameter totals include auxiliary heads; Conv/Linear MACs exclude inactive auxiliary computations and non-Conv/Linear operations. See [architecture](architecture.md) and [artifact availability](artifact_availability.md).
