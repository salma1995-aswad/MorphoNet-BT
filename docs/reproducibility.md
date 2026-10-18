# Reproducibility guide

## Three different activities

1. **Inspect reported results:** read the included CSVs and diagram; no data/model execution needed. Their SHA-256 provenance is recorded.
2. **Recompute from saved artifacts:** statistics need original aligned sample/patient predictions and immutable protocol metadata. These private artifacts are not included. Aggregate tables cannot recreate paired tests or bootstrap draws.
3. **Run inference or retrain:** requires authorized data, checkpoints (for inference/resume), canonical manifests/splits, matching code/config fingerprints and environment. New training is a new experiment; it cannot establish that the frozen run has been reproduced merely by using seed42.

## Retained workflows

See [the notebook guide](../notebooks/README.md) for the complete 17-notebook matrix, inputs and A4/A5 identities. Original A5 stages 04-07 support the reused comparator and prerequisite fingerprints; A4 uses the controlled-ablation route. Historical snapshots and exported diagnostic tables are not part of the cleaned public package.

## Execution gates and workspace

Run from the repository root, or explicitly set MORPHONET_ROOT. It points to a private experiment root with src/configs/notebooks and outputs under results/checkpoints/logs; no separate output relocation was introduced. Set MORPHONET_DATA_ROOT for discovery; No dotenv file is loaded automatically. Every code cell begins with a guard. The first code cell sets CONFIRM_PUBLIC_WORKFLOW=False. After deliberate review, change that assignment to True and enable only the relevant CONFIRM_FULL_TRAINING, CONFIRM_TEST_EVALUATION or analysis flag. Re-running the unedited first cell resets the public opt-in. Missing confirmation stops even a cell run out of order. No imports/model setup occur before the guard. This protects notebooks, not arbitrary direct calls to source functions; scripts/phase6_smoke.py is a historical experimental helper and must not be used as a static check.

Do not “Run All” against an existing frozen experiment. Early notebooks write config/manifest/setup files and represent the historical preparation sequence; they are not an idempotent final-release installer. Default.yaml is the original A5, not A4. Use the controlled-ablation route for A4. Most setup functions require earlier private audit/fingerprint files, not just raw data. Historical integrity enforcement deliberately remains fail-closed.

## Explicit portability blockers

- Private manifests embed source archive locations. Relocation changes bytes/hashes; MORPHONET_DATA_ROOT alone cannot rewrite frozen split records.
- Translation changes console text in two modules, and notebook gates/path settings/config reference locations change source bytes. Original source/config fingerprints therefore do not match this public derivative. Never alter old manifests or disable checks to make a historical checkpoint appear compatible. A formally reviewed portable-release manifest/compatibility policy is future work.
- Checkpoints, per-sample predictions, private PID records, original setup/memory/protected-file records and many intermediate visualization artifacts are excluded. No full experimental reproduction is claimed. Authors must define controlled access or an authorized de-identified artifact release before users can rerun the frozen evaluation.
- The research environment was inspected, not reinstalled; the separate static installation is described in environment_compatibility.md; hardware/runtime determinism across versions is not guaranteed. The original assertion that the environment is named dl remains.
- Phase17 manuscript assembly is not an experimental notebook. Those temporary assembly/compilation scripts and submission assets are omitted; the public aggregate scientific tables/diagram are retained. Notebook16 is the publication-results workflow, not a promise to rebuild the final submission PDF.

## Interpretation

A4 was selected post hoc using outer-held-out results. Five folds share training subjects and are not independent training replications. Patient/cluster bootstrap intervals condition on saved models/predictions, exclude model selection/retraining variance and are marginal rather than Holm-adjusted intervals. No adjacent patient-correctness ablation contrast survives Holm. Meningioma findings are exploratory; occlusion provides sensitivity evidence, not causal localization or clinical validity. See scientific_protocol.md for precise frozen definitions.
