# Artifact availability

This is a local public-release candidate. "Included" means present in this package, not already published online. Private artifacts are not approved for redistribution by this phase.

| Artifact | Classification | What is available / missing |
|---|---|---|
| Cheng/Figshare MRI dataset | Publicly obtainable from original provider | Provider access and terms must be checked; no MRI or masks included |
| Exact dataset release/version | Unavailable or unverified for public replay | DOI is recorded; exact downloaded release and byte identity require private inspection records |
| Canonical patient splits / dataset manifest | Available only in private research workspace; not approved for distribution | Only aggregate counts and an unpopulated schema are included |
| A4 checkpoints, five folds | Available only in private research workspace; not approved for distribution | No weights or pretrained demo |
| Historical A5 checkpoints | Available only in private research workspace; not approved for distribution | Recorded reference hashes/paths do not supply checkpoints |
| Five baseline checkpoint sets | Available only in private research workspace; not approved for distribution | No downloadable weights established |
| Slice/patient OOF predictions | Available only in private research workspace; not approved for distribution | Patient-linked records excluded |
| Statistical analysis inputs | Available only in private research workspace; not approved for distribution | Aligned predictions and grouping/protocol records absent; aggregate outputs included |
| Explainability / occlusion outputs | Aggregate summaries included; MRI/maps private and not approved | No patient panels or intervention records |
| Final aggregate results | Publicly included | 21 CSVs in results/tables |
| Verified architecture diagram | Publicly included; publication permission still subject to author approval | Original A4 PNG and PDF preserved |
| Original execution-integrity manifests | Available only in private research workspace | Public release hashes verify this derivative, not the original runtime |

Inspecting aggregate results works with this package. Recomputing paired statistics requires the missing aligned predictions, patient grouping and frozen analysis protocol. Inference requires selected checkpoints, authorized MRI and compatible preprocessing/source provenance. Retraining additionally requires canonical splits, masks for supervision/augmentation, complete runtime and approval; it is a new experiment, not proof of identical frozen outcomes. These are four distinct levels of access. No full experimental reproducibility or controlled-access service is promised.


## Licensing scope

The project MIT License does not automatically cover external data, dependencies, weights or MRI-derived assets. The official Figshare record identifies version 8, DOI 10.6084/m9.figshare.1512427.v8, under CC BY 4.0. The exact downloaded release used in the frozen study remains unverified. No dataset, patient-level split, checkpoint or sample-level prediction is included. Architecture diagrams and aggregate tables preserve their frozen bytes. See [licensing and attribution](licensing_decisions.md).
