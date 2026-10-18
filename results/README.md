# Final aggregate evidence

The 21 CSVs in tables/ retain the exact verified final aggregate values. T01-T08 cover the dataset, complexity, slice/patient comparisons, cumulative ablations, statistics, explainability and occlusion. S01-S13 preserve fold counts, paired/absolute intervals, robustness, meningioma analyses, occlusion estimands, class metrics, validation results and selection audit. Historical filename numbering is not manuscript table numbering. Classification metrics are stored as fractions; display percentages do not alter original values.

figures/A4_architecture.png and .pdf are the verified principal-model diagram. No MRI panel, sample-level predictions or patient records are distributed. All scientific assets are covered by the static integrity fixture in tests/data/scientific_sha256.json. Historical untrained/gradient/memory diagnostic exports were removed from this public package; their original records remain outside it. Final comparison evidence, including A5 and all baselines, remains intact.

Run `python scripts/show_results.py` for CSV-only inspection. These are archived results, not new experiments; complete statistical recomputation requires the missing aligned private predictions.
