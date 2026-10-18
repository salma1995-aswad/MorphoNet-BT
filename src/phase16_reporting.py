"""English scientific audit, captions and manuscript-ready source materials."""
from pathlib import Path
from src.phase16_publication import SELECTION, texescape, markdown
from src.phase14_statistics import dump

LIMITATIONS = """A4 selection used the same outer-held-out ablation results now being reported, so architecture-selection optimism is not removed by the existing confidence intervals or p-values. A5, not A4, had the higher five-fold validation Macro-F1 mean. No independent external dataset or prospective clinical validation is available. The fixed-model patient bootstrap captures patient sampling variability but not retraining, checkpoint selection, architecture selection, or all dependence from overlapping cross-validation training sets. Five folds are not five independent training replications. The ablation sequence is cumulative rather than factorial; it cannot isolate independent component effects or interactions. McNemar tests patient Accuracy, not Macro-F1 or AUC. Bootstrap intervals are unadjusted even where p-values are Holm-adjusted. Morphology outputs are auxiliary, not clinically validated segmentation. Coarse Grad-CAM and class-agnostic BAA have different meanings; interpolation does not add spatial evidence. Full-grid distance error is background dominated, so tumor-only error is retained. Occlusion control regions match shape and area but use a gross foreground proxy, not a validated brain mask or anatomical/intensity matching. Zero and local-median replacement can create out-of-distribution images; negative drops mean increased true-class logits. Pituitary intervals include zero. The matched controls are not independent patients, and 15 slices lack eligible controls. Computational counts cover convolution/linear operations; theoretical parameter storage is not deployment memory. Comparable latency is unavailable. These results support an exploratory technical study, not clinical readiness or universal superiority."""

CLAIMS = [
('A4 is the best validated model.','Qualified / rewrite','T03, T04, S13','A4 has the highest pooled Macro-F1 in this comparison and was selected after held-out inspection; A5 had higher mean validation Macro-F1.'),
('A4 uses fewer parameters than all five baselines.','Supported within this comparison','T02','A4 has 575,359 parameters, fewer than the five evaluated baselines and A5; this does not establish universal efficiency.'),
('A4 has the lowest compute and fastest inference.','Rejected','T02','A4 has 0.275712 G convolution/linear MACs; MobileNetV3-Small has fewer MACs. No comparable latency claim is supported.'),
('Morphology supervision independently improves generalization.','Qualified / rewrite','T05, S02','A2 exceeds A1 in pooled performance in a cumulative sequence, but the adjusted paired test is not significant and an independent causal effect is not identified.'),
('BAA improves classification performance.','Rejected as unconditional claim','T05, T07, S02','Adding BAA from A2 to A3 reduced pooled classification metrics; its spatial attention overlap does not prove improved classification.'),
('Dual-path fusion significantly improves Accuracy.','Qualified / rewrite','T05, S02','A4 exceeds A3; nominal exact p = 0.03515625 becomes Holm p = 0.24609375 in the frozen family.'),
('AMGF is necessary or improves the principal model.','Rejected','T05, T06','A5 adds AMGF but has lower pooled results than A4; A4 versus A5 is not statistically significant.'),
('A4 is statistically superior to all baselines.','Rejected','T06','At Holm-adjusted 0.05, patient Accuracy differs only versus ResNet50 and MobileNetV3-Small in the frozen 10-test family; results remain exploratory after selection.'),
('Grad-CAM and BAA prove precise tumor localization.','Rejected','T07, G','Primary Grad-CAM has limited tumor overlap; BAA is a different, class-agnostic gating map. Neither establishes clinical segmentation or causal localization.'),
('Occlusion proves all tumor classes are causally used.','Rejected','T08, S08, H','Overall and Meningioma paired sensitivity is positive; Glioma differs from raw tumor-drop behavior and Pituitary intervals include zero. Anatomical/intensity matching is incomplete.'),
('The model generalizes to independent clinical populations.','Unsupported','All evaluation sources','Patient-disjoint internal cross-validation was used; independent external generalization remains untested.'),
('The method is clinically deployable.','Unsupported','T07 and dataset scope','Clinical applicability requires independent, prospective and workflow-specific validation; none is demonstrated here.'),
('All morphology operations disappear at inference.','Rejected','A, source architecture','Only ground-truth supervision is absent at inference. Shared morphology features and predicted boundary logits are retained for BAA; region/distance auxiliary outputs are conditional.'),
]

def build_report(pkg,scores,split,config):
    claims=['# Scientific claims audit','',SELECTION,'']
    for claim,status,evidence,wording in CLAIMS:
        names=[i['name'] for i in pkg.inventory if any(i['name'].startswith(t.strip()) for t in evidence.split(','))]
        claims += [f'## {claim}',f'\n**Assessment:** {status}.\n\n**Supporting result:** {evidence}.',
                   '\n**Source artifacts:** '+('; '.join('tables/csv/'+n+'.csv' for n in names) or 'metadata/source_of_truth_manifest.json and figure source records')+'.',
                   '\n**Recommended wording:** '+wording,'']
    (pkg.out/'audits/scientific_claims_audit.md').write_text('\n'.join(claims),encoding='utf-8')
    discrepancies=[
        dict(issue='Historical model naming',type='model identity',resolution='Phase 7 main MorphoNet-BT and the Phase 8 complexity reference are A5. Principal A4 is separately labelled; A5_reference predictions align with Phase 7.'),
        dict(issue='Default configuration has use_amgf=true',type='configuration scope',resolution='default.yaml describes the historical full A5; the frozen ablation variant factory and A4 setup report disable AMGF. No configuration was changed.'),
        dict(issue='Selection-stage contrast',type='scientific interpretation',resolution='A5 validation mean is higher. A4 selection is post hoc from outer-held-out results; no independent confirmation claimed.'),
        dict(issue='Eleven displayed McNemar rows',type='multiple-testing family',resolution='Ten unique Phase 14 tests, plus the reverse A5/A4 display. A4/A3 is in the family but excluded from the six baseline-comparison rows and retained in sequential results.'),
        dict(issue='Pooled OOF and fold means differ',type='aggregation',resolution='Pooled recomputation matches frozen source metrics; fold mean and sample SD are separately labelled in S10.'),
        dict(issue='3064 versus 3049 slices',type='eligibility',resolution='All slices contribute to classification/explainability; 15 lack matched controls for paired occlusion. All 233 patients remain represented.'),
        dict(issue='Controls versus perturbation responses',type='denominator',resolution='30490 unique matched locations; 60980 control responses across two replacements. Neither is an independent statistical sample size.'),
        dict(issue='Model size and latency',type='measurement availability',resolution='Only theoretical FP32 parameter payload is consistently available. Historical A5-only latency is excluded from model comparisons; no A4 latency fabricated.'),
        dict(issue='Old journal page limit',type='external guidance version',resolution='Current SIVP instructions checked 2026-10-10 specify 10 two-column pages excluding references; older limits are not used.'),
    ]
    dump(pkg.out/'audits/discrepancy_resolutions.json',discrepancies)
    phase15=pkg.js('results/explainability/final_scientific_audit.json')
    phase151=pkg.js('results/explainability/occlusion_audit/final_scientific_audit.json')
    assert phase15['status']=='PASS' and phase15['prediction_class_mismatches']==0 and phase15['max_absolute_probability_error']==0
    assert phase151['status']=='PASS' and phase151['paired_slices']==3049 and phase151['control_locations']==30490
    assert phase151['original_class_mismatches']==0 and phase151['maximum_probability_error']==0
    audit={
        'dataset':'PASS: frozen 3064 slices, 233 patients, three ordered classes, all 15 role manifests hashed and patient disjointness verified',
        'architecture':'PASS: code and historical shape audit inspected; 575359 parameters, 275711872 inference MACs, no AMGF for A4, boundary predictor remains active',
        'metrics':'PASS: 132 source-versus-recomputed checks, exact patient probability aggregation and 55 fold receipts',
        'phase13_vs_phase14':'PASS: effect points and exact patient tests independently recalculated on aligned predictions',
        'phase13_vs_phase15':'PASS: prior signed/hash-protected reproduction audit reports 0 probability error and 0 class mismatch; no new reproduction inference in Phase 16',
        'phase15_vs_phase151':'PASS: original probability/class reproduction audit remains frozen; eligible cohort and controls distinguished',
        'statistics':'PASS: ten unique Holm tests preserved; reverse comparison not recounted; bootstrap intervals imported without changing family',
        'explainability':'PASS: target stage3 predicted-class Grad-CAM and auxiliary limitations disclosed; previous masks/checkpoints untouched',
        'occlusion':'PASS: prior exact area/shape/no-overlap checks verified via protected audit; patient-equal means and clustering retained',
        'selection':'PASS: A5 higher validation mean explicitly reported; A4 post-hoc selection and optimism disclosed',
        'unresolved_numerical_discrepancies':[],
        'limitations_not_resolved_by_audit':LIMITATIONS,
    }
    dump(pkg.out/'audits/cross_phase_consistency.json',audit)
    captions=['# Publication captions and table notes','',SELECTION,'','## Tables','']
    for t in pkg.inventory:captions += [f'### {t["name"]}',t['caption'],'']
    captions += ['## Figures','']
    for f in pkg.figures:captions += [f'### {f["name"]}',f['caption'],'']
    (pkg.out/'captions/all_captions.md').write_text('\n'.join(captions),encoding='utf-8')
    (pkg.out/'captions/all_captions.tex').write_text('\n\n'.join(texescape(f['name'])+': '+texescape(f['caption']) for f in pkg.figures),encoding='utf-8')
    packages=r'''% Current SIVP guidance checked 2026-10-10. Use the journal's current template.
% \documentclass[iicol]{sn-jnl}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable} % Extended supplementary tables, in a one-column supplement
\usepackage{array}
% No automatic resizebox: do not shrink scientific text below readable size.
% Main comparison tables use concise presentation views supplied separately.
% Input paths below assume the manuscript is in results/publication/.
'''
    (pkg.out/'manuscript_snippets/required_packages.tex').write_text(packages,encoding='utf-8')
    includes=[]
    for f in pkg.figures:
        if f['allocation']!='main':continue
        wide=f.get('width_mm',174)>90;env='figure*' if wide else 'figure'
        includes += [f'\\begin{{{env}}}[t]',r'\centering',f'\\includegraphics[width=\\{("textwidth" if wide else "columnwidth")}]{{figures/main/{f["name"]}.pdf}}',
                     '\\caption{'+texescape(f['caption'])+'}',f'\\label{{fig:{f["name"]}}}',f'\\end{{{env}}}','']
    (pkg.out/'manuscript_snippets/main_figures.tex').write_text('\n'.join(includes),encoding='utf-8')
    # Compact views keep full precision in CSV while giving readable manuscript widths.
    for name in ['T03_slice_comparison','T04_patient_comparison']:
        d=pkg.tables[name];lines=[r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Model & Acc. & mP & mR & mF1 & wF1 & AUC \\',r'\midrule']
        for row in d.itertuples(index=False,name=None):
            model=row[0].replace('MorphoNet-BT ','');cells=[texescape(model)]+[f'{100*v:.2f}' for v in row[1:]]
            if model=='A4':cells=[r'\textbf{'+x+'}' for x in cells]
            lines.append(' & '.join(cells)+r' \\')
        lines += [r'\bottomrule',r'\end{tabular}']
        (pkg.out/f'tables/latex/{name}_compact.tex').write_text('\n'.join(lines),encoding='utf-8')
    def compact(name,headers,rows):
        lines=[r'\begin{tabular}{l'+'r'*(len(headers)-1)+'}',r'\toprule',' & '.join(texescape(h) for h in headers)+r' \\',r'\midrule']
        for row in rows:
            cells=[texescape(v) for v in row]
            if str(row[0]) in ['A4','MorphoNet-BT A4']:cells=[r'\textbf{'+v+'}' for v in cells]
            lines.append(' & '.join(cells)+r' \\')
        lines += [r'\bottomrule',r'\end{tabular}']
        (pkg.out/f'tables/latex/{name}_compact.tex').write_text('\n'.join(lines),encoding='utf-8')
    compact('T01_dataset',['Class','Slices','Patients','Slices (%)','Patients (%)'],[[r.Class,r.Slices,r.Patients,f'{r[3]:.2f}',f'{r[4]:.2f}'] for r in pkg.tables['T01_dataset'].itertuples(index=False,name='Row')])
    compact('T02_complexity',['Model','Params (M)','MACs (G)','FLOPs (G)','FP32 MB'],[[str(r[0]).replace('MorphoNet-BT ',''),*[f'{v:.6f}' for v in r[1:]]] for r in pkg.tables['T02_complexity'].itertuples(index=False,name=None)])
    compact('T05_sequential_ablation',['Variant','Params','MACs (G)','sAcc','sF1','pAcc','pF1','pAUC'],[[r.Variant,r.Parameters,f'{r.MACs/1e9:.6f}',*[f'{100*v:.2f}' for v in r[3:]]] for r in pkg.tables['T05_sequential_ablation'].itertuples(index=False,name='Row')])
    from src.phase16_publication import NAMES
    compact('T06_exploratory_statistics',['A4 versus','b/c','Delta (pp)','95% CI (pp)','Exact p','Holm p'],[[NAMES.get(r.right,r.right).replace('MorphoNet-BT ',''),f'{r.left_only}/{r.right_only}',f'{r.effect_pp:.3f}',f'[{r.ci_low_pp:.3f}, {r.ci_high_pp:.3f}]',f'{r.p_exact:.6f}',f'{r.p_holm:.6f}'] for r in pkg.tables['T06_exploratory_statistics'].itertuples()])
    compact('T07_explainability',['Map','Metric','Mean','95% CI'],[[r.Map,r.Metric.replace('_',' '),f'{r.Mean:.4f}',f'[{r.CI_low:.4f}, {r.CI_high:.4f}]'] for r in pkg.tables['T07_explainability'].itertuples()])
    compact('T08_occlusion_sensitivity',['Replacement / class','n','Tumor','Control','Delta','95% CI'],[[('Zero' if r.Replacement=='zero' else 'Median')+' / '+r.Group,r.Patients,f'{r.Tumor_drop:.4f}',f'{r.Control_drop:.4f}',f'{r.Delta:.4f}',f'[{r.CI_low:.4f}, {r.CI_high:.4f}]'] for r in pkg.tables['T08_occlusion_sensitivity'].itertuples()])
    proof=[r'\documentclass[10pt]{article}',r'\usepackage[paperwidth=210mm,paperheight=297mm,margin=18mm]{geometry}',r'\usepackage{booktabs,longtable,graphicx}',r'\begin{document}',r'\section*{Phase 16 table layout proof}',r'Technical layout proof; not a manuscript. Compact performance metrics are percentages. Full-precision source CSVs and complete captions accompany this file.']
    for name in ['T01_dataset','T02_complexity','T03_slice_comparison','T04_patient_comparison','T05_sequential_ablation','T06_exploratory_statistics','T07_explainability','T08_occlusion_sensitivity']:
        proof += [r'\clearpage',r'\section*{'+texescape(name)+'}',r'\noindent{\small\setlength{\tabcolsep}{3pt}\input{tables/latex/'+name+r'_compact.tex}}\par',r'\medskip',texescape(next(t['caption'] for t in pkg.inventory if t['name']==name))]
    proof += [r'\end{document}']
    (pkg.out/'table_layout_proof.tex').write_text('\n'.join(proof),encoding='utf-8')
    (pkg.out/'manuscript_snippets/table_notes.md').write_text('''# Table assembly notes

Full-precision CSVs are the numerical source of truth; Markdown and LaTeX round only for display. Compact T03/T04 views express all six metrics as percentages (including AUC); mP = macro precision, mR = macro recall, mF1 = macro F1, wF1 = support-weighted F1. A4 is bold because it is the selected principal model, not because every cell is significantly superior. Use table* for wide tables; split extended columns logically rather than shrinking text. Longtable outputs belong in a separate one-column supplement, since longtable does not run inside a two-column float. Main manuscript captions/footnotes are supplied in captions/all_captions.md. Table numbering should follow final manuscript order.

T06: differences and CIs are percentage points, p-values are two-sided; Holm family size is ten unique tests. Intervals are unadjusted. T07: means/CIs are fractions except distance error; peak-inside percentage is 100 times the reported fraction. T08: logit units, patient-equal estimand, no formal p-values. T02: no comparable latency, no measured weight-file size; FP32 parameter MB is a theoretical payload.
''',encoding='utf-8')
    journal='''# Journal layout and allocation

Current official SIVP instructions were checked on 2026-10-10: https://link.springer.com/journal/11760/submission-guidelines . They specify the Springer Nature sn-jnl template with iicol and a maximum of 10 final two-column pages, excluding references. No current journal template is supplied in this project. Nominal figure widths are 84 mm (single column) and 174 mm (double column); validate final placement against the acquired manuscript template. New figures use 8-point DejaVu Sans, embedded PDF fonts and 600-dpi PNG. The three byte-identical historical Phase 15 figure pairs retain their native large-format layout and must not be shrunk to single-column size. Full captions live outside new figures.

Main recommendation: A architecture (double column), B compact performance (double), E ablation (single), F paired effects (single), G representative cached maps or the original MRI-overlay panel at readable double-column/large format. Prioritize tables T03/T04 (can be combined in final writing), T05 and T06. Summarize dataset and complexity in text if space is constrained.

Supplement: T01/T02 detail, T07/T08, every S table; C confusion matrices, D ROC, H occlusion, G failure cases and unchanged full MRI/morphology examples, plus single-column B alternatives. Preserve all unfavorable results. Figures and tables are a complete evidence package, not a claim that every item fits in ten pages. Phase 17 must perform final manuscript pagination. No Phase 17 writing or experiments have been started.
'''
    (pkg.out/'manuscript_snippets/journal_layout_and_allocation.md').write_text(journal,encoding='utf-8')
    outline=['# PHASE 16 PUBLICATION REPORT','', '## Executive summary',SELECTION,
             '\nAll results were consolidated from frozen artifacts. No training, model loading, MRI inference, checkpoint reselection or historical-file rewriting was performed. Numerical source checks passed; scientific limitations remain substantive.',
             '\n## Frozen protocol verification','3,064 grayscale CE-MRI slices from 233 patients, class order Meningioma/Glioma/Pituitary. Five patient-disjoint outer StratifiedGroupKFold partitions (seed 42); validation is a stratified patient-level inner split at 20%. Input is 1 x 256 x 256, per-image all-pixel z-score, no clipping. Patient predictions average saved slice probabilities in float64 without renormalization, then argmax. Training-stage tumor masks may constrain augmentation and provide privileged targets, but are never MRI-classifier inference inputs.',
             '\n## Final A4 architecture verification','575,359 parameters; 275,711,872 classification convolution/linear MACs (551,423,744 conventionally counted FLOPs). MSS, convolutional MFE stages, privileged morphology supervision, BAA, global/local paths and parameter-free arithmetic mean fusion are present; AMGF is absent. Region/distance outputs are conditional auxiliary outputs; shared morphology features and boundary prediction remain for BAA at inference. The default configuration describes historical A5 and must not be mistaken for A4 variant flags.',
             '\n## Dataset statistics', markdown(pkg.tables['T01_dataset']),
             '\n## Baseline comparison summary',markdown(pkg.tables['T02_complexity'],floatfmt='.6f'),markdown(pkg.tables['T03_slice_comparison'],floatfmt='.6f'),markdown(pkg.tables['T04_patient_comparison'],floatfmt='.6f'),
             '\n## Ablation and selection',markdown(pkg.tables['T05_sequential_ablation'],floatfmt='.6f'),markdown(pkg.tables['S13_selection_audit'],floatfmt='.8f'),
             '\nA3 and A5 reduce pooled performance relative to their predecessors. The validation-stage ordering does not justify claiming that A4 was selected on validation. Cumulative component effects are not independent.',
             '\n## Statistical findings',markdown(pkg.tables['T06_exploratory_statistics'],floatfmt='.8f'),markdown(pkg.tables['S02_sequential_statistics'],floatfmt='.8f'),
             '\nOnly comparisons versus ResNet50 and MobileNetV3-Small meet Holm-adjusted 0.05 for patient Accuracy. A4 versus A5 and DenseNet121 do not. There is no Holm-significant sequential comparison. Macro-F1/AUC bootstrap differences and patient-cluster slice intervals are fully retained in S03/S04; McNemar does not test these metrics.',
             '\n## Explainability findings',markdown(pkg.tables['T07_explainability'],floatfmt='.6f'),
             '\nPrimary Grad-CAM has low attention mass and peak-inside fraction; BAA overlaps more but is a different class-agnostic signal. Auxiliary outputs are not validated clinical segmentation.',
             '\n## Occlusion findings',markdown(pkg.tables['T08_occlusion_sensitivity'],floatfmt='.6f'),
             '\nThe primary overall patient-equal effect is positive. Meningioma is strongest; Pituitary intervals include zero with both replacements. Glioma paired contrast must not be confused with its raw tumor drop. Replacements preserve the original normalization. Controls are not anatomically or intensity matched, and 30,490 controls are dependent within 233 patients.',
             '\n## Publication figure inventory']
    outline += [f'- {f["name"]}: {f["allocation"]}; PNG/PDF; {f["kind"]}.' for f in pkg.figures]
    outline += ['\n## Publication table inventory']+[f'- {t["name"]}: {t["rows"]} rows; {t["allocation"]}; CSV, Markdown, booktabs LaTeX.' for t in pkg.inventory]
    outline += ['\n## Main versus supplementary allocation',journal.split('\n\n',1)[1],
                '\n## Cross-phase consistency', '\n'.join(f'- {k}: {v}' for k,v in audit.items() if k!='limitations_not_resolved_by_audit'),
                '\n## Scientific claims audit','See audits/scientific_claims_audit.md. Unsupported claims of universally lowest compute, independent component benefit, AMGF benefit, universal statistical superiority, precise causal localization, independent generalization and clinical readiness were rejected or qualified.',
                '\n## Discrepancies and resolutions','\n'.join(f'- {d["issue"]}: {d["resolution"]}' for d in discrepancies),
                '\n## Unresolved discrepancies','No unresolved numerical discrepancy was found in the recomputed frozen metrics. Unavailable comparable latency and independent validation remain unavailable; they were not fabricated.',
                '\n## Scientific limitations',LIMITATIONS,
                '\n## Phase 17 readiness','The numerical evidence package is ready for manuscript drafting with the stated limitations. Share this report, all captions, claims/cross-phase/selection audits, full-precision tables, PDF figures, LaTeX snippets and provenance metadata. Full manuscript LaTeX compilation, author metadata, bibliography and final journal pagination are not completed by this artifact phase. Phase 17 has not begun.']
    (pkg.out/'PHASE16_PUBLICATION_REPORT.md').write_text('\n\n'.join(outline),encoding='utf-8')
    (pkg.out/'manuscript_snippets/limitations_and_selection.md').write_text('# Selection and limitations\n\n'+SELECTION+'\n\n'+LIMITATIONS,encoding='utf-8')
