"""Patient-aware occlusion summaries, figures, and scientific reporting."""
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
from src.phase14_statistics import read,save,dump,draws,markdown_table
from .reporting import summarize
from .occlusion import POLICY,SETTINGS,translate,perturb,CLASSES

METRICS=['tumor_true_logit_drop','control_true_logit_drop','delta_true_logit_drop',
    'tumor_true_probability_drop','control_true_probability_drop','delta_true_probability_drop',
    'tumor_original_predicted_logit_drop','control_original_predicted_logit_drop','delta_original_predicted_logit_drop',
    'tumor_margin_drop','control_margin_drop','delta_margin_drop','tumor_flip','control_flip','delta_flip',
    'tumor_correct','control_correct','tumor_mean_absolute_intensity_change','control_mean_absolute_intensity_change']

def summary_groups(part):
    return [('overall',part)]+[(c,part[part.true_class==c]) for c in CLASSES]+[('original_correct',part[part.original_correct==1]),('original_incorrect',part[part.original_correct==0])]

def intervals(x):return np.percentile(x,[2.5,97.5],axis=0)

def analyze(root,out,slices,controls,originals,sampling):
    pids=sorted(originals.PID.unique());counts=draws(len(pids),10000,np.random.default_rng(42))
    indices=np.random.default_rng(42).integers(0,len(pids),size=(10000,len(pids)))
    np.savez_compressed(out/'metadata/patient_bootstrap_draws.npz',pids=np.array(pids),counts=counts)
    eligible=slices[slices.paired_available].copy()
    for c in METRICS:eligible[c]=pd.to_numeric(eligible[c],errors='raise')
    assert np.isfinite(eligible[METRICS].to_numpy()).all()
    summary=[];patient_rows=[];folds=[];distribution=[];fliprows=[]
    for strategy in ['zero','local_median']:
        part=eligible[eligible.strategy==strategy]
        for group,data in summary_groups(part):
            assert len(data)>0,f'No eligible slices for {strategy}/{group}'
            summary+=summarize(data,METRICS,pids,counts,dict(strategy=strategy,group=group))
        patients=part.groupby(['PID','fold','true_class'],sort=True)[METRICS].mean().reset_index()
        g=part.groupby('PID').delta_true_logit_drop
        patients['median_delta_sensitivity']=patients.PID.map(g.median())
        patients['positive_delta_slice_fraction']=patients.PID.map(g.apply(lambda v:float((v>0).mean())))
        patients['paired_slices']=patients.PID.map(g.size())
        patients['total_slices']=patients.PID.map(originals.groupby('PID').size())
        patients['strategy']=strategy;patient_rows.append(patients)
        v=patients.set_index('PID').delta_true_logit_drop.reindex(pids).to_numpy(float)
        sampled=v[indices]
        for metric,point,boot in [('patient_mean_delta',np.nanmean(v),np.nanmean(sampled,axis=1)),
                                  ('patient_median_delta',np.nanmedian(v),np.nanmedian(sampled,axis=1)),
                                  ('positive_patient_fraction',np.nanmean(np.where(np.isfinite(v),v>0,np.nan)),np.nanmean(np.where(np.isfinite(sampled),sampled>0,np.nan),axis=1))]:
            lo,hi=intervals(boot)
            distribution.append(dict(strategy=strategy,metric=metric,estimate=point,ci_low=lo,ci_high=hi,patients=int(np.isfinite(v).sum()),
                positive_patients=int((v>0).sum()),negative_patients=int((v<0).sum()),zero_patients=int((v==0).sum())))
        for fold in range(1,6):
            data=part[part.fold==fold]
            for estimand,values in [('slice_weighted',data[METRICS]),('patient_equal',data.groupby('PID')[METRICS].mean())]:
                folds.append(dict(strategy=strategy,fold=fold,estimand=estimand,slices=len(data),patients=data.PID.nunique(),**values.mean().to_dict()))
        for group,data in summary_groups(part):
            for endpoint,t,c in [('prediction_flip','tumor_flip','control_flip')]+([('original_correct_loss','tumor_correct','control_correct')] if group=='original_correct' else [])+([('original_incorrect_recovery','tumor_correct','control_correct')] if group=='original_incorrect' else []):
                temp=data[['PID',t,c]].copy();temp.columns=['PID','tumor_rate','control_rate']
                if endpoint=='original_correct_loss':temp[['tumor_rate','control_rate']]=1-temp[['tumor_rate','control_rate']]
                temp['paired_difference']=temp.tumor_rate-temp.control_rate
                fliprows+=summarize(temp,['tumor_rate','control_rate','paired_difference'],pids,counts,dict(strategy=strategy,group=group,endpoint=endpoint))
    summary=pd.DataFrame(summary);patients=pd.concat(patient_rows,ignore_index=True)
    # Retain every patient, including a subject with no eligible paired slices.
    all_patients=originals.groupby(['PID','fold','true_class']).size().rename('cohort_slices').reset_index()
    exports=[]
    for strategy in ['zero','local_median']:
        export=all_patients.merge(patients[patients.strategy==strategy],on=['PID','fold','true_class'],how='left')
        missing=export.strategy.isna()
        export['strategy']=strategy;export['paired_available']=~missing
        export.loc[missing,'paired_slices']=0
        export.loc[missing,'total_slices']=export.loc[missing,'cohort_slices']
        exports.append(export.astype(object).where(export.notna(),'unavailable_no_controls'))
    save(pd.concat(exports,ignore_index=True),out/'tables/occlusion_per_patient.csv')
    save(summary[summary.group=='overall'],out/'tables/occlusion_overall_summary.csv')
    save(summary[summary.group.isin(CLASSES)],out/'tables/occlusion_per_class_summary.csv')
    save(summary[summary.group.isin(['original_correct','original_incorrect'])],out/'tables/occlusion_correct_vs_incorrect.csv')
    save(pd.DataFrame(fliprows),out/'tables/occlusion_prediction_flips.csv')
    save(pd.DataFrame(folds),out/'tables/occlusion_fold_summary.csv')
    save(pd.DataFrame(distribution),out/'tables/occlusion_patient_effect_distribution.csv')
    # Join frozen Phase 15 data by exact slice identity, not positional coincidence.
    cam=read(root/'results/explainability/tables/gradcam_localization_per_slice.csv')
    cam=cam[(cam.layer=='stage3')&(cam.target=='predicted')].set_index('sample_id')
    region=read(root/'results/explainability/tables/morphology_region_metrics.csv').set_index('sample_id')
    ids=originals.sample_id.tolist();assert set(ids)==set(cam.index)==set(region.index)
    for col in ['PID','fold','true_label']:
        np.testing.assert_array_equal(cam.loc[ids,col],originals[col]);np.testing.assert_array_equal(region.loc[ids,col],originals[col])
    correlations=[];patient_assoc=[]
    for strategy in ['zero','local_median']:
        part=eligible[eligible.strategy==strategy].set_index('sample_id')
        joined=part[['PID','delta_true_logit_drop']].join(cam[['attention_mass','activation_ratio','peak_inside']]).join(region[['dice']].rename(columns={'dice':'region_dice'}))
        p=joined.groupby('PID').mean(numeric_only=True).sort_index();p['strategy']=strategy;patient_assoc.append(p.reset_index())
        rng=np.random.default_rng(42);idx=rng.integers(0,len(p),size=(10000,len(p)))
        for metric in ['attention_mass','activation_ratio','peak_inside','region_dice']:
            x,y=p[metric].to_numpy(),p.delta_true_logit_drop.to_numpy()
            a,b=rankdata(x[idx],axis=1),rankdata(y[idx],axis=1);a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
            den=np.sqrt(np.sum(a*a,1)*np.sum(b*b,1));assert (den>0).all()
            boot=(a*b).sum(1)/den;lo,hi=intervals(boot)
            correlations.append(dict(strategy=strategy,metric=metric,spearman_rho=float(spearmanr(x,y).statistic),ci_low=lo,ci_high=hi,patients=len(p)))
    correlations=pd.DataFrame(correlations);save(correlations,out/'tables/occlusion_gradcam_correlations.csv')
    patient_assoc=pd.concat(patient_assoc,ignore_index=True);save(patient_assoc,out/'tables/patient_correlation_inputs.csv')
    zero=eligible[eligible.strategy=='zero'].set_index('sample_id');local=eligible[eligible.strategy=='local_median'].set_index('sample_id').loc[zero.index]
    contrast=zero[['PID','true_class','original_correct']].copy();contrast['local_minus_zero_delta']=local.delta_true_logit_drop-zero.delta_true_logit_drop
    contrast['slice_sign_agreement']=(np.sign(local.delta_true_logit_drop)==np.sign(zero.delta_true_logit_drop)).astype(float)
    contrastrows=[]
    for group,data in summary_groups(contrast):contrastrows+=summarize(data,['local_minus_zero_delta','slice_sign_agreement'],pids,counts,dict(group=group))
    contrast_summary=pd.DataFrame(contrastrows);save(contrast_summary,out/'tables/method_sensitivity_comparison.csv')
    valid_counts=sampling.groupby(['true_class']).agg(slices=('sample_id','size'),patients=('PID','nunique'),controls=('n_controls','sum'),min_controls=('n_controls','min'),max_controls=('n_controls','max'),shortfall_slices=('controls_shortfall','sum'),zero_control_slices=('n_controls',lambda v:int((v==0).sum())))
    save(valid_counts.reset_index(),out/'tables/control_availability_by_class.csv')
    # Explicit coverage ledger retains subjects even when no paired effect can be calculated.
    coverage=originals.groupby(['PID','fold','true_class']).size().rename('total_slices').reset_index()
    pair_counts=zero.groupby('PID').size();coverage['paired_slices']=coverage.PID.map(pair_counts).fillna(0).astype(int)
    coverage['paired_available']=coverage.paired_slices>0;save(coverage,out/'tables/patient_coverage_audit.csv')
    make_figures(root,out,slices,controls,originals,summary,patient_assoc,correlations)
    write_report(out,summary,correlations,pd.DataFrame(distribution),pd.DataFrame(folds),pd.DataFrame(fliprows),contrast_summary,valid_counts.reset_index(),coverage)

def make_figures(root,out,slices,controls,originals,summary,assoc,correlations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    folder=out/'figures'
    def finish(fig,name):
        for ext in ['png','pdf']:fig.savefig(folder/f'{name}.{ext}',dpi=320,bbox_inches='tight')
        plt.close(fig)
    original=originals.set_index('sample_id')
    chosen=['66','2570','1315','36','2172','1566']
    manifest=[]
    for s in chosen:
        r=original.loc[s];manifest.append(dict(sample_id=s,fold=int(r.fold),true_class=r.true_class,selection='Phase 15 seeded random correct' if s in chosen[:3] else 'Phase 15 highest-confidence error',control_index=0))
    save(pd.DataFrame(manifest),out/'tables/representative_sample_manifest.csv')
    for kind,ids in [('correct_examples',chosen[:3]),('error_examples',chosen[3:])]:
        fig,axes=plt.subplots(3,5,figsize=(13,10.5),layout='constrained')
        for i,s in enumerate(ids):
            archive=np.load(out/f'cache/{s}.npz');image,mask=archive['image'],archive['mask'];trans=archive['translations'];r=original.loc[s]
            lo,hi=np.percentile(image,[1,99]);part=slices[(slices.sample_id==s)&(slices.strategy=='zero')].iloc[0]
            displays=[image,image,perturb(image,mask,0),image,image]
            if len(trans):
                c=translate(mask,*trans[0]);displays[3]=perturb(image,c,0)
                cr=controls[(controls.sample_id==s)&(controls.strategy=='zero')&(controls.control_index==0)].iloc[0]
                control_caption=f'Control: {CLASSES[int(cr.predicted_label)]}\nconfidence {cr.confidence:.3f}'
            else:c=np.zeros_like(mask);control_caption='No valid control'
            captions=[f'Original: {CLASSES[int(r.predicted_label)]}\nconfidence {r.confidence:.3f}','Ground-truth tumor',f'Tumor: {CLASSES[int(part.tumor_predicted_label)]}\nconfidence {part.tumor_confidence:.3f}',control_caption,'Head-interior proxy / control']
            captions[0]=f'Sample {s}; fold {r.fold}\nTrue: {r.true_class}\n'+captions[0]
            for j,ax in enumerate(axes[i]):
                ax.imshow(displays[j],cmap='gray',vmin=lo,vmax=hi);ax.set_axis_off();ax.set_title(captions[j],fontsize=9)
            axes[i,1].imshow(np.ma.masked_where(~mask,mask),cmap='autumn',alpha=.45,vmin=0,vmax=1)
            axes[i,4].contour(archive['interior'],levels=[.5],colors=['cyan'],linewidths=.6)
            if c.any():axes[i,4].contour(c,levels=[.5],colors=['lime'],linewidths=.8)
        fig.suptitle(f'Figure 1: {kind.replace("_"," ")}; zero replacement after normalization\nOriginal display limits reused; cyan is a head-interior proxy, not validated brain segmentation')
        finish(fig,'figure1_'+kind)
    def estimate(group,metric,strategy='zero',estimand='patient_equal'):
        return summary[(summary.group==group)&(summary.metric==metric)&(summary.strategy==strategy)&(summary.estimand==estimand)].iloc[0]
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for i,metric in enumerate(['tumor_true_logit_drop','control_true_logit_drop','delta_true_logit_drop']):
        r=estimate('overall',metric);ax.errorbar(i,r['mean'],yerr=[[r['mean']-r.ci_low],[r.ci_high-r['mean']]],fmt='o',capsize=5)
    ax.set_xticks(range(3),['Tumor drop','Mean control drop','Paired difference']);ax.axhline(0,color='gray',ls='--');ax.set_ylabel('True-class logit change')
    ax.set_title('Figure 2. Primary patient-mean effects with 95% bootstrap intervals');finish(fig,'figure2_paired_sensitivity')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,metric in zip(axes,['delta_true_logit_drop','delta_flip']):
        for i,cls in enumerate(CLASSES):
            r=estimate(cls,metric);scale=100 if metric=='delta_flip' else 1
            ax.errorbar(i,scale*r['mean'],yerr=[[scale*(r['mean']-r.ci_low)],[scale*(r.ci_high-r['mean'])]],fmt='o',capsize=4)
        ax.set_xticks(range(3),CLASSES);ax.axhline(0,color='gray',ls='--');ax.set_title('Delta sensitivity (logit units)' if metric!='delta_flip' else 'Tumor - control flip rate (pp)')
    fig.suptitle('Figure 3. Class-wise patient-mean primary effects');finish(fig,'figure3_class_sensitivity')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,metric in zip(axes,['delta_true_logit_drop','delta_flip']):
        for i,group in enumerate(['original_correct','original_incorrect']):
            r=estimate(group,metric);scale=100 if metric=='delta_flip' else 1
            ax.errorbar(i,scale*r['mean'],yerr=[[scale*(r['mean']-r.ci_low)],[scale*(r.ci_high-r['mean'])]],fmt='o',capsize=4)
        ax.set_xticks([0,1],['Originally correct','Originally incorrect']);ax.axhline(0,color='gray',ls='--');ax.set_title('Delta sensitivity (logit units)' if metric!='delta_flip' else 'Tumor - control flip rate (pp)')
    fig.suptitle('Figure 4. Conditional patient means; groups may share patients');finish(fig,'figure4_correct_vs_incorrect')
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained');p=assoc[assoc.strategy=='zero']
    for ax,metric in zip(axes.flat,['attention_mass','activation_ratio','peak_inside','region_dice']):
        r=correlations[(correlations.strategy=='zero')&(correlations.metric==metric)].iloc[0]
        ax.scatter(p[metric],p.delta_true_logit_drop,s=15,alpha=.6,color='#1565a8');ax.axhline(0,color='gray',ls='--')
        ax.set_xlabel('Patient mean '+metric.replace('_',' '));ax.set_ylabel('Patient mean DeltaSensitivity')
        ax.set_title(f'Spearman rho={r.spearman_rho:.3f}; 95% CI [{r.ci_low:.3f}, {r.ci_high:.3f}]',fontsize=10)
    fig.suptitle('Figure 5. Descriptive patient-level associations, not causal effects');finish(fig,'figure5_gradcam_relationships')
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for j,strategy in enumerate(['zero','local_median']):
        for i,group in enumerate(['overall']+CLASSES):
            r=estimate(group,'delta_true_logit_drop',strategy)
            ax.errorbar(i+(j-.5)*.15,r['mean'],yerr=[[r['mean']-r.ci_low],[r.ci_high-r['mean']]],fmt='o',capsize=3,color=['#1565a8','#ba6217'][j],label=strategy if i==0 else None)
    ax.set_xticks(range(4),['Overall']+CLASSES);ax.axhline(0,color='gray',ls='--');ax.set_ylabel('Patient-mean DeltaSensitivity');ax.legend()
    ax.set_title('Figure 6. Prespecified replacement-method sensitivity');finish(fig,'figure6_replacement_sensitivity')

def write_report(out,summary,correlations,distribution,folds,flips,contrast,availability,coverage):
    report='# Phase 15.1: Diagnostic Tumor Occlusion Sensitivity\n\n## Fixed protocol\n\n'+POLICY
    report+='\n\nOcclusion is a diagnostic perturbation approach; see [Zeiler and Fergus, ECCV 2014](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf). The current mask-shaped, matched-control protocol is explicitly defined above and is not claimed to be a validated clinical counterfactual intervention.\n'
    report+='\n## Coverage\n\n'+markdown_table(availability)+f'\n\nPatients with at least one paired slice: {int(coverage.paired_available.sum())}/{len(coverage)}. Zero-control samples remain in the per-slice and sampling tables with explicit unavailable paired fields. Confidence intervals use only available matched observations and this eligibility is independent of model responses.\n'
    for title,frame in [('All effects',summary),('Patient effect distribution',distribution),('Prediction flips and recovery',flips),('Fold summaries',folds),('Descriptive correlations',correlations),('Replacement-method sensitivity',contrast)]:
        formatted=frame.copy()
        for col in formatted.select_dtypes('float').columns:formatted[col]=formatted[col].map(lambda x:f'{x:.6f}')
        report+='\n\n## '+title+'\n\n'+markdown_table(formatted)
    def row(strategy,group,metric):return summary[(summary.strategy==strategy)&(summary.group==group)&(summary.metric==metric)&(summary.estimand=='patient_equal')].iloc[0]
    def fmt(strategy,group,metric):
        r=row(strategy,group,metric);return f'{r["mean"]:.4f} (95% CI {r.ci_low:.4f} to {r.ci_high:.4f})'
    primary=row('zero','overall','delta_true_logit_drop')
    outcome='greater tumor perturbation sensitivity' if primary.ci_low>0 else ('greater control perturbation sensitivity' if primary.ci_high<0 else 'limited evidence of a directional tumor-control sensitivity difference')
    interpretation=f'The primary patient-level interval supports {outcome} under the tested zero-replacement intervention. This does not establish causal reliance, clinical trustworthiness, or independent confirmation. Head-interior controls are not validated intracranial masks; anatomical and intensity differences may confound the comparison. Zero replacement and local replacement may both create out-of-distribution MRI appearances. Training-set overlap across folds and training randomness are not fully represented by patient bootstrap intervals. Phase 15 Grad-CAM and region-Dice correlations are descriptive. No settings, checkpoints, architecture, or preprocessing were changed in response to these results.'
    report+='\n\n## Scientific interpretation and limitations\n\n'+interpretation
    paired_n=int(coverage.paired_slices.sum());total_n=int(coverage.total_slices.sum())
    cohort_note=f'Paired comparisons included {paired_n:,} eligible slices from {int(coverage.paired_available.sum())} patients. The remaining {total_n-paired_n} slices had no valid matched control and retained original/tumor responses with explicitly unavailable paired effects. No background-control fallback was used.'
    prose=f'''### Diagnostic occlusion sensitivity

We performed a post-hoc diagnostic occlusion audit using the five frozen A4 checkpoints and their corresponding held-out partitions, covering 3,064 MRI slices from 233 patients. Original classifications were reproduced before perturbation. Tumor masks were used solely to construct interventions, while the classifier continued to receive MRI images only. The primary intervention replaced tumor pixels with zero after the original 256x256 z-score normalization, without renormalizing the perturbed image. Up to ten shape- and area-matched translated masks were sampled outside each tumor using a fixed seed. Controls were constrained to an eroded, non-air head-interior proxy; this proxy is not a validated intracranial segmentation.

{cohort_note}

The primary endpoint was the true-class logit decrease after tumor occlusion minus the mean decrease after control occlusions, first averaged within patients. Mean tumor and control logit drops were {fmt('zero','overall','tumor_true_logit_drop')} and {fmt('zero','overall','control_true_logit_drop')}, respectively. The paired patient-mean difference was {fmt('zero','overall','delta_true_logit_drop')}. Class-specific effects were {fmt('zero','Meningioma','delta_true_logit_drop')} for Meningioma, {fmt('zero','Glioma','delta_true_logit_drop')} for Glioma, and {fmt('zero','Pituitary','delta_true_logit_drop')} for Pituitary. Intervals were obtained from 10,000 patient-bootstrap resamples (seed 42), preserving paired tumor/control responses and within-patient slices. No formal p-values were generated.

A prespecified secondary intervention used the median normalized intensity in a five-pixel external non-tumor neighborhood of each target mask. Its patient-mean paired effect was {fmt('local_median','overall','delta_true_logit_drop')}. The findings provide exploratory evidence of {outcome} under the primary tested intervention, rather than proof of causal dependence or clinical validity. Anatomical differences between tumor and control locations, imperfect exclusion of non-brain structures, artificial occlusion boundaries, and dependence on replacement strategy limit interpretation. The frozen model, preprocessing, and explanation settings were not modified using these findings.
'''
    (out/'PUBLICATION_OCCLUSION_SUBSECTION.md').write_text(prose,encoding='utf-8')
    report+='\n\n## Publication-ready subsection\n\n'+prose
    (out/'PHASE15_1_OCCLUSION_REPORT.md').write_text(report,encoding='utf-8')
    save(summary,out/'tables/all_effect_summaries.csv')
    for name,frame in [('overall_effects',summary[summary.group=='overall']),('class_effects',summary[summary.group.isin(CLASSES)]),('correlations',correlations)]:
        (out/'tables'/f'{name}.tex').write_text(frame.to_latex(index=False,float_format=lambda x:f'{x:.6f}',escape=True),encoding='utf-8')
