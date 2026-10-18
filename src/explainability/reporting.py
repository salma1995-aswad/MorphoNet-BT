"""Patient-cluster summaries, descriptive associations and transparent figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from scipy.ndimage import binary_erosion
from src.phase14_statistics import read, save, dump, draws, markdown_table
from src.data.dataset import load_checked_sample
from src.data.preprocessing import preprocess
from .core import resize, normalize, LAYERS

METRICS=['attention_mass','activation_ratio','peak_inside','dice_0.3','iou_0.3','dice_0.5','iou_0.5','dice_0.7','iou_0.7']
LABELS={'attention_mass':'Tumor attention mass','activation_ratio':'Tumor/background ratio','peak_inside':'Peak inside tumor',
        'dice_0.3':'Dice at 0.3','iou_0.3':'IoU at 0.3','dice_0.5':'Dice at 0.5','iou_0.5':'IoU at 0.5','dice_0.7':'Dice at 0.7','iou_0.7':'IoU at 0.7'}

def summarize(frame,metrics,pids,counts,identity):
    rows=[]
    grouped=frame.groupby('PID')
    sums=grouped[metrics].sum().reindex(pids,fill_value=0).to_numpy(float)
    sizes=grouped.size().reindex(pids,fill_value=0).to_numpy(float)
    means=np.divide(sums,sizes[:,None],out=np.zeros_like(sums),where=sizes[:,None]>0)
    for estimand,values,weights in [('slice_weighted',sums,sizes),('patient_equal',means,(sizes>0).astype(float))]:
        den=counts@weights
        assert (den>0).all()
        replicates=(counts@values)/den[:,None]
        lo,hi=np.percentile(replicates,[2.5,97.5],axis=0)
        estimate=values.sum(axis=0)/weights.sum()
        for k,metric in enumerate(metrics):
            rows.append(dict(**identity,estimand=estimand,metric=metric,mean=estimate[k],ci_low=lo[k],ci_high=hi[k],slices=len(frame),patients=frame.PID.nunique()))
    return rows

def summary_tables(out,cams,counts,pids):
    allrows=[];foldrows=[]
    for layer in LAYERS:
        for target in ['predicted','true']:
            frame=cams[(cams.layer==layer)&(cams.target==target)]
            groups=[('overall',frame)]
            if target=='predicted':
                groups += [(c,frame[frame.true_class==c]) for c in ['Meningioma','Glioma','Pituitary']]
                groups += [('correct',frame[frame.correct==1]),('incorrect',frame[frame.correct==0])]
            for group,part in groups:
                allrows+=summarize(part,METRICS,pids,counts,dict(layer=layer,target=target,group=group))
            for fold in range(1,6):
                part=frame[frame.fold==fold]
                for estimand,values in [('slice_weighted',part[METRICS]),('patient_equal',part.groupby('PID')[METRICS].mean())]:
                    for metric,value in values.mean().items():foldrows.append(dict(layer=layer,target=target,fold=fold,estimand=estimand,metric=metric,mean=value,slices=len(part),patients=part.PID.nunique()))
    summary=pd.DataFrame(allrows);folds=pd.DataFrame(foldrows)
    save(summary[(summary.group=='overall')],out/'tables/gradcam_localization_summary.csv')
    save(summary[summary.group.isin(['Meningioma','Glioma','Pituitary'])],out/'tables/gradcam_per_class_summary.csv')
    save(summary[summary.group.isin(['correct','incorrect'])],out/'tables/gradcam_correct_vs_incorrect.csv')
    save(folds,out/'tables/explainability_fold_summary.csv')
    patients=cams.groupby(['layer','target','PID','fold','true_class'],sort=True).agg(**{m:(m,'mean') for m in METRICS},slices=('sample_id','size'),correct_slice_fraction=('correct','mean'),zero_map_fraction=('zero_map','mean')).reset_index()
    save(patients,out/'tables/gradcam_localization_per_patient.csv')
    return summary,folds,patients

def morphology_summaries(out,pids,counts):
    rows=[];foldrows=[];wide=None
    for kind,metrics in [('region',['dice','iou','precision','recall']),('boundary',['dice','f1_2px','precision_2px','recall_2px']),('distance',['mae','rmse','tumor_mae','background_mae'])]:
        frame=read(out/f'tables/morphology_{kind}_metrics.csv')
        groups=[('overall',frame)]+[(c,frame[frame.true_class==c]) for c in ['Meningioma','Glioma','Pituitary']]+[('correct',frame[frame.correct==1]),('incorrect',frame[frame.correct==0])]
        for group,part in groups:rows+=summarize(part,metrics,pids,counts,dict(head=kind,group=group))
        for fold in range(1,6):
            part=frame[frame.fold==fold]
            for estimand,values in [('slice_weighted',part[metrics]),('patient_equal',part.groupby('PID')[metrics].mean())]:
                for metric,value in values.mean().items():foldrows.append(dict(head=kind,fold=fold,estimand=estimand,metric=metric,mean=value))
        patient=frame.groupby('PID')[metrics].mean().add_prefix(kind+'_')
        wide=patient if wide is None else wide.join(patient)
    summary=pd.DataFrame(rows);save(summary,out/'tables/morphology_summary.csv');save(pd.DataFrame(foldrows),out/'tables/morphology_fold_summary.csv')
    save(wide.reset_index(),out/'tables/morphology_per_patient.csv')
    return summary,wide

def associations(out,patients,wide):
    p=patients[(patients.layer=='stage3')&(patients.target=='predicted')].set_index('PID').join(wide).sort_index()
    rng=np.random.default_rng(42);indices=rng.integers(0,len(p),size=(10000,len(p)))
    rows=[]
    for metric in ['attention_mass','activation_ratio','correct_slice_fraction']:
        x,y=p.region_dice.to_numpy(),p[metric].to_numpy()
        a,b=rankdata(x[indices],axis=1),rankdata(y[indices],axis=1)
        a-=a.mean(axis=1,keepdims=True);b-=b.mean(axis=1,keepdims=True)
        boot=np.sum(a*b,axis=1)/np.sqrt(np.sum(a*a,axis=1)*np.sum(b*b,axis=1))
        assert np.isfinite(boot).all()
        lo,hi=np.percentile(boot,[2.5,97.5])
        rows.append(dict(x='patient_mean_region_dice',y=metric,spearman_rho=float(spearmanr(x,y).statistic),ci_low=lo,ci_high=hi,n_patients=len(p),bootstrap_iterations=10000))
    df=pd.DataFrame(rows);save(df,out/'tables/patient_morphology_attention_correlations.csv');return df

def diagnostic_failures(out,cams,selection):
    primary=cams[(cams.layer=='stage3')&(cams.target=='predicted')]
    errors=primary[primary.correct==0];extra=[]
    for cls in ['Meningioma','Glioma','Pituitary']:
        part=errors[errors.true_class==cls].copy();part['numeric_id']=part.sample_id.astype(int)
        for label,ascending in [('lowest_attention_mass_error',True),('highest_attention_mass_error',False)]:
            if part.empty:continue
            r=part.sort_values(['attention_mass','numeric_id'],ascending=[ascending,True]).iloc[0]
            extra.append(dict(sample_id=r.sample_id,PID=r.PID,fold=r.fold,true_class=r.true_class,predicted_class=r.predicted_class,confidence=r.confidence,selection=label,selection_uses_overlap=True))
    manifest=pd.concat([selection,pd.DataFrame(extra)],ignore_index=True)
    save(manifest,out/'tables/representative_sample_manifest.csv')
    counts=[]
    for cls,part in [('Overall',primary)]+[(c,primary[primary.true_class==c]) for c in ['Meningioma','Glioma','Pituitary']]:
        for correctness in [1,0]:
            d=part[part.correct==correctness]
            counts.append(dict(group=cls,correct=correctness,n_slices=len(d),n_patients=d.PID.nunique(),majority_activation_outside=int((d.attention_mass<.5).sum()),majority_activation_inside=int((d.attention_mass>=.5).sum()),peak_inside=int(d.peak_inside.sum()),zero_maps=int(d.zero_map.sum())))
    df=pd.DataFrame(counts);save(df,out/'tables/failure_pattern_counts.csv')
    confusion=pd.crosstab(primary.true_class,primary.predicted_class).reindex(index=['Meningioma','Glioma','Pituitary'],columns=['Meningioma','Glioma','Pituitary'],fill_value=0)
    save(confusion.reset_index(),out/'tables/class_confusion.csv')
    # Matched error-only contrasts keep target policy explicit.
    contrast=[]
    for layer in LAYERS:
        a=cams[(cams.layer==layer)&(cams.target=='true')].set_index('sample_id')
        b=cams[(cams.layer==layer)&(cams.target=='predicted')&(cams.correct==0)].set_index('sample_id').loc[a.index]
        for m in METRICS:contrast.append(dict(layer=layer,metric=m,n_errors=len(a),predicted_target_mean=b[m].mean(),true_target_mean=a[m].mean(),true_minus_predicted=a[m].mean()-b[m].mean()))
    save(pd.DataFrame(contrast),out/'tables/error_true_vs_predicted_target.csv')
    return manifest,df,confusion

def sample_visual(root,out,sid,records,config):
    record=records.loc[str(sid)].to_dict();record['sample_id']=str(sid)
    raw,mask=load_checked_sample(record,True);arrays,_=preprocess(raw,mask,'test',config)
    maps=np.load(out/f'maps/{sid}.npz',allow_pickle=False)
    cam=normalize(resize(maps['cam_pred_stage3'],raw.shape))
    display=np.asarray(raw,float);lo,hi=np.percentile(display,[1,99]);display=np.clip((display-lo)/max(hi-lo,1e-8),0,1)
    native={key:1/(1+np.exp(-resize(maps[f'{key}_logits_low'],(256,256)))) for key in ['region','boundary','distance']}
    return raw,mask.astype(bool),arrays,maps,cam,display,native

def make_figures(root,out,manifest,summary,morphsummary,only_morphology=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import yaml
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    records=pd.concat([read(root/f'splits/fold_{f}_test.csv') for f in range(1,6)]).set_index('sample_id')
    config=yaml.safe_load((root/'results/architecture/preprocessing_v1_approved.yaml').read_text())
    folder=out/'figures';folder.mkdir(exist_ok=True)
    def finish(fig,name):
        for ext in ['png','pdf']:fig.savefig(folder/f'{name}.{ext}',dpi=320,bbox_inches='tight')
        plt.close(fig)
    def panel(rows,name,title):
        fig,axes=plt.subplots(len(rows),8,figsize=(18,2.4*len(rows)),squeeze=False,layout='constrained')
        headers=['MRI','GT region','Predicted-class CAM','CAM overlay','GT boundary','Region probability','Boundary probability','True-class CAM / error']
        for i,r in enumerate(rows.itertuples()):
            raw,mask,arrays,maps,cam,gray,native=sample_visual(root,out,r.sample_id,records,config)
            boundary=mask&~binary_erosion(mask,structure=np.ones((3,3)))
            a=axes[i]
            for ax in a:ax.set_axis_off()
            a[0].imshow(gray,cmap='gray',vmin=0,vmax=1)
            a[1].imshow(gray,cmap='gray');a[1].imshow(np.ma.masked_where(~mask,mask),cmap='autumn',vmin=0,vmax=1,alpha=.45)
            a[2].imshow(cam,cmap='inferno',vmin=0,vmax=1)
            a[3].imshow(gray,cmap='gray');a[3].imshow(cam,cmap='inferno',vmin=0,vmax=1,alpha=.45)
            a[4].imshow(gray,cmap='gray');a[4].imshow(np.ma.masked_where(~boundary,boundary),cmap='autumn',vmin=0,vmax=1)
            a[5].imshow(resize(native['region'],raw.shape),cmap='viridis',vmin=0,vmax=1)
            a[6].imshow(resize(native['boundary'],raw.shape),cmap='viridis',vmin=0,vmax=1)
            if 'cam_true_stage3' in maps:a[7].imshow(normalize(resize(maps['cam_true_stage3'],raw.shape)),cmap='inferno',vmin=0,vmax=1)
            else:a[7].text(.5,.5,'Same target\n(correct)',ha='center',va='center',transform=a[7].transAxes)
            a[0].text(0,-.08,f'Sample {r.sample_id}; fold {r.fold}\nTrue: {r.true_class}\nPred: {r.predicted_class} ({r.confidence:.3f})',transform=a[0].transAxes,fontsize=8,va='top')
            if i==0:
                for ax,h in zip(a,headers):ax.set_title(h,fontsize=9)
        fig.suptitle(title+'\nPrimary stage3; fixed scales [0,1]; overlays alpha=0.45; MRI display uses 1st-99th percentiles only',fontsize=11)
        finish(fig,name)
    if not only_morphology:
        panel(manifest[manifest.selection=='random_correct'],'figure1_representative_gradcam','Figure 1. Seeded correct examples from all three classes')
        panel(manifest[manifest.selection.isin(['low_confidence_correct','high_confidence_error'])],'figure2_correct_vs_incorrect','Figure 2. Low-confidence successes and high-confidence errors (selection before CAM)')
        # Complete per-example panels include all fixed-policy selections, not just attractive maps.
        for r in manifest.itertuples():
            panel(manifest.loc[(manifest.sample_id==r.sample_id)&(manifest.selection==r.selection)].iloc[:1],f'example_{r.sample_id}_{r.selection}',f'Selection: {r.selection.replace("_"," ")}')
    rows=manifest[manifest.selection=='random_correct']
    fig,axes=plt.subplots(len(rows),7,figsize=(15,7.5),layout='constrained')
    for i,r in enumerate(rows.itertuples()):
        raw,mask,arrays,maps,cam,gray,native=sample_visual(root,out,r.sample_id,records,config)
        values=[resize(gray,(256,256)),arrays['region'][0],native['region'],arrays['boundary'][0],native['boundary'],arrays['distance'][0],native['distance']]
        for j,(ax,val) in enumerate(zip(axes[i],values)):
            ax.imshow(val,cmap='gray' if j==0 else 'viridis',vmin=0,vmax=1);ax.set_axis_off()
            if i==0:ax.set_title(['MRI','GT region','Region probability','GT boundary','Boundary probability','GT distance','Distance prediction'][j])
        axes[i,0].set_title(f'MRI: {r.true_class}; sample {r.sample_id}',fontsize=9,loc='center')
    fig.suptitle('Figure 3. Frozen auxiliary morphology outputs on the 256x256 grid; not clinical segmentation')
    finish(fig,'figure3_morphology_predictions')
    if only_morphology:return
    fig,axes=plt.subplots(1,3,figsize=(11,4),layout='constrained')
    for ax,metric in zip(axes,['attention_mass','activation_ratio','peak_inside']):
        d=summary[(summary.target=='predicted')&(summary.group=='overall')&(summary.estimand=='slice_weighted')&(summary.metric==metric)]
        x=np.arange(len(d));ax.errorbar(x,d['mean'],yerr=[d['mean']-d.ci_low,d.ci_high-d['mean']],fmt='o',capsize=4,color='#1565a8')
        ax.set_xticks(x,d.layer);ax.set_title(LABELS[metric]);ax.grid(axis='y',alpha=.2)
        if metric!='activation_ratio':ax.set_ylim(0,1)
    fig.suptitle('Figure 4. All held-out slices: means and patient-cluster 95% intervals')
    finish(fig,'figure4_quantitative_localization')
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,metric in zip(axes,['attention_mass','activation_ratio','peak_inside']):
        d=summary[(summary.layer=='stage3')&(summary.target=='predicted')&(summary.group.isin(['Meningioma','Glioma','Pituitary']))&(summary.estimand=='slice_weighted')&(summary.metric==metric)]
        x=np.arange(len(d));ax.errorbar(x,d['mean'],yerr=[d['mean']-d.ci_low,d.ci_high-d['mean']],fmt='o',capsize=4,color='#1565a8')
        ax.set_xticks(x,d.group);ax.set_title(LABELS[metric]);ax.grid(axis='y',alpha=.2)
        if metric!='activation_ratio':ax.set_ylim(0,1)
    fig.suptitle('Figure 5. Per-class primary Grad-CAM: patient-cluster 95% intervals')
    finish(fig,'figure5_per_class_explainability')

LIMITATIONS='''Grad-CAM is a post-hoc, coarse, class-conditioned localization method, not a segmentation algorithm or proof of causal feature importance. Overlap with tumor annotations does not establish clinical validity, and a confident, incorrect classifier may attend inside the tumor. Spatial resolution is limited to 16x16 (primary) or 8x8 (secondary); interpolation does not add independent detail. MRI background, normalization and display contrast can affect interpretation. Masks are available only as evaluation targets, never model inference inputs. Slices within a patient are correlated; patient-cluster intervals are conditional on fixed trained models, omit training randomness, and do not fully capture overlapping CV training-set dependence. No independent external or prospective clinical dataset is available. A4 was selected after Phase 13 results, so these findings are exploratory and cannot tune or validate that selection. Region/boundary/distance heads are auxiliary predictions, not a separately validated clinical segmentation system. Mean distance errors over the full grid may be dominated by background, so tumor-only MAE is also reported. Fixed CAM thresholds are sensitivity settings, not test-optimized operating points. Ratios can be large when outside activation is tiny; epsilon and zero-map counts are disclosed. Primary correct/incorrect grouping uses slice correctness; patient-equal summaries average within the relevant slice subgroup, not patient-level classification correctness. Extreme failure examples are explicitly selected extremes, not representative prevalence estimates. No A5 explanation comparison was executed; it was optional and omitted to keep the scope focused and avoid additional checkpoint/compute complexity. Phase 16 was not started.'''

def build_report(root,out,cams,selection):
    from .pipeline import POLICY
    pids=sorted(cams.PID.unique());counts=draws(len(pids),10000,np.random.default_rng(42))
    np.savez_compressed(out/'metadata/shared_patient_bootstrap.npz',pids=np.array(pids),counts=counts)
    summary,folds,patients=summary_tables(out,cams,counts,pids)
    morphsummary,wide=morphology_summaries(out,pids,counts)
    corr=associations(out,patients,wide)
    manifest,failures,confusion=diagnostic_failures(out,cams,selection)
    baa=read(out/'tables/baa_localization_per_slice.csv')
    baasummary=pd.DataFrame(summarize(baa,METRICS,pids,counts,dict(map='class_agnostic_boundary_attention',group='overall')))
    save(baasummary,out/'tables/baa_localization_summary.csv')
    zero=cams.groupby(['layer','target']).agg(slices=('sample_id','size'),zero_maps=('zero_map','sum')).reset_index();save(zero,out/'tables/zero_map_audit.csv')
    make_figures(root,out,manifest,summary,morphsummary)
    tables={'gradcam_all_summaries':summary,'gradcam_fold_summaries':folds,'morphology_summary':morphsummary,'baa_summary':baasummary,'patient_correlations':corr,'failure_counts':failures,'zero_maps':zero}
    report='# Phase 15: A4 Explainability and Morphological Interpretability\n\n## Protocol\n\n'+POLICY+'\n\nStandard Grad-CAM follows [Selvaraju et al., ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html).\n\n## Scientific limitations\n\n'+LIMITATIONS
    for name,df in tables.items():
        formatted=df.copy()
        for c in formatted.select_dtypes('float').columns:formatted[c]=formatted[c].map(lambda v:f'{v:.6f}')
        report+='\n\n## '+name.replace('_',' ').title()+'\n\n'+markdown_table(formatted)
    # Publication prose contains computed values only.
    def value(metric,group='overall',estimand='slice_weighted'):
        r=summary[(summary.layer=='stage3')&(summary.target=='predicted')&(summary.group==group)&(summary.estimand==estimand)&(summary.metric==metric)].iloc[0]
        return f'{r["mean"]:.4f} (95% CI {r.ci_low:.4f}-{r.ci_high:.4f})'
    def morphvalue(head,metric):
        r=morphsummary[(morphsummary['head']==head)&(morphsummary.group=='overall')&(morphsummary.estimand=='slice_weighted')&(morphsummary.metric==metric)].iloc[0]
        return f'{r["mean"]:.4f} (95% CI {r.ci_low:.4f}-{r.ci_high:.4f})'
    prose=f'''### Explainability and morphological interpretability

We examined all 3,064 held-out MRI slices from 233 patients using the corresponding frozen A4 checkpoint for each of five patient-disjoint folds. Standard logit-targeted Grad-CAM was computed at stage3 (16x16), selected a priori on architectural grounds, and stage4 (8x8) as a secondary layer. Predicted-class maps were retained for all slices; true-class maps were additionally computed for 227 misclassified slices. Maps were bilinearly resized to original image dimensions. Tumor annotations were used only for evaluation, never as classifier inputs. Uncertainty was estimated using 10,000 patient-cluster bootstrap draws (seed 42).

For primary predicted-class maps, mean tumor attention mass was {value('attention_mass')}, the tumor-to-background activation ratio was {value('activation_ratio')}, and the peak-inside-tumor fraction was {value('peak_inside')}. Mean attention mass was {value('attention_mass','Meningioma')} for Meningioma, {value('attention_mass','Glioma')} for Glioma, and {value('attention_mass','Pituitary')} for Pituitary. At the fixed 0.5 heatmap threshold, mean Dice was {value('dice_0.5')} and IoU was {value('iou_0.5')}. Correctly and incorrectly classified slices had attention masses of {value('attention_mass','correct')} and {value('attention_mass','incorrect')}, respectively; these are descriptive associations rather than causal effects.

The frozen auxiliary region head achieved mean Dice {morphvalue('region','dice')} and IoU {morphvalue('region','iou')}. Boundary F1 with a fixed two-pixel tolerance on the 256x256 grid was {morphvalue('boundary','f1_2px')}. Normalized distance-map MAE and per-slice RMSE were {morphvalue('distance','mae')} and {morphvalue('distance','rmse')}, respectively. These exploratory results characterize spatial localization and auxiliary morphology prediction on this dataset; they do not establish causal explanations, external generalization, or clinical validity. A4 was selected after the preceding ablation evaluation, and no model or explanation setting was optimized using these findings.
'''
    (out/'PUBLICATION_EXPLAINABILITY_SUBSECTION.md').write_text(prose,encoding='utf-8')
    report+='\n\n## Publication-ready subsection\n\n'+prose+'\n\n## Representative sample selection\n\n'+markdown_table(manifest.drop(columns='PID'))
    (out/'PHASE15_EXPLAINABILITY_REPORT.md').write_text(report,encoding='utf-8')
    # A compact, full-precision table bundle for downstream scientific review.
    save(summary,out/'tables/gradcam_all_summaries.csv')
    for name,df in tables.items():
        (out/'tables'/f'{name}.tex').write_text(df.to_latex(index=False,float_format=lambda v:f'{v:.5f}',escape=True),encoding='utf-8')
    figure_files=[p.name for p in (out/'figures').glob('*') if p.suffix in ['.png','.pdf']]
    dump(out/'metadata/figure_manifest.json',dict(files=sorted(figure_files),png_dpi=320,PID_in_labels=False,primary_layer='stage3',display_only_contrast='per-image 1st-99th percentiles; model input unchanged'))
