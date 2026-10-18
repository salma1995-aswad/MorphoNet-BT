"""Scientific plotting of existing tables/probabilities/cached maps only."""
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from src.phase16_publication import MAIN, NAMES, SELECTION, prediction_sources
from src.phase14_statistics import CLASSES, PROBS

COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#777777']

def build_figures(pkg,frames,scores):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
                         'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.linewidth':.6,'lines.linewidth':1.1,'savefig.dpi':600})
    def finish(fig,name,caption,sources,allocation='supplementary'):
        path=pkg.out/'figures'/allocation/name
        # Keep physical width exact; constrained layout handles labels inside the canvas.
        fig.savefig(path.with_suffix('.png'),dpi=600,facecolor='white')
        fig.savefig(path.with_suffix('.pdf'),facecolor='white')
        pkg.figures.append(dict(name=name,allocation=allocation,width_mm=float(fig.get_figwidth()*25.4),height_mm=float(fig.get_figheight()*25.4),dpi=600,
            source_files=[pkg.source(s) for s in sources],caption=caption,formats=['png','pdf'],kind='new rendering from frozen values/maps'))
        plt.close(fig)
    def sub(w,h,*args,**kwargs): return plt.subplots(*args,figsize=(w/25.4,h/25.4),layout='constrained',**kwargs)
    def panel(ax,letter):ax.text(0,1.02,letter,transform=ax.transAxes,fontweight='bold',va='bottom')

    # Architecture diagram: actual frozen tensor shapes, not a newly run forward pass.
    fig,ax=sub(174,170);ax.set_xlim(0,10);ax.set_ylim(0,13);ax.axis('off')
    def box(x,y,w,h,text,color='#E9F2F8',dash=False):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04',facecolor=color,edgecolor='#34495E',linewidth=.7,linestyle='--' if dash else '-'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
    def arrow(x,y,u,v,dash=False): ax.annotate('',xy=(u,v),xytext=(x,y),arrowprops=dict(arrowstyle='->',color='#34495E',lw=.8,linestyle='--' if dash else '-'))
    x=0.25;w=4.15
    blocks=[(11.65,'MRI only: 1 × 256 × 256'),(10.25,'Multi-scale stem (MSS)\n32 × 128 × 128'),(8.85,'MFE stage 1 (2 blocks)\n48 × 64 × 64'),(7.45,'MFE stage 2 (2 blocks)\n80 × 32 × 32'),(6.05,'MFE stage 3 (3 blocks)\n128 × 16 × 16')]
    for y,t in blocks:box(x,y,w,.94,t)
    for (y,_),(v,_) in zip(blocks,blocks[1:]):arrow(x+w/2,y,x+w/2,v+.94)
    box(5.05,7.45,4.55,.94,'Morphology shared features\n32 × 32 × 32')
    arrow(4.4,7.92,5.05,7.92)
    box(5.05,6.05,4.55,.94,'Predicted boundary logits\n1 × 32 × 32 (inference active)')
    arrow(7.325,7.45,7.325,6.99)
    box(5.05,9.05,4.55,1.03,'Region / distance predictions\nAuxiliary outputs when requested','#FFF2DA',True)
    arrow(7.325,8.39,7.325,9.05,True)
    box(5.05,10.65,4.55,1.65,'TRAINING ONLY\nGround-truth tumor masks → targets\nRegion + boundary + distance losses\nNo ground-truth input at inference','#FFF2DA',True)
    arrow(7.325,10.65,7.325,10.08,True)
    box(.25,4.25,4.15,1.05,'Global path\nDilated 5 × 5 depthwise + 1 × 1')
    box(5.05,4.25,4.55,1.05,'Local path\n3 × 3 depthwise → BAA → 1 × 1')
    arrow(2.325,6.05,2.325,5.3);arrow(3.5,6.05,6.2,5.3);arrow(7.325,6.05,7.325,5.3)
    box(2.15,2.85,5.55,.94,'Fixed arithmetic mean: (global + local) / 2\n128 × 16 × 16; no AMGF')
    arrow(2.325,4.25,3.8,3.79);arrow(7.325,4.25,6.15,3.79)
    box(2.15,1.45,5.55,.94,'MFE stage 4 (2 blocks): 192 × 8 × 8\nGlobal average pooling → 192 features')
    arrow(4.925,2.85,4.925,2.39)
    box(2.15,.05,5.55,.94,'Dropout (inactive at evaluation) → linear → 3 logits')
    arrow(4.925,1.45,4.925,.99)
    finish(fig,'A_architecture','A4 architecture verified against frozen code and shape audit. Solid boxes denote classification-time operations; dashed orange boxes denote auxiliary supervision/output operations. The shared morphology features and predicted boundary branch remain active for MRI-only classification. BAA resizes the predicted boundary logits, adds a local score and applies sigmoid gating through x(1+a). The global path is a dilated convolution, not all-to-all attention. MFE denotes the named convolutional backbone, not explicit mathematical morphology operators. Auxiliary targets never enter inference. Tensor dimensions are channels x height x width.', ['src/models/ablation.py','src/models/morphonet_bt.py','src/models/blocks.py','src/models/attention.py','results/ablation/setup/ablation_shape_audit.csv'],'main')

    labels=[NAMES[m].replace('MorphoNet-BT ','') for m in MAIN]
    measures=[('slice','Macro F1','Slice Macro-F1 (%)'),('patient','Macro F1','Patient Macro-F1 (%)'),('patient','Accuracy','Patient Accuracy (%)')]
    fig,axs=sub(174,85,1,3,sharey=True)
    for i,(ax,(level,metric,label)) in enumerate(zip(axs,measures)):
        vals=[100*scores[m][level][metric] for m in MAIN]; ax.barh(np.arange(7),vals,color=COLORS,height=.65)
        ax.set_yticks(np.arange(7),labels);ax.set_xlim(0,100);ax.set_xticks([0,50,100]);ax.set_xlabel(label);panel(ax,chr(97+i))
    axs[0].invert_yaxis()
    finish(fig,'B_performance','Pooled OOF classification performance (percent), with full 0–100 axes. A4 is the post-hoc principal model and A5 the historical full model. No error bars are implied; patient-cluster intervals are provided in S04. '+SELECTION,[p for m in MAIN for p in prediction_sources(m)],'main')
    for level,metric,label in measures:
        fig,ax=sub(84,80);vals=[100*scores[m][level][metric] for m in MAIN]
        ax.barh(np.arange(7),vals,color=COLORS,height=.65);ax.set_yticks(np.arange(7),labels);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel(label)
        finish(fig,'B_'+level+'_'+metric.replace(' ','_'),'Single-column alternative to B: '+label+'. Full 0–100 axis. '+SELECTION,[p for m in MAIN for p in prediction_sources(m)])

    fig,axs=sub(174,82,1,2)
    for ax,level,letter in zip(axs,['slice','patient'],'ab'):
        d=frames['A4'][level];cm=confusion_matrix(d.true_label,d.predicted_label,labels=[0,1,2]);ax.imshow(cm,cmap='Blues')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{cm[i,j]}\n{100*cm[i,j]/cm[i].sum():.1f}%',ha='center',va='center',color='white' if cm[i,j]>cm.max()*.5 else 'black')
        ax.set_xticks(range(3),['Men','Gli','Pit']);ax.set_yticks(range(3),['Men','Gli','Pit']);ax.set_xlabel('Predicted class');ax.set_ylabel('True class');panel(ax,letter+'  '+level.capitalize())
    finish(fig,'C_confusion','A4 confusion matrices: (a) 3,064 slices and (b) 233 patients. Each cell shows a count and its percentage within the true-class row. Men = Meningioma, Gli = Glioma, Pit = Pituitary. Patient probabilities are means over each patient’s slices. '+SELECTION,prediction_sources('A4'))
    fig,axs=sub(174,83,1,2)
    for ax,level,letter in zip(axs,['slice','patient'],'ab'):
        d=frames['A4'][level]
        for k,cls in enumerate(CLASSES):
            fpr,tpr,_=roc_curve(d.true_label==k,d[PROBS[k]]);a=roc_auc_score(d.true_label==k,d[PROBS[k]])
            ax.plot(fpr,tpr,color=COLORS[k],label=f'{cls}: {a:.4f}')
        ax.plot([0,1],[0,1],color='#777777',ls='--',lw=.7);ax.set(xlim=(0,1),ylim=(0,1),xlabel='False-positive rate',ylabel='True-positive rate')
        ax.legend(loc='lower right',fontsize=8,frameon=False);panel(ax,letter+'  '+level.capitalize());ax.text(.05,.53,f'Macro AUC = {scores["A4"][level]["Macro ROC-AUC"]:.4f}',transform=ax.transAxes,fontsize=8)
    finish(fig,'D_ROC','A4 one-vs-rest ROC curves from frozen probabilities: (a) slices, (b) patients. Legends report class AUC; macro AUC is the unweighted mean of three binary AUCs. Curves are descriptive and do not define a new operating threshold. '+SELECTION,prediction_sources('A4'))

    fig,ax=sub(84,78);variants=[f'A{i}' for i in range(6)]
    for level,color,marker in [('slice',COLORS[0],'o'),('patient',COLORS[1],'s')]:ax.plot(variants,[100*scores[m][level]['Macro F1'] for m in variants],marker=marker,color=color,label=level.capitalize())
    ax.axvspan(3.85,4.15,color='#DDDDDD',zorder=-1);ax.set_ylim(0,100);ax.set_ylabel('Pooled Macro-F1 (%)');ax.set_xlabel('Cumulative variant (A5 adds AMGF)');ax.legend(loc='lower left',frameon=False)
    finish(fig,'E_ablation','Cumulative ablation Macro-F1 with a full percentage axis. Grey marks the post-hoc selected A4. A0 core; A1 adds MSS; A2 morphology supervision; A3 BAA; A4 dual paths and fixed mean fusion; A5 AMGF. Components were not independently randomized or factorially tested; A3 and A5 show decreases. '+SELECTION,[p for m in variants for p in prediction_sources(m)],'main')
    d=pkg.tables['T06_exploratory_statistics'];fig,ax=sub(84,87)
    y=np.arange(len(d));ax.errorbar(d.effect_pp,y,xerr=np.vstack([d.effect_pp-d.ci_low_pp,d.ci_high_pp-d.effect_pp]),fmt='o',color=COLORS[0],capsize=2)
    ax.set_yticks(y,[NAMES.get(m,m).replace('MorphoNet-BT ','') for m in d.right]);ax.invert_yaxis();ax.axvline(0,color='#555555',ls='--',lw=.8);ax.set_xlabel('A4 − comparator\nAccuracy difference (pp)');ax.set_xlim(-2,13);ax.set_xticks([0,4,8,12]);ax.grid(axis='x',alpha=.15)
    finish(fig,'F_patient_effects','Patient Accuracy differences for A4 minus comparator (percentage points). Error bars are paired patient-bootstrap 95% percentile CIs (10,000 replicates, seed 42), not multiplicity-adjusted confidence intervals. Exact McNemar with the frozen 10-test Holm family supports differences only versus ResNet50 and MobileNetV3-Small at 0.05. CI exclusion of zero must not substitute for adjusted p-values. '+SELECTION,['results/statistical_analysis/paired_effects_with_ci.csv','results/statistical_analysis/exact_mcnemar_holm.csv'],'main')

    # Cached arrays are already generated Phase 15 outputs. No raw MRI is opened.
    for suffix,ids,selection in [('representative',[66,2570,1315],'previous seeded random-correct selection'),('failures',[36,2172,1566],'previous high-confidence-error selection')]:
        fig,axs=sub(174,118,3,5)
        keys=['cam_pred_stage3','baa','region_logits_low','boundary_logits_low','distance_logits_low']
        titles=['Grad-CAM','BAA','Region','Boundary','Distance']
        sources=[]
        for i,sid in enumerate(ids):
            source=f'results/explainability/maps/{sid}.npz';sources.append(source);z=np.load(pkg.root/source)
            record=frames['A4']['slice'].loc[frames['A4']['slice'].sample_id==str(sid)].iloc[0]
            for j,key in enumerate(keys):
                a=z[key].astype(float)
                if 'logits' in key:a=1/(1+np.exp(-np.clip(a,-700,700)))
                if key.startswith('cam_'): a=a/a.max() if a.max()>0 else np.zeros_like(a)
                axs[i,j].imshow(a,cmap='cividis',vmin=0,vmax=1);axs[i,j].set_xticks([]);axs[i,j].set_yticks([])
                if i==0:axs[i,j].set_title(titles[j],fontsize=8)
                if j==0:axs[i,j].set_ylabel(f'{record.true_class}\nID {sid}',fontsize=8)
        finish(fig,'G_'+suffix+'_cached_maps','Frozen Phase 15 maps for '+selection+'. Rows are Meningioma, Glioma and Pituitary. Cached predicted-class Grad-CAM and BAA are 16 x 16; auxiliary region/boundary/distance logits are 32 x 32 and shown after sigmoid. Grad-CAM is divided by its positive native peak (zero maps stay zero), BAA displays its original sigmoid gate. All panels use a 0–1 color scale (dark low, light high); the quantities have different meanings and are not equivalent localization methods. No MRI or model inference was performed in Phase 16. Original MRI overlays and full-resolution morphology comparisons are preserved in the accompanying G source figures. Sample IDs preserve the historical selection; successful cases do not represent failure prevalence.',sources+['results/explainability/tables/representative_sample_manifest.csv'],'supplementary')
    for source_name in ['figure1_representative_gradcam','figure2_correct_vs_incorrect','figure3_morphology_predictions']:
        source=f'results/explainability/figures/{source_name}';name='G_source_'+source_name
        for ext in ['png','pdf']:
            pkg.source(source+'.'+ext);shutil.copyfile(pkg.root/(source+'.'+ext),pkg.out/'figures/supplementary'/(name+'.'+ext))
        pkg.figures.append(dict(name=name,allocation='supplementary',source_files=[source+'.png',source+'.pdf'],caption='Unchanged Phase 15 source visualization, including original MRI overlays and target/prediction comparisons. Historical layout and selection retained; use at its native large format in the supplement. No Phase 16 MRI inference. Not intended for shrinking to single-column width.',formats=['png','pdf'],kind='byte-identical historical figure'))

    # Re-layout already-rendered panels, without reading source MRI or changing maps.
    source='results/explainability/figures/figure1_representative_gradcam.png'
    original=Image.open(pkg.root/source).convert('RGB');scale=original.width/2048
    fig,axs=sub(174,129,3,4)
    crops=[]
    for i,(top,bottom) in enumerate([(84,265),(336,517),(588,768)]):
        for j,(left,right) in enumerate([(12,192),(271,450),(788,968),(1306,1486)]):
            bounds=tuple(round(v*scale) for v in (left,top,right,bottom));crops.append(bounds)
            axs[i,j].imshow(original.crop(bounds));axs[i,j].set_xticks([]);axs[i,j].set_yticks([])
            if i==0:axs[i,j].set_title(['MRI','Tumor annotation','Grad-CAM overlay','Region probability'][j],fontsize=8)
            if j==0:axs[i,j].set_ylabel(f'{CLASSES[i]}\nID {[66,2570,1315][i]}',fontsize=8)
    finish(fig,'G_representative_overlays','Previously seeded random-correct examples from Phase 15: Meningioma (66), Glioma (2570), Pituitary (1315). MRI, annotated region, predicted-class stage3 Grad-CAM overlay and auxiliary region probability are byte-derived panel crops of the existing rendered Phase 15 figure, re-laid out for readable print labels. The MRI display uses its historical 1st–99th percentile contrast and overlay alpha 0.45; probability maps retain the original 0–1 scale. No raw MRI, new maps, thresholds or predictions were generated. These successful examples are not representative of error prevalence; failures remain in the supplement. Auxiliary predictions are not clinically validated segmentation.',[source,'results/explainability/tables/representative_sample_manifest.csv'],'main')
    pkg.figures[-1]['historical_panel_crop_boxes_pixels']=crops

    d=pkg.tables['T08_occlusion_sensitivity'];fig,ax=sub(84,87)
    for i,strategy in enumerate(['zero','local_median']):
        part=d[d.Replacement==strategy];y=np.arange(4)+(i-.5)*.16
        ax.errorbar(part.Delta,y,xerr=np.vstack([part.Delta-part.CI_low,part.CI_high-part.Delta]),fmt=['o','s'][i],color=COLORS[i],capsize=2,label=strategy.replace('_',' '))
    ax.set_yticks(range(4),['Overall','Meningioma','Glioma','Pituitary']);ax.invert_yaxis();ax.axvline(0,color='#555555',ls='--',lw=.8);ax.set_xlabel('Paired ΔSensitivity (logit units)');ax.legend(loc='lower right',frameon=False)
    finish(fig,'H_occlusion','Patient-equal paired true-class logit sensitivity: tumor drop minus mean matched-control drop. Error bars are 95% patient-bootstrap percentile CIs (10,000 replicates, seed 42); replacements use the same frozen control locations. Overall: 233 patients/3,049 slices; Meningioma 82/707, Glioma 89/1,412, Pituitary 62/930. Controls match tumor shape and area but are not anatomically or intensity matched; dependence and out-of-distribution perturbations limit causal interpretation. No formal subgroup p-values.',['results/explainability/occlusion_audit/tables/all_effect_summaries.csv'])
