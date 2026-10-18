"""Logit Grad-CAM, localization, and auxiliary morphology measurements."""
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt

LAYERS = ('stage3', 'stage4')
THRESHOLDS = (.3, .5, .7)
EPSILON = 1e-8

def state_digest(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def resize(array,shape):
    tensor=torch.as_tensor(np.asarray(array),dtype=torch.float32).reshape(1,1,*array.shape)
    return F.interpolate(tensor,size=shape,mode='bilinear',align_corners=False)[0,0].numpy()

def normalize(array):
    array=np.maximum(np.asarray(array,dtype=np.float64),0)
    assert np.isfinite(array).all()
    maximum=float(array.max())
    return array/maximum if maximum>0 else np.zeros_like(array)

class GradCAM:
    """Scoped forward hooks; autograd.grad obtains gradients without parameter .grad."""
    def __init__(self,model):
        self.model=model;self.features={};self.native={};self.handles=[]
    def __enter__(self):
        for path in LAYERS:
            self.handles.append(self.model.get_submodule(path).register_forward_hook(self._capture(path,self.features)))
        for key in ['region','boundary','distance']:
            path=f'morphology_head.{key}_predictor'
            self.handles.append(self.model.get_submodule(path).register_forward_hook(self._capture(key,self.native)))
        return self
    @staticmethod
    def _capture(key,store):
        def hook(module,args,output):store[key]=output
        return hook
    def compute(self,images,true_labels):
        assert not self.model.training
        self.features.clear();self.native.clear()
        images=images.detach().requires_grad_(True)
        outputs=self.model(images,return_aux=True)
        logits=outputs['logits'];pred=logits.argmax(1);wrong=pred.ne(true_labels)
        features=[self.features[k] for k in LAYERS]
        gradient=torch.autograd.grad(logits.gather(1,pred[:,None]).sum(),features,retain_graph=bool(wrong.any()))
        predicted={k:torch.relu((g.mean((2,3),keepdim=True)*a).sum(1)).detach().cpu().numpy() for k,g,a in zip(LAYERS,gradient,features)}
        true={}
        if bool(wrong.any()):
            gradient=torch.autograd.grad(logits[wrong].gather(1,true_labels[wrong,None]).sum(),features)
            true={k:torch.relu((g.mean((2,3),keepdim=True)*a).sum(1)).detach().cpu().numpy() for k,g,a in zip(LAYERS,gradient,features)}
        result=dict(probabilities=logits.detach().softmax(1).cpu().numpy(),predicted=pred.detach().cpu().numpy(),
                    cam_pred=predicted,cam_true=true,native={k:v.detach().cpu().numpy()[:,0] for k,v in self.native.items()},
                    region=outputs['region_logits'].detach().sigmoid().cpu().numpy()[:,0],
                    boundary=outputs['boundary_logits'].detach().sigmoid().cpu().numpy()[:,0],
                    distance=outputs['distance'].detach().cpu().numpy()[:,0],
                    baa=outputs['boundary_attention'].detach().cpu().numpy()[:,0],
                    feature_shapes={k:list(self.features[k].shape[1:]) for k in LAYERS})
        self.features.clear();self.native.clear()
        assert all(p.grad is None for p in self.model.parameters())
        return result
    def __exit__(self,*args):
        for handle in self.handles:handle.remove()
        self.handles.clear();self.features.clear();self.native.clear()

def overlap(pred,target):
    pred=np.asarray(pred,bool);target=np.asarray(target,bool)
    tp=int((pred&target).sum());npred=int(pred.sum());ntarget=int(target.sum())
    assert ntarget>0
    return dict(dice=2*tp/(npred+ntarget),iou=tp/(npred+ntarget-tp),precision=tp/npred if npred else 0.,recall=tp/ntarget)

def localization(heatmap,mask):
    heatmap=normalize(heatmap);mask=np.asarray(mask,bool)
    assert mask.any() and (~mask).any() and heatmap.shape==mask.shape
    total=heatmap.sum();empty=total==0
    inside=float(heatmap[mask].mean());outside=float(heatmap[~mask].mean())
    result=dict(attention_mass=float(heatmap[mask].sum()/total) if not empty else 0.,
                activation_ratio=inside/(outside+EPSILON),peak_inside=float(mask.flat[int(heatmap.argmax())]) if not empty else 0.,
                zero_map=float(empty),tumor_area_fraction=float(mask.mean()))
    for threshold in THRESHOLDS:
        metrics=overlap(heatmap>=threshold,mask)
        result[f'dice_{threshold:.1f}']=metrics['dice'];result[f'iou_{threshold:.1f}']=metrics['iou']
    assert all(np.isfinite(v) for v in result.values())
    return result

def morphology(region,boundary,distance,targets):
    target_region=targets['region'][0].astype(bool);target_boundary=targets['boundary'][0].astype(bool)
    pred_boundary=boundary>=.5
    region_metrics=overlap(region>=.5,target_region)
    boundary_metrics={'dice':overlap(pred_boundary,target_boundary)['dice']}
    # Euclidean matching in the frozen 256x256 grid, without thinning or threshold tuning.
    precision=float((distance_transform_edt(~target_boundary)[pred_boundary]<=2).mean()) if pred_boundary.any() else 0.
    recall=float((distance_transform_edt(~pred_boundary)[target_boundary]<=2).mean()) if pred_boundary.any() else 0.
    boundary_metrics.update(f1_2px=2*precision*recall/(precision+recall) if precision+recall else 0.,precision_2px=precision,recall_2px=recall)
    err=distance.astype(float)-targets['distance'][0]
    distance_metrics=dict(mae=float(np.abs(err).mean()),rmse=float(np.sqrt(np.square(err).mean())),
                          tumor_mae=float(np.abs(err[target_region]).mean()),background_mae=float(np.abs(err[~target_region]).mean()))
    return region_metrics,boundary_metrics,distance_metrics

def synthetic_tests():
    mask=np.zeros((8,8),bool);mask[2:6,2:6]=True
    metrics=localization(mask.astype(float),mask)
    assert metrics['attention_mass']==1 and metrics['peak_inside']==1
    assert metrics['dice_0.5']==1 and metrics['iou_0.5']==1
    uniform=localization(np.ones((8,8)),mask)
    assert uniform['attention_mass']==.25 and abs(uniform['activation_ratio']-1)<1e-7
    zero=localization(np.zeros((8,8)),mask)
    assert zero['zero_map']==1 and zero['peak_inside']==0 and zero['attention_mass']==0
    assert overlap(np.zeros_like(mask),mask)['dice']==0
    # A known linear logit permits an independent closed-form Grad-CAM check.
    a=torch.arange(24,dtype=torch.float32).reshape(1,3,2,4).requires_grad_(True)
    weights=torch.tensor([1.,-2.,3.]).reshape(1,3,1,1)
    logit=(a*weights).sum()
    grad=torch.autograd.grad(logit,a)[0]
    cam=torch.relu((grad.mean((2,3),keepdim=True)*a).sum(1))
    torch.testing.assert_close(cam,torch.relu((weights*a).sum(1)))
    targets={'region':mask[None].astype(np.float32),'boundary':mask[None].astype(np.float32),'distance':mask[None].astype(np.float32)}
    r,b,d=morphology(mask.astype(float),mask.astype(float),mask.astype(float),targets)
    assert r['dice']==b['dice']==b['f1_2px']==1 and d['mae']==d['rmse']==0
    return dict(status='PASS',checks=['closed-form logit Grad-CAM','perfect/empty/uniform localization','fixed-threshold overlaps','perfect morphology and tolerant boundary matching'])
