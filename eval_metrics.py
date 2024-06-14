from evaluation.metrics_new import *
from evaluation.beat_align import *
from evaluation.pfc import *
from evaluation.metrics_new import quantized_metrics, calc_and_save_feats
import torch 
from scipy.spatial.transform import Rotation as R
from evaluation.features.kinetic import extract_kinetic_features
from evaluation.features.manual_new import extract_manual_features
from vis import SMPLSkeleton

log_file = "evaluate_result\sinmdm-footrefine-no-val\\result.log"
        
if __name__ == '__main__':

    #TODO: fix the path
    gt_root = 'evaluate_result\sinmdm-footrefine-no-val\gt'
    
    calc_and_save_feats(gt_root)

    pred_roots = [
        'evaluate_result\sinmdm-footrefine-no-val\inference'
    ]
    
    with open(log_file, 'a') as f:
        for pred_root in pred_roots:
            print(pred_root, file=f, flush=True)
            calc_and_save_feats(pred_root)
            print("FID Metrics", file=f, flush=True)
            print(quantized_metrics(pred_root, gt_root), file=f, flush=True)
            print("Beat Accuracy", file=f, flush=True)
            print(calc_ba_score(pred_root), file=f, flush=True)
            print("PFC", file=f, flush=True)
            print(calc_physical_score(pred_root), file=f, flush=True)