from evaluation.metrics_new import *
from evaluation.beat_align import *
from evaluation.pfc import *
from evaluation.metrics_new import quantized_metrics, calc_and_save_feats

log_file = "result/log/bailando.log"
    
if __name__ == '__main__':

    #TODO: fix the path
    gt_root = 'result\Bailando\gt'
    
    # calc_and_save_feats(gt_root)

    pred_roots = [
        'result\Bailando\ep000010'
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