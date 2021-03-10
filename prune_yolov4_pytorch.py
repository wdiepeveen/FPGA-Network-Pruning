import os
from pruning.yolov4_weight_pruning_pytorch import do_weight_pruning

if __name__ == "__main__":

    cfgfile = os.path.join("cfg", "yolov4.cfg")
    weightfile = os.path.join("weights", "yolov4.weights")

    max_prune_rates = [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9]

    # Results folders
    model_dir = "models"
    result_dir = "results/pruning"

    for max_prune_rate in max_prune_rates:

        res_gl = do_weight_pruning(cfgfile=cfgfile,
                                   weightfile=weightfile,
                                   max_prune_rate=max_prune_rate,
                                   model_dir=model_dir,
                                   results_dir=result_dir,
                                   )

        # Pr_gl, prune_rate_gl = res_gl

