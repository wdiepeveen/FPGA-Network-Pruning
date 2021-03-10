import time
import os

import numpy as np

import torch
import torch.nn.utils.prune as prune

from pytorch_YOLOv4.tool.darknet2pytorch import Darknet
from pytorch_YOLOv4.tool.torch_utils import do_detect
from pytorch_YOLOv4.tool.utils import load_class_names, plot_boxes_cv2

use_cuda = False
local_pruning = True
global_pruning = True


def do_weight_pruning(cfgfile=None,  # config file name
                      weightfile=None,  # weight file name
                      roofline_prune_rates=None,
                      max_prune_rate=None,
                      model_dir="models",
                      results_dir="results",
                      save_orig=False,  # save original network
                      ):
    import cv2
    imgfile = "data/dog.jpg"
    model = Darknet(cfgfile)

    # m.print_network()
    model.load_weights(weightfile)
    # print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        model.cuda()

    # Save original network
    model_path = os.path.join(model_dir, "yolov4_darknet")
    res_path = os.path.join(results_dir, "prediction")

    if save_orig:
        torch.save(model, model_path + ".pt")

    num_classes = model.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (model.width, model.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    # Global weight pruning
    print("Start global pruning")
    parameters_to_prune = []
    for name, module in model.named_modules():
        # prune global connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=max_prune_rate)

    # Check sparsity
    global_sparsity = []
    global_num_weights = []
    global_prune_rates = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            sparsity = float(torch.sum(module.weight == 0))
            num_weights = float(module.weight.nelement())
            rate = sparsity/num_weights

            global_sparsity.append(sparsity)
            global_num_weights.append(num_weights)
            global_prune_rates.append(rate)
            print("Sparsity in {}: {:.2f}%".format(name, 100*rate))

    prune_rate_gl = float(sum(global_sparsity))/float(sum(global_num_weights))
    print("Global sparsity (global pruning): {:.2f}%".format(100*prune_rate_gl))

    postfix = "_global_weight_{}".format(int(100 * max_prune_rate))
    model_path_gl = model_path + postfix + ".pt"
    # Save pruned pt model
    torch.save(model, model_path_gl)

    # Make prediction with model
    start = time.time()
    boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)
    finish = time.time()
    print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes[0], savename=res_path + postfix + ".jpg", class_names=class_names)

    if roofline_prune_rates is not None:

        print("Start global + roofline pruning")
        # Loop through each prune percent
        rl_prune_rates = np.minimum(global_prune_rates, roofline_prune_rates)

        # Load the original unprunned model
        # We cannot copy otherwise it keeps reusing the same model throughout all loops
        model = Darknet(cfgfile)

        model.load_weights(weightfile)

        if use_cuda:
            model.cuda()

        local_sparsity = []
        local_num_weights = []
        j = 0
        for name, module in model.named_modules():
            # prune local connections in all 2D-conv layers
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=rl_prune_rates[j])

                # Check sparsity
                sparsity = float(torch.sum(module.weight == 0))
                num_weights = float(module.weight.nelement())

                local_sparsity.append(sparsity)
                local_num_weights.append(num_weights)
                print("Sparsity in {}: {:.2f}%".format(name, 100 * sparsity / num_weights))

                j+=1

        prune_rate_rl = float(sum(local_sparsity)) / float(sum(local_num_weights))
        print("Global sparsity (roofline global pruning): {:.2f}%".format(100 * prune_rate_rl))

        postfix = "_roofline_global_weight_{}".format(int(100*max_prune_rate))
        # Save pruned pt model
        torch.save(model, model_path + postfix + ".pt")

        # Make prediction with model
        start = time.time()
        boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

        plot_boxes_cv2(img, boxes[0], savename=res_path + postfix + ".jpg", class_names=class_names)

    if roofline_prune_rates is not None:
        return (global_prune_rates, prune_rate_gl), (rl_prune_rates, prune_rate_rl)
    else:
        return (global_prune_rates, prune_rate_gl)




