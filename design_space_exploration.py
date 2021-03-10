import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from fpga.sparse_convolution import SpConv
from pruning.yolov4_weight_pruning_pytorch import do_weight_pruning

if __name__ == "__main__":

    # Setup FPGA parameters
    max_prune_rate = 0.5  # a feasible prune rate for your application (found prior to the design process)

    # Tunable parameters (To optimize in this script)
    Nin = 40  # Batch size of the input
    Ny = 13  # Number of parallel convolutions in the column direction
    Nscu = 2  # Number of SCUs
    Df = 3840  # Depth of line buffer in the Feature buffer
    BW = 8  # aimed bit word length after quantizing

    # Board parameters
    freq = 0.2e9  # Clock frequency
    bandwidth = 4e8  # Memory bandwidth

    cfgfile = os.path.join("cfg", "yolov4.cfg")
    weightfile = os.path.join("weights", "yolov4.weights")

    # Results folders
    model_dir = "models"
    result_dir = "results/design"

    sparse_conv = SpConv(Nin=Nin, Ny=Ny, Nscu=Nscu, Df=Df, BW=BW, freq=freq, bandwidth=bandwidth,
                         # Model related
                         cfgfile=cfgfile, weightfile=weightfile)

    Pr = sparse_conv.roofline_pruning_rates()

    # Compute base case
    I_base, GF_base = sparse_conv.roofline_evaluate([0]*len(Pr))

    Is = np.linspace(start=0, stop=max(max(I_base), 200), num=1000)
    # Plot base case results
    plt.figure()
    # Line plot of roofline
    plt.plot(Is, np.minimum(bandwidth * Is, sparse_conv.peak_performance))
    # Scatter plot of results
    plt.scatter(I_base, GF_base)
    plt.savefig(os.path.join(result_dir, "result_roofline_without_weight_pruning" + ".eps"))

    prune_rates_gl = []
    prune_rates_rl = []

    res_gl, res_rl = do_weight_pruning(cfgfile=cfgfile,
                                       weightfile=weightfile,
                                       roofline_prune_rates=Pr,
                                       max_prune_rate=max_prune_rate,
                                       model_dir=model_dir,
                                       results_dir=result_dir,
                                       save_orig=True,
                                       )

    Pr_gl, prune_rate_gl = res_gl
    Pr_rl, prune_rate_rl = res_rl

    prune_rates_gl.append(prune_rate_gl)
    prune_rates_rl.append(prune_rate_rl)

    # Evaluate pruning
    I_gl, GF_gl = sparse_conv.roofline_evaluate(Pr_gl)
    I_rl, GF_rl = sparse_conv.roofline_evaluate(Pr_rl)

    # Plot Global weight pruning results
    plt.figure()
    # Line plot of roofline
    plt.plot(Is, np.minimum(bandwidth * Is, sparse_conv.peak_performance))
    # Scatter plot of results
    plt.scatter(I_gl, GF_gl)
    plt.savefig(os.path.join(result_dir, "result_{}p_roofline_global_weight_pruning".format(int(max_prune_rate * 100)) + ".eps"))

    # Plot Global weight + roofline pruning
    plt.figure()
    # Line plot of roofline
    plt.plot(Is, np.minimum(bandwidth * Is, sparse_conv.peak_performance))
    # Scatter plot of results
    plt.scatter(I_rl, GF_rl)
    plt.savefig(os.path.join(result_dir, "result_{}p_roofline_global_roofline_weight_pruning".format(
        int(max_prune_rate * 100)) + ".eps"))

    # Density plot of points along I axis
    # TODO replace distplot for a non-depricated function
    # TODO get either roofline in plots or density in previous plots
    plt.figure()
    sns.distplot(I_base, hist=False, kde=True,
                 kde_kws={"linewidth": 3},
                 # kde_kws={"shade": True, "linewidth": 3},
                 label="no pruning")
    sns.distplot(I_gl, hist=False, kde=True,
                 kde_kws={"linewidth": 3},
                 # kde_kws={"shade": True, "linewidth": 3},
                 label="global weight")
    sns.distplot(I_rl, hist=False, kde=True,
                 kde_kws={"linewidth": 3},
                 # kde_kws={"shade": True, "linewidth": 3},
                 label="roofline")
    plt.legend(prop={"size": 16})
    plt.xlim((0, max(max(I_base), 200)))
    plt.savefig(os.path.join(result_dir, "result_{}p_roofline_density".format(
        int(max_prune_rate * 100)) + ".eps"))

