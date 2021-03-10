import torch
import numpy as np

from pytorch_YOLOv4.tool.darknet2pytorch import Darknet

from utils.conv_parameters import get_conv_parameters

"""
The class object SpConv holds all parameters in order to optimize the FPGA flow for the Sparse Convolution algorithm
proposed in "Sparse-YOLO: Hardware/Software co-design of an FPGA accelerator for YOLOv2" by
Wang, Zixiao and Xu, Ke and Wu, Shuaixiao and Liu, Li and Liu, Lingzhi and Wang, Dong (2020)
"""


class SpConv:
    def __init__(self,
                 # FPGA related
                 Nin=None,  # Batch size of the input
                 Ny=None,  # Number of parallel convolutions in the column direction
                 Nscu=None,  # Number of SCUs
                 Df=None,  # Depth of line buffer in the Feature buffer
                 BW=None,  # bit wordlength
                 freq=None,  # Clock frequency
                 bandwidth=None,  # Memory bandwidth
                 # Model related
                 cfgfile=None,  # config file name
                 weightfile=None,  # weight file name
                 model=None  # also possible to provid the model directly
                 ):

        self.Nin = Nin
        self.Ny = Ny
        self.Nscu = Nscu
        self.freq = freq
        self.Df = Df
        self.BW = BW
        self.bandwidth = bandwidth
        self.peak_performance = Nin * Ny * Nscu * freq

        if cfgfile is not None:  # We assume here that we imput YOLOv4
            assert weightfile is not None

            m = Darknet(cfgfile)

            self.model = m
        else:
            if model is None:
                raise RuntimeError("Config file expected if no model is provided")
            else:
                self.model = model

    def roofline_pruning_rates(self):
        """
        Computes the layer-wise optimal pruning rate
        :return: list of pruning percentages
        """

        conv_filters, conv_widths, conv_heights, strides, kernel_sizes = get_conv_parameters(model=self.model)

        Pr = []
        j = 1
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):

                Wj = conv_widths[j]
                Hj = conv_heights[j]
                Kj = kernel_sizes[j]
                Cj = conv_filters[j - 1]
                Mj = conv_filters[j]
                Sj = strides[j]

                Pj = Kj ** 2 * Cj * Mj
                Oj = Wj * Hj * Pj

                # Compute Arithmetic intensities
                Swinj = np.floor(self.Df / (Cj * Sj))

                Gxj = np.ceil(
                    Wj / (np.floor((Swinj - Kj) / Sj) + 1)
                )
                Gyj = np.ceil(Hj / self.Ny)

                Hfj = Gxj * Gyj * Swinj * ((self.Ny - 1) * Sj - Kj) * Cj * self.BW / 8
                Hwj = Gxj * Gyj * Pj * 2 / self.Nin

                # Ideal pruning rate
                Prj = 1 - (self.peak_performance * Hfj) / (self.bandwidth * Oj - self.peak_performance * Hwj)

                if Prj < 0:
                    Prj = 0
                elif Prj > 1:
                    Prj = 1  # should choose max allowable pruning rate

                Pr.append(Prj)

                j += 1

        return Pr

    def roofline_evaluate(self, Pr):

        conv_filters, conv_widths, conv_heights, strides, kernel_sizes = get_conv_parameters(model=self.model)

        j = 1
        I = []
        GF = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                Wj = conv_widths[j]
                Hj = conv_heights[j]
                Kj = kernel_sizes[j]
                Cj = conv_filters[j - 1]
                Mj = conv_filters[j]
                Sj = strides[j]

                Pj = Kj ** 2 * Cj * Mj
                Oj = Wj * Hj * Pj

                # Compute Arithmetic intensities
                Swinj = np.floor(self.Df / (Cj * Sj))

                Gxj = np.ceil(
                    Wj / (np.floor((Swinj - Kj) / Sj) + 1)
                )
                Gyj = np.ceil(Hj / self.Ny)

                Hfj = Gxj * Gyj * Swinj * ((self.Ny - 1) * Sj - Kj) * Cj * self.BW / 8
                Hwj = Gxj * Gyj * Pj * 2 / self.Nin

                Ij = Oj * (1 - Pr[j-1]) / (Hfj + Hwj * (1 - Pr[j-1]))

                # Evaluate roofline model
                I.append(Ij)
                GF.append(min(self.bandwidth * Ij, self.peak_performance))

                j += 1

        return I, GF
