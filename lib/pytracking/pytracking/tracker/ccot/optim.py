import torch
import torch.nn.functional as F
from pytracking import complex, optimization, fourier, TensorList
from pytracking.utils.plotting import plot_graph
import math

class FilterOptim(optimization.ConjugateGradientBase):
    def __init__(self, params, reg_energy):
        super(FilterOptim, self).__init__(params.fletcher_reeves, params.standard_alpha, params.direction_forget_factor, (params.debug >= 3))

        # Parameters
        self.params = params

        self.reg_energy = reg_energy
        self.sample_energy = None

        self.residuals = torch.zeros(0)


    def register(self, filter, training_samples, yf, sample_weights, reg_filter):
        self.filter = filter
        self.training_samples = training_samples    # (h, w, num_samples, num_channels, 2)
        self.yf = yf
        self.sample_weights = sample_weights
        self.reg_filter = reg_filter


    def run(self, num_iter, new_xf: TensorList = None):
        if num_iter == 0:
            return

        if new_xf is not None:
            new_sample_energy = complex.abs_sqr(new_xf)
            if self.sample_energy is None:
                self.sample_energy = new_sample_energy
            else:
                self.sample_energy = (1 - self.params.precond_learning_rate) * self.sample_energy + self.params.precond_learning_rate * new_sample_energy

        # Compute right hand side
        self.b = complex.mtimes(self.sample_weights.view(1,1,1,-1), self.training_samples).permute(2,3,0,1,4)
        self.b = complex.mult_conj(self.yf, self.b)

        self.diag_M = (1 - self.params.precond_reg_param) * (self.params.precond_data_param * self.sample_energy +
                            (1 - self.params.precond_data_param) * self.sample_energy.mean(1, keepdim=True)) + self.params.precond_reg_param * self.reg_energy

        _, res = self.run_CG(num_iter, self.filter)

        if self.debug:
            self.residuals = torch.cat((self.residuals, res))
            plot_graph(self.residuals, 9)



    def A(self, hf: TensorList):
        # Classify
        sh = complex.mtimes(self.training_samples, hf.permute(2,3,1,0,4)) # (h, w, num_samp, num_filt, 2)
        sh = complex.mult(self.sample_weights.view(1,1,-1,1), sh)

        # Multiply with transpose
        hf_out = complex.mtimes(sh.permute(0,1,3,2,4), self.training_samples, conj_b=True).permute(2,3,0,1,4)

        # Add regularization
        for hfe, hfe_out, reg_filter in zip(hf, hf_out, self.reg_filter):
            reg_pad1 = min(reg_filter.shape[-2] - 1, hfe.shape[-3] - 1)
            reg_pad2 = min(reg_filter.shape[-1] - 1, 2*hfe.shape[-2]- 2)

            # Add part needed for convolution
            if reg_pad2 > 0:
                hfe_conv = torch.cat([complex.conj(hfe[...,1:reg_pad2+1,:].flip((2,3))), hfe], -2)
            else:
                hfe_conv = hfe.clone()

            # Shift data to batch dimension
            hfe_conv = hfe_conv.permute(0,1,4,2,3).reshape(-1, 1, hfe_conv.shape[-3], hfe_conv.shape[-2])

            # Do first convolution
            hfe_conv = F.conv2d(hfe_conv, reg_filter, padding=(reg_pad1, reg_pad2))

            # Do second convolution
            remove_size = min(reg_pad2, hfe.shape[-2]-1)
            hfe_conv = F.conv2d(hfe_conv[...,remove_size:], reg_filter)

            # Reshape back and add
            hfe_out += hfe_conv.reshape(hfe.shape[0], hfe.shape[1], 2, hfe.shape[2], hfe.shape[3]).permute(0,1,3,4,2)
        return hf_out


    def ip(self, a: torch.Tensor, b: torch.Tensor):
        return fourier.inner_prod_fs(a, b)
        

    def M1(self, hf):
        return complex.div(hf, self.diag_M)
