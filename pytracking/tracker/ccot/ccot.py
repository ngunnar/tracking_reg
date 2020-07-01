from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import math
from pytracking import complex, dcf, fourier, TensorList
from pytracking.libs.tensorlist import tensor_operation
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor
from .optim import FilterOptim
from pytracking.features import augmentation

class CCOT(BaseTracker):

    def initialize_features(self, im):
        if not getattr(self, 'features_initialized', False):
            self.params.features.initialize(im)
        self.features_initialized = True


    def initialize(self, image, info: dict, gpu_device) -> dict:
        # Initialize some stuff
        self.frame_num = 1
        self.params.device = 'cuda:{0}'.format(gpu_device) if self.params.use_gpu else 'cpu'

        # Convert image
        im = numpy_to_torch(image)
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])

        # Initialize features
        self.initialize_features(im)

        # Chack if image is color
        self.params.features.set_is_color(image.shape[2] == 3)

        # Get feature specific params
        self.fparams = self.params.features.get_fparams('feature_params')

        # Get position and size
        self.points = TensorList([torch.Tensor([p[0], p[1]]) for p in info['points']])
        self.org_points = self.points.clone()
        self.target_sz = torch.Tensor([info['target_sz'][0], info['target_sz'][1]])
        
        # Use odd square search area and set sizes
        feat_max_stride = max(self.params.features.stride())
        self.img_sample_sz = self.image_sz.clone()
        self.img_sample_sz += feat_max_stride - self.img_sample_sz % (2 * feat_max_stride)

        # Set other sizes (corresponds to ECO code)
        self.img_support_sz = self.img_sample_sz
        self.mid_point = self.img_support_sz // 2
        self.feature_sz = self.params.features.size(self.img_sample_sz)
        self.filter_sz = self.feature_sz + (self.feature_sz + 1) % 2
        self.output_sz = self.img_support_sz    # Interpolated size of the output

        # Number of filters
        self.num_filters = len(self.filter_sz)

        # Get window function
        #self.window = TensorList([dcf.hann2d(sz).to(self.params.device) for sz in self.feature_sz])
        self.window = TensorList([torch.ones((1,1,int(sz[0].item()),int(sz[1].item()))).to(self.params.device) for sz in self.feature_sz])
        #self.window = TensorList([dcf.tukey2d(sz).to(self.params.device) for sz in self.feature_sz])

        # Get interpolation function
        self.interp_fs = TensorList([dcf.get_interp_fourier(sz, self.params.interpolation_method,
                                     self.params.interpolation_bicubic_a, self.params.interpolation_centering,
                                     self.params.interpolation_windowing, self.params.device) for sz in self.filter_sz])

        # Get label function
        output_sigma_factor = self.fparams.attribute('output_sigma_factor')
        sigma = (self.filter_sz / self.img_support_sz) * torch.sqrt(self.target_sz.prod()) * output_sigma_factor
        yf_zero = TensorList([dcf.label_function(sz, sig).to(self.params.device) for sz, sig in zip(self.filter_sz, sigma)])
        yf_zero = complex.complex(yf_zero)
        self.yf = TensorList()
        for p in self.points:
            shift_sample = 2*math.pi*(self.mid_point-p) / self.img_support_sz
            self.yf.append(TensorList([fourier.shift_fs(yfs, shift_sample) for yfs in yf_zero]))
        
        # Optimization options
        self.params.precond_learning_rate = self.fparams.attribute('learning_rate')
        if self.params.CG_forgetting_rate is None or max(self.params.precond_learning_rate) >= 1:
            self.params.direction_forget_factor = 0
        else:
            self.params.direction_forget_factor = (1 - max(self.params.precond_learning_rate))**self.params.CG_forgetting_rate

        # Extract and transform sample
        x = self.generate_init_samples(im).to(self.params.device)
        self.x = x
        # Transform to get the training sample
        train_xf = self.preprocess_sample(x)

        # Shift the samples back
        if 'shift' in self.params.augmentation:
            for xf in train_xf:
                if xf.shape[0] == 1:
                    continue
                for i, shift in enumerate(self.params.augmentation['shift']):
                    shift_samp = 2 * math.pi * torch.Tensor(shift) / self.img_support_sz
                    xf[1+i:2+i,...] = fourier.shift_fs(xf[1+i:2+i,...], shift=shift_samp)

        # Initialize first-frame training samples
        num_init_samples = train_xf.size(0)
                
        self.init_training_samples = train_xf.permute(2, 3, 0, 1, 4)
        
        # Initialize memory
        # Initialize filter
        self.training_samples = TensorList(
            [xf.new_zeros(xf.shape[2], xf.shape[3], self.params.sample_memory_size, xf.shape[1], 2) for xf in train_xf])
        self.filters = TensorList([
            TensorList([xf.new_zeros(1, xf.shape[1], xf.shape[2], xf.shape[3], 2) for xf in train_xf])
             for i in range(len(self.points))])
        
        self.init_sample_weights = TensorList([xf.new_ones(1) / xf.shape[0] for xf in train_xf])
        self.sample_weights = TensorList([xf.new_zeros(self.params.sample_memory_size) for xf in train_xf]) 
        for sw, init_sw, num in zip(self.sample_weights, self.init_sample_weights, num_init_samples):
            sw[:num] = init_sw
    
        # Get regularization filter
        self.reg_filter = TensorList([dcf.get_reg_filter(self.img_support_sz, self.target_sz, fparams).to(self.params.device) for fparams in self.fparams])
        self.reg_energy = self.reg_filter.view(-1) @ self.reg_filter.view(-1)
        
        # Sample counters and weights
        self.num_stored_samples = num_init_samples
        self.previous_replace_ind = [None]*len(self.num_stored_samples)
        
        for train_samp, init_samp in zip(self.training_samples, self.init_training_samples):
            train_samp[:,:,:init_samp.shape[2],:,:] = init_samp
        
        sample_energy = complex.abs_sqr(self.training_samples).mean(dim=2, keepdim=True).permute(2, 3, 0, 1)
        # Do joint optimization
        for i in range(len(self.points)):
            print('{0}'.format(i), end=', ')        
            ts = self.training_samples.clone()
            yf = self.yf[i]
            filters = self.filters[i]            
            i_sw = self.init_sample_weights.clone()
            re = self.reg_energy.clone()
            sw = self.sample_weights.clone()
            rf = self.reg_filter.clone()            
            filter_optimizer = FilterOptim(self.params, re)
            filter_optimizer.register(filters, ts, yf, sw, rf)
            filter_optimizer.sample_energy = sample_energy.clone()
            
            filter_optimizer.run(self.params.init_CG_iter)
            
            # Post optimization
            filter_optimizer.run(self.params.post_init_CG_iter)
            self.filters[i] = filter_optimizer.filter
        self.symmetrize_filter()
        print()


    def track(self, image, update = False) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num
        # Convert image
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #

        # Get sample
        test_xf = self.extract_fourier_sample(im)

        # Compute scores
        sfs = self.apply_filters(test_xf)
        out = TensorList([self.localize_and_update_target(sfs[i], i) for i in range(len(self.points))])
        
        return out

    def apply_filters(self, sample_xf: TensorList) -> torch.Tensor:
        return TensorList([complex.mult(f, sample_xf).sum(1, keepdim=True) for f in self.filters])

    def apply_filter(self, sample_xf: TensorList) -> torch.Tensor:
        return complex.mult(self.filter, sample_xf).sum(1, keepdim=True)

    def localize_and_update_target(self, sf: TensorList, i):
        if self.params.score_fusion_strategy == 'weightedsum':
            weight = self.fparams.attribute('translation_weight')
            sf = fourier.sum_fs(weight * sf)
            scores = fourier.sample_fs(sf, self.output_sz)
        else:
            raise ValueError('Unknown score fusion strategy.')

        # Get maximum
        max_score, max_disp = dcf.max2d(scores)
        max_disp = max_disp.float().cpu()

        # Convert to displacements in the base scale
        if self.params.score_fusion_strategy in ['sum', 'weightedsum']:
            disp = (max_disp + self.output_sz / 2) % self.output_sz - self.output_sz / 2
        elif self.params.score_fusion_strategy == 'transcale':
            disp = max_disp - self.output_sz / 2

        # Compute translation vector and scale change factor
        translation_vec = disp.view(-1) * (self.img_support_sz / self.output_sz)
        
        # Update pos
        new_pos = self.mid_point.round() + translation_vec
        
        inside_ratio = 0.2
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.points[i] = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)
        
        return self.points[i].round(), max_score, scores

    def extract_fourier_sample(self, im: torch.Tensor) -> TensorList:
        x = F.interpolate(im, self.output_sz.long().tolist(), mode='bilinear')
        x = TensorList([f.get_feature(x) for f in self.params.features.features]).unroll().to(self.params.device)
        return self.preprocess_sample(x)

    def preprocess_sample(self, x: TensorList) -> TensorList:
        x *= self.window
        sample_xf = fourier.cfft2(x)
        return TensorList([dcf.interpolate_dft(xf, bf) for xf, bf in zip(sample_xf, self.interp_fs)])

    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        # Do data augmentation
        transforms = [augmentation.Identity()]
        if 'shift' in self.params.augmentation:
            transforms.extend([augmentation.Translation(shift) for shift in self.params.augmentation['shift']])
        if 'fliplr' in self.params.augmentation and self.params.augmentation['fliplr']:
            transforms.append(augmentation.FlipHorizontal())
        if 'rotate' in self.params.augmentation:
            transforms.extend([augmentation.Rotate(angle) for angle in self.params.augmentation['rotate']])
        if 'blur' in self.params.augmentation:
            transforms.extend([augmentation.Blur(sigma) for sigma in self.params.augmentation['blur']])

        im_patch = F.interpolate(im, self.output_sz.long().tolist(), mode='bilinear')
        im_patches = torch.cat([T(im_patch) for T in transforms])
        init_samples = TensorList([f.get_feature(im_patches) for f in self.params.features.features]).unroll()

        # Remove augmented samples for those that shall not have
        for i, use_aug in enumerate(self.fparams.attribute('use_augmentation')):
            if not use_aug:
                init_samples[i] = init_samples[i][0:1, ...]

        if 'dropout' in self.params.augmentation:
            num, prob = self.params.augmentation['dropout']
            for i, use_aug in enumerate(self.fparams.attribute('use_augmentation')):
                if use_aug:
                    init_samples[i] = torch.cat([init_samples[i], F.dropout2d(init_samples[i][0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        return init_samples
    
    def symmetrize_filter(self):
        for f in self.filters:
            for hf in f:
                hf[:,:,:,0,:] /= 2
                hf[:,:,:,0,:] += complex.conj(hf[:,:,:,0,:].flip((2,)))