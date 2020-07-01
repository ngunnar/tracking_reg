from pytracking.utils import TrackerParams, FeatureParams
from pytracking.features.extractor import MultiResolutionExtractor
from pytracking.features import deep
import torch

def parameters(gpu_device = 0):
    params = TrackerParams()

    params.debug = 0
    params.visualization = True
    params.visdom_info = {'use_visdom':False}

    params.use_gpu = True

    # Feature specific parameters
    shallow_params = TrackerParams()
    deep_params = TrackerParams()

    # Conjugate Gradient parameters
    params.CG_iter = 5                  # The number of Conjugate Gradient iterations in each update after the first frame
    params.init_CG_iter = 100           # The total number of Conjugate Gradient iterations used in the first frame
    params.post_init_CG_iter = 0        # CG iterations to run after GN
    params.fletcher_reeves = True      # Use the Fletcher-Reeves (true) or Polak-Ribiere (false) formula in the Conjugate Gradient
    params.standard_alpha = True        # Use the standard formula for computing the step length in Conjugate Gradient
    params.CG_forgetting_rate = 75	 	# Forgetting rate of the last conjugate direction
    params.precond_data_param = 0.7	 	# Weight of the data term in the preconditioner
    params.precond_reg_param = 0.02	 	# Weight of the regularization term in the preconditioner

    # Learning parameters
    shallow_params.learning_rate = 0.025
    deep_params.learning_rate = 0.0075
    shallow_params.output_sigma_factor = 1/12
    deep_params.output_sigma_factor = 1/4

    # Training parameters
    params.sample_memory_size = 200     # Memory size

    # Detection parameters
    params.score_fusion_strategy = 'weightedsum'                 # Fusion strategy
    shallow_params.translation_weight = 0.4                      # Weight of this feature
    deep_params.translation_weight = 1 - shallow_params.translation_weight

    # Init augmentation parameters
    params.augmentation = {'fliplr': False,
                           'blur': [(2, 0.2), (0.2, 2), (3,1), (1, 3), (2, 2)],
                           'dropout': (7, 0.2)}

    # Whether to use augmentation for this feature
    deep_params.use_augmentation = True
    shallow_params.use_augmentation = True

    # Interpolation parameters
    params.interpolation_method = 'bicubic'    # The kind of interpolation kernel
    params.interpolation_bicubic_a = -0.75     # The parameter for the bicubic interpolation kernel
    params.interpolation_centering = False      # Center the kernel at the feature sample
    params.interpolation_windowing = False     # Do additional windowing on the Fourier coefficients of the kernel

    # Regularization parameters
    shallow_params.use_reg_window = True           # Use spatial regularization or not
    shallow_params.reg_window_min = 1e-4		   # The minimum value of the regularization window
    shallow_params.reg_window_edge = 10e-3         # The impact of the spatial regularization
    shallow_params.reg_window_power = 2            # The degree of the polynomial to use (e.g. 2 is a quadratic window)
    shallow_params.reg_sparsity_threshold = 0.1   # A relative threshold of which DFT coefficients that should be set to zero

    deep_params.use_reg_window = True           # Use spatial regularization or not
    deep_params.reg_window_min = 10e-4			# The minimum value of the regularization window
    deep_params.reg_window_edge = 50e-3         # The impact of the spatial regularization
    deep_params.reg_window_power = 2            # The degree of the polynomial to use (e.g. 2 is a quadratic window)
    deep_params.reg_sparsity_threshold = 0.1    # A relative threshold of which DFT coefficients that should be set to zero


    fparams = FeatureParams(feature_params=[shallow_params, deep_params])
    features = deep.ResNet18m1(output_layers=['vggconv1', 'layer3'], 
                               use_gpu=params.use_gpu, 
                               gpu_device = gpu_device, 
                               fparams=fparams, 
                               pool_stride=[2, 1], 
                               normalize_power=2)
    
    params.features = MultiResolutionExtractor([features])

    return params