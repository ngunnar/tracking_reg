from pytracking.utils import TrackerParams, FeatureParams
from pytracking.features import color
from pytracking.features.extractor import SingleResolutionExtractor, MultiResolutionExtractor
import torch

def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = True
    params.visdom_info = {'use_visdom':False}

    params.use_gpu = True

    # Feature specific parameters
    shallow_params = TrackerParams()

    # Conjugate Gradient parameters
    params.CG_iter = 5                  # The number of Conjugate Gradient iterations in each update after the first frame
    params.init_CG_iter = 50           # The total number of Conjugate Gradient iterations used in the first frame
    params.init_GN_iter = 10            # The number of Gauss-Newton iterations used in the first frame (only if the projection matrix is updated)
    params.post_init_CG_iter = 0        # CG iterations to run after GN
    params.fletcher_reeves = False      # Use the Fletcher-Reeves (true) or Polak-Ribiere (false) formula in the Conjugate Gradient
    params.standard_alpha = True        # Use the standard formula for computing the step length in Conjugate Gradient
    params.CG_forgetting_rate = 75	 	# Forgetting rate of the last conjugate direction
    params.precond_data_param = 0.3	 	# Weight of the data term in the preconditioner
    params.precond_reg_param = 0.15	 	# Weight of the regularization term in the preconditioner
    params.precond_proj_param = 35	 	# Weight of the projection matrix part in the preconditioner

    # Learning parameters
    shallow_params.learning_rate = 0.025
    shallow_params.output_sigma_factor = 1/4#1/8 #1/16#1.0#1/16

    # Training parameters
    params.sample_memory_size = 200     # Memory size
    params.train_skipping = 201          # How often to run training (every n-th frame)

    # Detection parameters
    params.score_upsample_factor = 1                             # How much Fourier upsampling to use
    params.score_fusion_strategy = 'weightedsum'                 # Fusion strategy
    shallow_params.translation_weight = 1.0                      # Weight of this feature

    # Init augmentation parameters
    params.augmentation = {'fliplr': False,
                           #'rotate': [5, -5, 10, -10, 20, -20, 30, -30, 45,-45, -60, 60],
                           'blur': [(2, 0.2), (0.2, 2), (3,1), (1, 3), (2, 2)],
                           #'shift': [(6, 6), (-6, 6), (6, -6), (-6,-6)],
                           'dropout': (7, 0.2)
                           }

    # Whether to use augmentation for this feature
    #deep_params.use_augmentation = True
    shallow_params.use_augmentation = True

    # Interpolation parameters
    params.interpolation_method = 'bicubic'    # The kind of interpolation kernel
    params.interpolation_bicubic_a = -0.75     # The parameter for the bicubic interpolation kernel
    params.interpolation_centering = True      # Center the kernel at the feature sample
    params.interpolation_windowing = False     # Do additional windowing on the Fourier coefficients of the kernel

    # Regularization parameters
    shallow_params.use_reg_window = True           # Use spatial regularization or not
    shallow_params.reg_window_min = 1e-4		   # The minimum value of the regularization window
    shallow_params.reg_window_edge = 10e-3         # The impact of the spatial regularization
    shallow_params.reg_window_power = 2            # The degree of the polynomial to use (e.g. 2 is a quadratic window)
    shallow_params.reg_sparsity_threshold = 0.05   # A relative threshold of which DFT coefficients that should be set to zero

    fparams = FeatureParams(feature_params=[shallow_params])
    features = color.RGB(fparams = fparams, 
            pool_stride = None, 
            output_size = None, 
            normalize_power = None, 
            use_for_color = True,
            use_for_gray = False)

    params.features = MultiResolutionExtractor([features])

    return params