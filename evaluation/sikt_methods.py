import SimpleITK as sitk


def bspline_intra_modal_registration(fixed_image, moving_image, spacing, metric,fixed_image_mask=None):
    R = sitk.ImageRegistrationMethod()    
    # Determine the number of BSpline control points using the physical spacing we want for the control grid. 
    grid_physical_spacing = [spacing, spacing] # A control point every spacing mm
    image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]    
    mesh_size = [int(image_size/grid_spacing + 0.5) \
                 for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]
    initial_transform = sitk.BSplineTransformInitializer(image1 = fixed_image, transformDomainMeshSize = mesh_size, order=3)    
    R.SetInitialTransform(initial_transform)
    
    # Get user-specified metric
    if (metric == "mi"):
        R.SetMetricAsMattesMutualInformation()
    elif (metric == "mse"):
        R.SetMetricAsMeanSquares()
    elif (metric == "corr"):
        R.SetMetricAsCorrelation()
    else:
        raise Exception("No metric specified")
    
    # Settings for metric sampling, usage of a mask is optional. When given a mask the sample points will be 
    # generated inside that region. Also, this implicitly speeds things up as the mask is smaller than the
    # whole image.
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.01)
    if fixed_image_mask:
        registration_method.SetMetricFixedMask(fixed_image_mask)
        
    # Multi-resolution framework.
    # Pyramid
    R.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    # Smooth Regularization
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # Interpolation to next level
    R.SetInterpolator(sitk.sitkLinear)
    
    # Optimizer
    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100)    
    return R.Execute(fixed_image, moving_image)

def dvf_registration(fixed_image, moving_image, intDiffThreshold, metric):
    R = sitk.ImageRegistrationMethod()
    # Get user-specified metric
    if (metric == "mi"):
        R.SetMetricAsMattesMutualInformation()
    elif (metric == "mse"):
        R.SetMetricAsMeanSquares()
    elif (metric == 'demon'):
        R.SetMetricAsDemons(intDiffThreshold)
    elif (metric == "corr"):
        R.SetMetricAsCorrelation()
    else:
        raise Exception("No metric specified")
    
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.25)
    R.SetOptimizerScalesFromPhysicalShift()

    # Displacement field transform
    toDisplacementFilter = sitk.TransformToDisplacementFieldFilter()
    toDisplacementFilter.SetReferenceImage(fixed_image)
    DVF = sitk.DisplacementFieldTransform(
        toDisplacementFilter.Execute(sitk.Transform(moving_image.GetDimension(),sitk.sitkIdentity)))
    DVF.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0,varianceForTotalField=1)
    
    R.SetInitialTransform(DVF)
    #R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )

    # Multi-resolution setup
    # Pyramid
    R.SetShrinkFactorsPerLevel(shrinkFactors=[4,2,1])
    # Smooth
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # Interpolation to next level
    R.SetInterpolator(sitk.sitkLinear)
    # Optimizer
    R.SetOptimizerAsGradientDescent(learningRate=10,numberOfIterations=200,convergenceWindowSize=15)    
    return R.Execute(fixed_image,moving_image)