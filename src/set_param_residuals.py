####### Residual plotting script

import deeplenstronomy as dl
from .plot_residuals import plot_residuals
import matplotlib.pyplot as plt


def set_param_residuals(observed):
    # This is the observed image from the single.yaml file
    observed_image = observed.CONFIGURATION_1_images[0,0]

    # Update the parameters and regenerate the image. 
    config_file = 'single.yaml'
    best_fit = dl.make_dataset(config_file)
    best_fit.update_param({'PLANE_1-OBJECT_1-MASS_PROFILE_1-theta_E-g': best_fit[0]}, 'CONFIGURATION_1')  
    best_fit.update_param({'PLANE_1-OBJECT_1-MASS_PROFILE_1-e1-g': best_fit[1]}, 'CONFIGURATION_1')
    best_fit.update_param({'PLANE_1-OBJECT_1-MASS_PROFILE_1-e2-g': best_fit[2]}, 'CONFIGURATION_1')
    best_fit.update_param({'PLANE_2-OBJECT_1-LIGHT_PROFILE_1-magnitude-g': best_fit[3]}, 'CONFIGURATION_1')
    best_fit.update_param({'PLANE_2-OBJECT_1-LIGHT_PROFILE_1-center_x-g': best_fit[4]}, 'CONFIGURATION_1')
    best_fit.update_param({'PLANE_2-OBJECT_1-LIGHT_PROFILE_1-center_y-g': best_fit[5]}, 'CONFIGURATION_1')
    best_fit.update_param({'PLANE_2-OBJECT_1-LIGHT_PROFILE_1-R_sersic-g': best_fit[6]}, 'CONFIGURATION_1')
    best_fit.update_param({'PLANE_2-OBJECT_1-LIGHT_PROFILE_1-n_sersic-g': best_fit[7]}, 'CONFIGURATION_1')
    best_fit.update_param({'PLANE_2-OBJECT_1-LIGHT_PROFILE_1-e1-g': best_fit[8]}, 'CONFIGURATION_1')
    best_fit.update_param({'PLANE_2-OBJECT_1-LIGHT_PROFILE_1-e2-g': best_fit[9]}, 'CONFIGURATION_1')
    best_fit.regenerate()
    bestfit_image = best_fit.CONFIGURATION_1_images[0,0]

    fig, ax = plot_residuals(observed_image, bestfit_image)