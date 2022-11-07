import numpy as np
import torch


limits = np.array([prior_low, prior_high]).T

# A bit cumbersome, but we need to make a vector of the true parameter values. 
theta_E = observed.CONFIGURATION_1_metadata['PLANE_1-OBJECT_1-MASS_PROFILE_1-theta_E-g']
le1 = observed.CONFIGURATION_1_metadata['PLANE_1-OBJECT_1-MASS_PROFILE_1-e1-g']
le2 = observed.CONFIGURATION_1_metadata['PLANE_1-OBJECT_1-MASS_PROFILE_1-e2-g']
smag = observed.CONFIGURATION_1_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-magnitude-g']
x = observed.CONFIGURATION_1_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-center_x-g']
y = observed.CONFIGURATION_1_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-center_y-g']
R = observed.CONFIGURATION_1_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-R_sersic-g']
n = observed.CONFIGURATION_1_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-n_sersic-g']
se1 = observed.CONFIGURATION_1_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-e1-g']
se2 = observed.CONFIGURATION_1_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-e2-g']

true_parameter = torch.tensor([theta_E[0], le1[0], le2[0], smag[0], x[0], y[0], R[0], n[0], se1[0], se2[0]])

# Now let's make a vector of the median (one definition of "best-fit") values of the posterior for each parameter
best_fit = []
for i in range(10):
  best_fit.append(np.median(samples[:,i]))
best_fit_t = torch.tensor(best_fit)