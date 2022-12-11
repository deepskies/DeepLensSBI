import torch
import numpy as np
import deeplenstronomy.deeplenstronomy as dl
from deeplenstronomy.visualize import view_image
from utils import get_dataset
from sbi.analysis import pairplot
import matplotlib.pyplot as plt

test_file_path = 'path/to/test/pickle/file'

dataset_test = get_dataset(test_file_path, train=False)
test_metadata = dataset_test.CONFIGURATION_1_metadata


def get_cornerplot(posterior, sample_num=10):

    # Visualize the sample_num lens (use only in jupyter)
    # view_image(dataset_test.CONFIGURATION_1_images[sample_num,0])

    x_test = torch.tensor(dataset_test.CONFIGURATION_1_images[sample_num,0].reshape(1,1024))
    x_test = torch.tensor(x_test, dtype=torch.float32)
    samples = posterior.set_default_x(x_test).sample((10000,))
    prior_low = [0.5, -0.6, -0.6, -1, -1, -0.6, -0.6, 19, 0.1, 0.5, -0.6, -0.6]
    prior_high = [3.0, 0.6, 0.6, 1, 1, 0.6, 0.6, 24, 2, 6, 0.6, 0.6]

    limits = np.array([prior_low, prior_high]).T

    # A bit cumbersome, but we need to make a vector of the true parameter values.
    theta_E = test_metadata['PLANE_1-OBJECT_1-MASS_PROFILE_1-theta_E-g']
    le1 = test_metadata['PLANE_1-OBJECT_1-MASS_PROFILE_1-e1-g']
    le2 = test_metadata['PLANE_1-OBJECT_1-MASS_PROFILE_1-e2-g']
    x = test_metadata['PLANE_1-OBJECT_1-MASS_PROFILE_1-center_x-g']
    y = test_metadata['PLANE_1-OBJECT_1-MASS_PROFILE_1-center_y-g']

    g1 = test_metadata['PLANE_1-OBJECT_1-SHEAR_PROFILE_1-gamma1-g']
    g2 = test_metadata['PLANE_1-OBJECT_1-SHEAR_PROFILE_1-gamma2-g']

    smag = test_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-magnitude-g']
    R = test_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-R_sersic-g']
    n = test_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-n_sersic-g']
    se1 = test_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-e1-g']
    se2 = test_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-e2-g']

    true_parameter = torch.tensor([theta_E[sample_num],le1[sample_num],le2[sample_num],x[sample_num],y[sample_num],g1[sample_num],g2[sample_num],smag[sample_num],R[sample_num],n[sample_num],se1[sample_num],se2[sample_num]])


    # Now let's make a vector of the median (one definition of "best-fit") values of the posterior for each parameter
    best_fit = [np.median(samples[:,param].cpu()) for param in range(12)]
    best_fit_t = torch.tensor(best_fit)

    # Let's plot a corner plot.
    fig, ax = pairplot(samples.cpu(),
                       points=[true_parameter,best_fit_t],
                       labels=[r'$\theta_E$',r'$le1$',r'$1e2$',r'x',r'y',r'g1',r'g2',r'smag',r'R',r'n',r'se1','se2'],
                       limits=limits,
                       points_colors=['r','b'],
                       points_offdiag={'markersize': 6},
                       fig_size=[12, 12])

    plt.show()