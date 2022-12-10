import torch
from sbi import utils
from sbi.inference import SNPE, SNLE, SNRE
import numpy as np


def train(prior, train_data, embedding_net, model='maf', hidden_features=10, num_transforms=2):

    # instantiate the neural density estimator
    neural_posterior = utils.posterior_nn(model=model,
                                      embedding_net=embedding_net,
                                      hidden_features=hidden_features,
                                      num_transforms=num_transforms)

    # setup the inference procedure with the SNPE-C procedure
    inference = SNPE(prior=prior, density_estimator=neural_posterior, device="cpu")

    # Reshape the images so each image is a vector.
    # Note: The first number should be the same as the size of the dataset in training_sim.yaml

    train_images = torch.tensor(train_data.CONFIGURATION_1_images.reshape(1000,1,1024),dtype=torch.float32)[:,0]
    train_metadata = train_data.CONFIGURATION_1_metadata


    # Now let's choose the parameters we are interested in training the SBI network on.
    # You can see the complete list of available parameters in deeplenstronomy by printing dataset.CONFIGURATION_1_metadata.keys().
    # In many cases we are only interested in a subset of all the available parameters

    # These 5 are lens mass parameters of a SIE: Einstein radius, ellipticity and position
    theta_E = train_metadata['PLANE_1-OBJECT_1-MASS_PROFILE_1-theta_E-g']
    #  print(theta_E.shape)
    le1 = train_metadata['PLANE_1-OBJECT_1-MASS_PROFILE_1-e1-g']
    le2 = train_metadata['PLANE_1-OBJECT_1-MASS_PROFILE_1-e2-g']
    x = train_metadata['PLANE_1-OBJECT_1-MASS_PROFILE_1-center_x-g']
    y = train_metadata['PLANE_1-OBJECT_1-MASS_PROFILE_1-center_y-g']

    # These are shear parameters
    g1 = train_metadata['PLANE_1-OBJECT_1-SHEAR_PROFILE_1-gamma1-g']
    g2 = train_metadata['PLANE_1-OBJECT_1-SHEAR_PROFILE_1-gamma2-g']

    # These are source light parameters - magnitude and parameters of a Sersic profile.
    smag = train_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-magnitude-g']
    R = train_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-R_sersic-g']
    n = train_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-n_sersic-g']
    se1 = train_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-e1-g']
    se2 = train_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-e2-g']

    # Now let's put them in a tensor form that SBI can read.
    theta = torch.tensor(np.array([theta_E,le1,le2,smag,x,y,R,n,se1,se2]).T,dtype=torch.float32)

    # Now that we have both the simulated images and parameters defined properly, we can train the SBI.
    density_estimator = inference.append_simulations(theta,train_images).train()
    posterior = inference.build_posterior(density_estimator)
    return posterior
