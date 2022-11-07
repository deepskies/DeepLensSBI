import torch
from sbi import utils, inference
from sbi.inference import SNPE, SNLE, SNRE
from network import SummaryNet

embedding_net = SummaryNet()

prior_low = [1.0, 0.00, 0.00, 19.9, -0.1, -0.1, 1.0, 3, 0.0, 0.0]
prior_high = [4.0, 0.05, 0.05, 20.1, 0.1, 0.1, 2.0, 5, 0.05, 0.05]

prior = utils.BoxUniform(low=torch.tensor(prior_low), 
                             high=torch.tensor(prior_high))                           

# instantiate the neural density estimator
neural_posterior = utils.posterior_nn(model='maf', 
                                      embedding_net=embedding_net,
                                      hidden_features=10,
                                      num_transforms=2)

# setup the inference procedure with the SNPE-C procedure
inference = SNPE(prior=prior, density_estimator=neural_posterior)

# Reshape the images so each image is a vector. Note: The first number should be the same as the size of the dataset in training_sim.yaml 
images = torch.tensor(dataset.CONFIGURATION_1_images.reshape(1000,1,1024),dtype=torch.float32)[:,0]

# Now let's choose the parameters we are interested in training the SBI network on. 
# You can see the complete list of available parameters in deeplenstronomy by printing dataset.CONFIGURATION_1_metadata.keys(). 
# In many cases we are only interested in a subset of all the available parameters

# These 3 are lens mass parameters of a SIE: Einstein radius and ellipticity. positions fixed to (0,0)
theta_E = dataset.CONFIGURATION_1_metadata['PLANE_1-OBJECT_1-MASS_PROFILE_1-theta_E-g']
le1 = dataset.CONFIGURATION_1_metadata['PLANE_1-OBJECT_1-MASS_PROFILE_1-e1-g']
le2 = dataset.CONFIGURATION_1_metadata['PLANE_1-OBJECT_1-MASS_PROFILE_1-e2-g']

# These are source light parameters - magnitude and parameters of a Sersic profile. 
smag = dataset.CONFIGURATION_1_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-magnitude-g']
x = dataset.CONFIGURATION_1_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-center_x-g']
y = dataset.CONFIGURATION_1_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-center_y-g']
R = dataset.CONFIGURATION_1_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-R_sersic-g']
n = dataset.CONFIGURATION_1_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-n_sersic-g']
se1 = dataset.CONFIGURATION_1_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-e1-g']
se2 = dataset.CONFIGURATION_1_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-e2-g']

# Now let's put them in a tensor form that SBI can read.
theta = torch.tensor(np.array([theta_E,le1,le2,smag,x,y,R,n,se1,se2]).T,dtype=torch.float32)

# Now that we have both the simulated images and parameters defined properly, we can train the SBI. 
density_estimator = inference.append_simulations(theta,images).train()
posterior = inference.build_posterior(density_estimator)
