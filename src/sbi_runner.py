# Requires nsbi conda package, can be loaded with condapack
# https://conda.github.io/conda-pack/ (on the target machine section)

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from sbi import utils
from sbi.inference import SNPE
import numpy as np
import pickle

from network import SummaryNet

# set seed for numpy and torch

def set_seed(seed_val):
    """
    Set seed for numpy and torch
    Parameters
    ----------
    seed_val : int
        seed value to set
    
    Returns
    -------
    None
    """
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    np.random.seed(seed_val)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.benchmark = False

#TODO: change the train file names before code release

def get_data(num_params):
    """
    Get the training and test data for the given number of parameters
    Parameters
    ----------
    num_params : int
        Number of parameters to use in the simulation
    
    Returns
    -------
    dataset_train : torch.Tensor
        Training data
    Raises
    ------
    ValueError
        If the number of parameters is invalid
    FileNotFoundError
        If the training or test data is not found
    """
    if num_params == 1:
        train_data_path = "SBI_dataset/1param_200k_train.pkl"
    elif num_params == 5:
        train_data_path = 'SBI_dataset/5param_model_training_500k_Aug29.pkl'
    elif num_params == 12:
        train_data_path = 'SBI_dataset/12_model_training_des_1M.pkl'
    else:
        raise ValueError("Invalid number of parameters")

    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Training data not found at {train_data_path}")
    
    with open(train_data_path, "rb") as train_data_file:
        dataset_train = pickle.load(train_data_file)
    
    return dataset_train


def get_model(out_features):
    """
    Get the model for the given number of parameters
    Parameters
    ----------
    out_features : int
        Number of out_features to use in the simulation
    
    Returns
    -------
    embedding_net : torch.nn.Module
        Model for the given number of parameters
    """
    embedding_net = SummaryNet(out_features).to('cuda:0')
    return embedding_net


def get_priors(num_params):
    """
    Get the priors for the given number of parameters
    Parameters
    ----------
    num_params : int
        Number of parameters to infer
    Returns
    -------
    prior_low : list
        Lower bounds for the priors
    prior_high : list
        Upper bounds for the priors
    """
    if num_params == 1:
        prior_low = [0.3]
        prior_high = [4.0]
    elif num_params == 5:
        prior_low = [0.3, -0.8, -0.8, -2, -2]
        prior_high = [4.0, 0.8, 0.8, 2, 2]
    elif num_params == 12:
        prior_low = [0.3, -0.8, -0.8, -2, -2, -0.8, -0.8, 18, 0.1, 0.5, -0.8, -0.8]
        prior_high = [4.0, 0.8, 0.8, 2, 2, 0.8, 0.8, 25, 3, 8, 0.8, 0.8]
    else:
        raise ValueError("Invalid number of parameters")
    return prior_low, prior_high


def get_parameters(num_params, dataset_train):
    train_metadata = dataset_train.CONFIGURATION_1_metadata

    common_params = ['theta_E-g', 'e1-g', 'e2-g', 'center_x-g', 'center_y-g']
    if num_params == 1:
        params = [train_metadata[f'PLANE_1-OBJECT_1-MASS_PROFILE_1-{common_params[0]}']]
    elif num_params == 5:
        params = [train_metadata[f'PLANE_1-OBJECT_1-MASS_PROFILE_1-{param}'] for param in common_params]
    elif num_params == 12:
        shear_params = [train_metadata['PLANE_1-OBJECT_1-SHEAR_PROFILE_1-gamma1-g'], train_metadata['PLANE_1-OBJECT_1-SHEAR_PROFILE_1-gamma2-g']]

        source_light_params = [train_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-magnitude-g'], train_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-R_sersic-g'], train_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-n_sersic-g'], train_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-e1-g'], train_metadata['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-e2-g']]

        params = [train_metadata[f'PLANE_1-OBJECT_1-MASS_PROFILE_1-{param}'] for param in common_params]
        params.extend(shear_params)
        params.extend(source_light_params)
    else:
        raise ValueError("Invalid number of parameters requested")

    return params


def trainer(num_params, embedding_net, hidden_features, num_transforms, dataset_train):
    """
    Train the model
    Parameters
    ----------
    num_params : int
        Number of parameters to infer
    embedding_net : torch.nn.Module
        Model for the given number of parameters
    hidden_features : int
        Number of hidden features to use in the Normalizing Flows
    num_transforms : int
        Number of transforms to use in the Normalizing Flows
    dataset_train : torch.Tensor
        Training data
    Returns
    -------
    posterior : Posterior distribution 
    """
    prior_low, prior_high = get_priors(num_params)

    prior = utils.BoxUniform(low=torch.tensor(prior_low),
                                high=torch.tensor(prior_high), device="cuda:0")                          

    # instantiate the neural density estimator
    neural_posterior = utils.posterior_nn(model='maf',
                                         embedding_net=embedding_net,
                                         hidden_features=hidden_features,
                                         num_transforms=num_transforms)

    # setup the inference procedure with the SNPE-C procedure
    inference = SNPE(prior=prior, density_estimator=neural_posterior, device="cuda:0")

    parameters = get_parameters(num_params, dataset_train)

    num_images = dataset_train.CONFIGURATION_1_images.shape[0]
    images = torch.cuda.FloatTensor(dataset_train.CONFIGURATION_1_images.reshape(num_images,1,1024))[:,0]
    theta = torch.cuda.FloatTensor(np.array(parameters).T)
    density_estimator = inference.append_simulations(theta,images).train(show_train_summary=True)#(stop_after_epochs=5)
    posterior = inference.build_posterior(density_estimator)
    plt.plot(inference.summary['train_log_probs'], label='training')
    plt.plot(inference.summary['validation_log_probs'], label='validation')
    plt.ylabel('log_probs', fontsize=18)
    plt.xlabel('epochs', fontsize=18)
    plt.title(f'Experiment: {num_params} param, {hidden_features} HF, {num_transforms} NT, {out_features} out_features, seed:{seed}')
    plt.legend()
    plt.show()
    return posterior


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SBI training.")
    parser.add_argument("--num_params", type=int)
    parser.add_argument("--hidden_features", type=int)
    parser.add_argument("--num_transforms", type=int)
    parser.add_argument("--out_features", type=int)
    parser.add_argument("--seed", type=int)
    
    args = parser.parse_args()

    num_params = args.num_params
    hidden_features = args.hidden_features
    num_transforms = args.num_transforms
    out_features = args.out_features
    seed = args.seed

    set_seed(seed)
    dataset_train = get_data(num_params)

    # out_features = num_params * 4
    embedding_net = get_model(out_features)
    posterior = trainer(num_params, embedding_net, hidden_features, num_transforms, dataset_train)

    file_name = f"{num_params}param_hf{hidden_features}_nt{num_transforms}_of{out_features}_seed{seed}.pkl"
    with open(file_name, "wb") as open_file:
        pickle.dump(posterior, open_file)
