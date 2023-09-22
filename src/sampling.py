import torch
import sbi
import deeplenstronomy.deeplenstronomy as dl

config_file = 'single.yaml'

def sampling(posterior):
    observed = dl.make_dataset(config_file)
    # Visualize that lens
    #view_image(observed.CONFIGURATION_1_images[0,0])
    x_observed = torch.tensor(observed.CONFIGURATION_1_images[0,0].reshape(1,1024))
    x_observed = torch.tensor(x_observed, dtype=torch.float32)

    samples = posterior[0].set_default_x(x_observed).sample((10000,))
    return samples