import pickle
from train import train
from sbi import utils
from network import SummaryNet
from utils import get_dataset

embedding_net = SummaryNet()

TRAIN_DATA_FILE_PATH = 'path/to/train/data/pickle/file'

prior_low = [0.3, -0.8, -0.8, -2, -2, -0.8, -0.8, 18, 0.1, 0.5, -0.8, -0.8]
prior_high = [4.0, 0.8, 0.8, 2, 2, 0.8, 0.8, 25, 3, 8, 0.8, 0.8]

prior = utils.BoxUniform(low=torch.tensor(prior_low), high=torch.tensor(prior_high), device='cpu')

train_data = get_dataset(TRAIN_DATA_FILE_PATH)

posterior = train(prior, train_data, embedding_net, model='maf', hidden_features=10, num_transforms=2)


def save_posterior(posterior):
    file_name = 'exp1_posterior.pkl'
    with open(file_name) as open_file:
        pickle.dump(posterior, open_file)
    

def load_posterior(posterior_file):
    with open(posterior_file, 'rb') as f:
        posterior = pickle.load(f)

    return posterior


