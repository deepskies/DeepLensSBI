import pickle


def get_dataset(data_file_path, train=True):
    """

    :param data_file_path: path to pkl file with train/ test data
    :param train: get test data if False
    :return: specified dataset (train/ test)
    """

    data_file = open(data_file_path, 'rb')
    dataset = pickle.load(data_file)
    data_file.close()

    return dataset
