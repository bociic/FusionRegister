from data.dataset_RGB import *


def get_training_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderFileTrain(rgb_dir)

def get_training_1(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderFile1(rgb_dir)

def get_validation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    # return DataLoaderFileVal(rgb_dir)
    return DataLoaderFileVal(rgb_dir)
def get_test_data(input_dir):

    return DataLoaderFileTest( input_dir)
