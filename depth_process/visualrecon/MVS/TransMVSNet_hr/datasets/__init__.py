import importlib


def find_dataset_def(dataset_name):
    # module_name = 'datasets.{}'.format(dataset_name)
    module_name = 'MVS.TransMVSNet_hr.datasets.{}'.format(dataset_name)
    module = importlib.import_module(module_name)
    return getattr(module, "MVSDataset")
