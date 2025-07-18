import os
from util import info
from cmdstanpy import CmdStanModel


def save_model_file(folder: str, model_name: str, model_content: str):
    file_name = model_name + '.stan'
    save_obj = os.path.join(folder, file_name)
    # check if save_obj exists
    if os.path.exists(save_obj):
        info('util/__init__.py', f"File {save_obj} already exists. Overwriting it.")
    with open(save_obj, 'w') as f:
        f.write(model_content)
    return save_obj


def fit_model(model_path: str, data_json_path: str):
    model = CmdStanModel(stan_file=model_path)
    fit = model.sample(data=data_json_path)

