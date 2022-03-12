import os
import hydra
import pandas as pd
from hydra import utils
from preprocessing.utils import clean_text


def preprocess(in_path, out_path, config):
    data = pd.read_csv(in_path)

    data.loc[:, config.variables.text_vars].fillna("", inplace=True)

    for text_var in config.variables.text_vars:
        data[text_var] = data[text_var].map(clean_text)

    data.to_csv(out_path, index=False)


@hydra.main(config_path="../config/preprocessing.yaml")
def run_preprocessing(config):

    assert len(config.dataset.input_paths) == len(config.dataset.output_paths)

    current_path = utils.get_original_cwd()

    for in_path, out_path in zip(
        config.dataset.input_paths, config.dataset.output_paths
    ):

        full_in_path = os.path.join(current_path, in_path)
        full_out_path = os.path.join(current_path, out_path)

        preprocess(full_in_path, full_out_path, config)


if __name__ == "__main__":
    run_preprocessing()
