import os
import hydra
from hydra import utils
from preprocessing import preprocess


@hydra.main(config_path="config", config_name="preprocessing.yaml")
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
