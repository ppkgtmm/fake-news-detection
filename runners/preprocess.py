import os
import hydra
import logging
from hydra import utils
from preprocessing import preprocess

log = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="preprocessing.yaml")
def run_preprocessing(config):

    assert len(config.dataset.input_paths) == len(config.dataset.output_paths)

    log.info("Config param validation successful")

    current_path = utils.get_original_cwd()

    log.info("Begin text data preprocessing")

    for in_path, out_path in zip(
        config.dataset.input_paths, config.dataset.output_paths
    ):

        full_in_path = os.path.join(current_path, in_path)
        full_out_path = os.path.join(current_path, out_path)

        preprocess(full_in_path, full_out_path, config)
        log.info(
            "Done preprocessing file {}, output saved to {}".format(in_path, out_path)
        )

    log.info("End text data preprocessing")


if __name__ == "__main__":
    run_preprocessing()
