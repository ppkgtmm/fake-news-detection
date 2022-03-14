import hydra
import logging
from modeling import do_modeling

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="modeling.yaml")
def run_modeling(config):

    assert len(config.dataset.input_paths) == len(config.dataset.headers)
    assert len(config.dataset.headers) == len(config.dataset.target_vals)

    log.info("Config param validation successful")

    log.info("Begin modeling process")

    do_modeling(config)

    log.info("End modeling process")


if __name__ == "__main__":
    run_modeling()
