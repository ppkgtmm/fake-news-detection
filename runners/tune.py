import hydra
import logging
from modeling import do_tuning

log = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="modeling.yaml")
def run_tuning(config):

    assert len(config.dataset.input_paths) == len(config.dataset.headers)
    assert len(config.dataset.headers) == len(config.variables.target_vals)

    log.info("Config param validation successful")

    log.info("Begin parameter tuning process")

    do_tuning(config)

    log.info("End parameter tuning process")


if __name__ == "__main__":
    run_tuning()
