import hydra
import logging
from hydra import utils
import pandas as pd
from visualization import get_save_path, visualize, visualize_label_distribution

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="visualization.yaml")
def run_visualization(config):

    assert len(config.dataset.input_paths) == len(config.variables.target_vals)
    assert len(config.variables.target_vals) == len(config.style.target_colors)

    log.info("Config param validation successful")

    current_path = utils.get_original_cwd()
    full_path_to_out_dir = get_save_path(current_path, config.visualizations.output_dir)

    data_parts = []

    log.info("Begin visualization process")

    for idx, in_path in enumerate(config.dataset.input_paths):
        full_in_path = get_save_path(current_path, in_path)

        data = visualize(
            full_in_path,
            config.variables.target_vals[idx],
            full_path_to_out_dir,
            config,
        )
        log.info("Visualization of {} was successful".format(full_in_path))

        data_parts.append(data)

    visualize_label_distribution(
        pd.concat(data_parts, axis=0), full_path_to_out_dir, config
    )

    log.info("End visualization process")


if __name__ == "__main__":
    run_visualization()
