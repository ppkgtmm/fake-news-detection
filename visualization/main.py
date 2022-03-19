import logging
import pandas as pd
from visualization.utilities import *

log = logging.getLogger(__name__)


def visualize_text(data, config, target_val, out_dir):
    text_vars = config.variables.text_vars
    bins = config.visualizations.hist_bins
    for text_var in text_vars:
        plot_word_cloud(
            data[text_var],
            get_save_path(
                out_dir, "{}_{}_{}.png".format(target_val, text_var, "word_cloud")
            ),
        )
        plot_word_count_distribution(
            data[text_var],
            get_save_path(
                out_dir, "{}_{}_{}.jpg".format(target_val, text_var, "word_count_dist")
            ),
            bins,
        )
        plot_avg_word_len_distribution(
            data[text_var],
            get_save_path(
                out_dir,
                "{}_{}_{}.jpg".format(target_val, text_var, "avg_word_len_dist"),
            ),
            bins,
        )
        log.info("Visualization of {} field successful".format(text_var))


def visualize_categories(data, target_val, out_dir, config):
    for cat_var in config.variables.cat_vars:
        plot_category_distribution(
            data[cat_var],
            cat_var,
            get_save_path(
                out_dir, "{}_{}_{}.jpg".format(target_val, cat_var, "distribution")
            ),
        )
        log.info("Visualization of {} distribution successful".format(cat_var))


def visualize(in_path, target_val, out_dir, config):
    target_var = config.variables.target_var

    data = pd.read_csv(in_path).fillna("")
    log.info("Read data file successful")

    data[target_var] = target_val

    visualize_text(data, config, target_val, out_dir)
    log.info("Done text fields visualization")

    visualize_categories(data, target_val, out_dir, config)
    log.info("Done categorical fields visualization")

    return data


def visualize_label_distribution(data, out_dir, config):
    plot_label_distribution(
        data, get_save_path(out_dir, "label_distribution.jpg"), config
    )

    log.info("Visualization of label distribution successful")
