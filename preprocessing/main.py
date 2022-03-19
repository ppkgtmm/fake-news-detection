import logging
import pandas as pd
from preprocessing.utilities import clean_text

log = logging.getLogger(__name__)


def preprocess(in_path, out_path, config):
    data = pd.read_csv(in_path)

    if config.variables.drop_vars:
        data.drop(columns=config.variables.drop_vars, inplace=True)

    data.loc[:, config.variables.text_vars].fillna("", inplace=True)

    for text_var in config.variables.text_vars:
        data[text_var] = data[text_var].map(clean_text)

        log.info("Done preprocessing {} variable".format(text_var))

    data.to_csv(out_path, index=False)
