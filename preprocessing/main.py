import pandas as pd
from preprocessing.utilities import clean_text


def preprocess(in_path, out_path, config):
    data = pd.read_csv(in_path)

    data.loc[:, config.variables.text_vars].fillna("", inplace=True)

    for text_var in config.variables.text_vars:
        data[text_var] = data[text_var].map(clean_text)

    data.to_csv(out_path, index=False)
