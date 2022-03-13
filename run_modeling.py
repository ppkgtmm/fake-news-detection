import hydra
from modeling import do_modeling


@hydra.main(config_path="config", config_name="modeling.yaml")
def run_modeling(config):
    do_modeling(config)


if __name__ == "__main__":
    run_modeling()
