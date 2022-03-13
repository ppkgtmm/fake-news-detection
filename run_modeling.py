import hydra
from modeling import build_model


@hydra.main(config_path="config", config_name="modeling.yaml")
def run_modeling(config):
    build_model(config)


if __name__ == "__main__":
    run_modeling()
