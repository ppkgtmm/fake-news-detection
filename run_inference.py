import hydra
from inference import serve_model


@hydra.main(config_path="config", config_name="inference.yaml")
def run_inference(config):
    serve_model(config)


if __name__ == "__main__":
    run_inference()
