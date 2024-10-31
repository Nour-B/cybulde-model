import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from cybulde.config_schemas.training.training_task_schemas import setup_config

setup_config()


@hydra.main(config_name="test_training_task", version_base= None)
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

    '''
    model = instantiate(config)

    texts = ["Hello, how are you"]
    encodings = model.backbone.transformation(texts)

    output = model(encodings)
    print(f"{output.shape=}")
    '''


if __name__ == "__main__":
    main()
