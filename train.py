import os
import hydra
import json
import logging
import neptune

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from models import WGAN, JSGAN
from trainer import Trainer
from utils import get_RICH

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def main(config: DictConfig):

    logger.info("Setting up logger")
    project = neptune.init(
        project_qualified_name=config["neptune"]["project_name"],
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNzFhNGI0YTYtNzM1OS00MDRkLWFiZWUtZmNkZjNiZjMzOTAxIn0=",
    )

    logger.info("Setting up experiment")
    neptune_logger = project.create_experiment(
        name=config["neptune"]["experiment_name"],
        tags=OmegaConf.to_container(config["neptune"]["tags"]),
        params=OmegaConf.to_container(config),
    )

    logger.info("Getting dataset")
    input_size, dll_shape, train_dataset, valid_dataset, scaler = get_RICH(
        config["data"]["particle"],
        config["data"]["drop_weights"],
        config["data"]["data_path"],
    )

    logger.info("Creating train dataloader")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        pin_memory=True,
    )

    logger.info("Creating validation dataloader")
    validation_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        pin_memory=True,
    )

    logger.info("Creating GAN model")
    model = JSGAN(config)

    save_path = config["save_path"]
    folder_name = f"{config['gan_type']}_{config['generator_architecture']}_{config['critic_architecture']}"
    save_path = os.path.join(save_path, folder_name)

    logger.info("Creating trainer")
    trainer = Trainer(
        gan_model=model,
        max_epoch=config["max_epoch"],
        display_step=config["display_step"],
        critic_step=config["critic_step"],
        device=config["device"],
        save_path=save_path,
        neptune_logger=neptune_logger,
    )
    logger.info("Calculating inference time")
    trainer.calculate_inference_time(
        (1, config["generator"]["params"]["input_size"]), config["repetitions"]
    )

    logger.info("Starting train")
    trainer.fit(
        train_loader=train_loader,
        validation_loader=validation_loader,
        start=config["starting_epoch"],
    )


if __name__ == "__main__":
    main()
