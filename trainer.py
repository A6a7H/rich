import os
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Callable, Any, List
from torch.nn import functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class Trainer():
    def __init__(
        self,
        gan_model,
        max_epoch: int = 10000,
        display_step: int = 1000,
        critic_step: int = 8,
        device: str = "cuda",
        save_path: str = ".",
        neptune_logger: Callable = None,
    ):  
        self.gan_model = gan_model
        self.max_epoch = max_epoch
        self.display_step = display_step
        self.critic_step = critic_step
        self.device = device
        self.save_path = save_path
        self.neptune_logger = neptune_logger
        self.generator_optimizer, self.critic_optimizer = gan_model.configure_optimizers()

    def fit(self, train_loader=None, validation_loader=None, start=0):
        for epoch in tqdm(range(self.max_epoch)):
            for iteration, batch in enumerate(train_loader):
                for _ in range(self.critic_step):
                    self.critic_optimizer.zero_grad()

                    critic_losses = self.gan_model.train_critic(batch)
                    critic_total_loss = critic_losses['C/loss']
                    critic_total_loss.backward(retain_graph=True)

                    self.critic_optimizer.step()

                self.generator_optimizer.zero_grad()

                generator_losses = self.gan_model.train_critic(batch)
                generator_total_loss = critic_losses['G/loss']
                generator_total_loss.backward(retain_graph=True)
                self.generator_optimizer.step()

                for key, value in generator_losses.items():
                    self.neptune_logger.log_metric(key, value.item())

                for key, value in critic_losses.items():
                    self.neptune_logger.log_metric(key, value.item())

                if iteration % self.display_step == 0:
                    name = f"epoch_{epoch}_iter_{iteration}.pt"
                    save_path_with_iter = os.path.join(self.save_path, name)
                    self.gan_model.save_models(save_path_with_iter, epoch)
                    with torch.no_grad():
                        generated_list = []
                        dlls_list = []
                        for batch in self.validation_loader:
                            generated = self.gan_model.generate(batch)
                            generated_list.append(generated)
                            dlls_list.append(batch[1])
                        fig = self.gan_model.get_histograms(torch.cat(dlls_list, dim=0), 
                                                            torch.cat(generated_list, dim=0))
                        self.neptune_logger.log_image("Histograms", fig)