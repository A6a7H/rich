from typing import Callable
from rich.utils.modelutils import get_noise, get_gradient, gradient_penalty
from torch.utils.data import DataLoader
from abc import ABC
from tqdm import tqdm
from typing import Any, List
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

class BaseTrainer(ABC):
    def __init__(self):
        pass

    def train(self):
        raise NotImplementedError

    # def train_generator(self):
    #     raise NotImplementedError

    # def train_discriminator(self):
    #     raise NotImplementedError

    # def evaluate(self):
    #     raise NotImplementedError

    def draw(self):
        raise NotImplementedError


class RichNodeWGANTrainer(BaseTrainer):
    def __init__(
        self,
        generator_model: torch.nn.Module,
        discriminator_model: torch.nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        generator_optimizer: torch.optim,
        discriminator_optimizer: torch.optim,
        generator_scheduler: torch.optim.lr_scheduler,
        discriminator_scheduler: torch.optim.lr_scheduler,
        generator_loss: Callable = None,
        critic_loss: Callable = None,
        z_dimensions: int = 32,
        epochs: int = 10000,
        display_step: int = 1000,
        critic_iteration: int = 8,
        generator_iteration: int = 1,
        weights_exist: bool = True,
        device: str = "cuda",
        logger: Any = None,
    ):
        super(RichNodeTrainer, self).__init__()
        self.generator_model = generator_model
        self.discriminator_model = discriminator_model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_scheduler = generator_scheduler
        self.discriminator_scheduler = discriminator_scheduler
        self.generator_loss = generator_loss
        self.critic_loss = critic_loss
        self.z_dimensions = z_dimensions
        self.epochs = epochs
        self.display_step = display_step
        self.critic_iteration = critic_iteration
        self.generator_iteration = generator_iteration
        self.weights_exist = weights_exist
        self.device = device
        self.logger = logger

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            for critic_iteration in range(self.critic_iteration):
                if self.weights_exist:
                    x, weight, dlls = [
                        i.to(self.device) for i in next(self.train_loader)
                    ]
                else:
                    x, dlls = [i.to(self.device) for i in next(self.train_loader)]
                    weight = torch.ones((x.shape[0]), device=self.device)

                self.discriminator_optimizer.zero_grad()
                self.generator_model.eval()
                self.discriminator_model.train()

                noized_x = torch.cat(
                    [x, get_noise(x.shape[0], self.z_dimensions).to(self.device)],
                    dim=1,
                )
                real_full = torch.cat([dlls, x], dim=1)
                generated = torch.cat([self.generator_model(noized_x), x], dim=1)

                crit_fake_pred = self.discriminator_model(generated)
                crit_real_pred = self.discriminator_model(real_full)
                
                generator_loss = torch.mean(crit_fake_pred - crit_real_pred)
                # epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
                epsilon = torch.rand(real_full_0.shape[0], 1)
                epsilon = epsilon.expand(real_full_0.size())
                epsilon = epsilon.to(self.device)

                gradient = get_gradient(
                    self.discriminator_model, real_full, generated, None, epsilon
                )
                gp = gradient_penalty(gradient)

                critic_loss = 10 * gp + generator_loss
                critic_loss.backward(retain_graph=True)
                self.discriminator_optimizer.step()

            if self.weights_exist:
                x, weight, dlls = [i.to(self.device) for i in next(self.train_loader)]
            else:
                x, dlls = [i.to(self.device) for i in next(self.train_loader)]
                weight = torch.ones((x.shape[0]), device=self.device)

            self.generator_model.train()
            self.discriminator_model.eval()
            self.generator_optimizer.zero_grad()

            noized_x = torch.cat(
                [x, get_noise(x.shape[0], self.z_dimensions).to(self.device)],
                dim=1,
            )

            real_full = torch.cat([dlls, x], dim=1)
            generated = torch.cat([self.generator_model(noized_x_1), x], dim=1)
            crit_fake_pred = self.discriminator_model(fake)

            generator_loss = -torch.mean(crit_fake_pred)

            generator_loss.backward()
            self.generator_optimizer.step()

            self.logger.log_metrics(
                {
                    "Generator loss": generator_loss.item(),
                    "Critic loss": critic_loss.item(),
                    "Gradient Norm": torch.norm(
                        gradient.view(len(gradient), -1).norm(2, dim=1)
                    ),
                },
                step=epoch,
            )

            self.generator_scheduler.step()
            self.discriminator_scheduler.step()

            if epoch % self.display_step == 0:
                with torch.no_grad():
                    if self.weights_exist:
                        x, weight, dlls = [
                            i.to(self.device) for i in next(self.validation_loader)
                        ]
                    else:
                        x, dlls = [i.to(self.device) for i in next(self.validation_loader)]
                        weight = torch.ones((x.shape[0]))
                    real_full_0 = torch.cat([dlls, x], dim=1)
                    noized_x = torch.cat(
                        [x, get_noise(x.shape[0], self.z_dimensions).to(self.device)],
                        dim=1,
                    )
                    generated = self.generator_model(noized_x)
                    self.draw(dlls, generated)
        self.logger.end()

    def draw(
        self,
        dlls,
        generated,
        dll_name: List[str] = [
            "RichDLLe",
            "RichDLLk",
            "RichDLLmu",
            "RichDLLp",
            "RichDLLbt",
        ],
    ):
        clear_output(False)
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        for INDEX, ax in zip([i for i in range(len(dll_name))], axs.flatten()):
            _, bins, _ = ax.hist(dlls[:, INDEX].cpu(), bins=100, label="data")
            ax.hist(generated[:, INDEX].cpu(), bins=bins, label="generated", alpha=0.5)
            ax.legend()
            ax.set_title(dll_name[INDEX])
        self.logger.log_figure()
        plt.show()



class RichNodeTrainer(BaseTrainer):
    def __init__(
        self,
        generator_model: torch.nn.Module,
        discriminator_model: torch.nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        generator_optimizer: torch.optim,
        discriminator_optimizer: torch.optim,
        generator_scheduler: torch.optim.lr_scheduler,
        discriminator_scheduler: torch.optim.lr_scheduler,
        generator_loss: Callable = None,
        critic_loss: Callable = None,
        z_dimensions: int = 32,
        epochs: int = 10000,
        display_step: int = 1000,
        critic_iteration: int = 8,
        generator_iteration: int = 1,
        weights_exist: bool = True,
        device: str = "cuda",
        logger: Any = None,
    ):
        super(RichNodeTrainer, self).__init__()
        self.generator_model = generator_model
        self.discriminator_model = discriminator_model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_scheduler = generator_scheduler
        self.discriminator_scheduler = discriminator_scheduler
        self.generator_loss = generator_loss
        self.critic_loss = critic_loss
        self.z_dimensions = z_dimensions
        self.epochs = epochs
        self.display_step = display_step
        self.critic_iteration = critic_iteration
        self.generator_iteration = generator_iteration
        self.weights_exist = weights_exist
        self.device = device
        self.logger = logger

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            for critic_iteration in range(self.critic_iteration):
                if self.weights_exist:
                    x, weight, dlls = [
                        i.to(self.device) for i in next(self.train_loader)
                    ]
                else:
                    x, dlls = [i.to(self.device) for i in next(self.train_loader)]
                    weight = torch.ones((x.shape[0]), device=self.device)

                self.discriminator_optimizer.zero_grad()
                self.generator_model.eval()
                self.discriminator_model.train()

                noized_x_1 = torch.cat(
                    [x, get_noise(x.shape[0], self.z_dimensions).to(self.device)],
                    dim=1,
                )

                noized_x_2 = torch.cat(
                    [x, get_noise(x.shape[0], self.z_dimensions).to(self.device)],
                    dim=1,
                )

                real_full_0 = torch.cat([dlls, x], dim=1)
                generated_1 = torch.cat([self.generator_model(noized_x_1), x], dim=1)
                generated_2 = torch.cat([self.generator_model(noized_x_2), x], dim=1)

                generator_loss = torch.mean(
                    self.critic_loss(real_full_0, generated_2) * weight
                    - self.critic_loss(generated_1, generated_2) * weight
                )
                # epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
                epsilon = torch.rand(real_full_0.shape[0], 1)
                epsilon = epsilon.expand(real_full_0.size())
                epsilon = epsilon.to(self.device)

                gradient = get_gradient(
                    self.critic_loss, real_full_0, generated_1, generated_2, epsilon
                )
                gp = gradient_penalty(gradient)

                critic_loss = 10 * gp - generator_loss
                critic_loss.backward()
                self.discriminator_optimizer.step()

            if self.weights_exist:
                x, weight, dlls = [i.to(self.device) for i in next(self.train_loader)]
            else:
                x, dlls = [i.to(self.device) for i in next(self.train_loader)]
                weight = torch.ones((x.shape[0]), device=self.device)

            self.generator_model.train()
            self.discriminator_model.eval()
            self.generator_optimizer.zero_grad()

            noized_x_1 = torch.cat(
                [x, get_noise(x.shape[0], self.z_dimensions).to(self.device)],
                dim=1,
            )

            noized_x_2 = torch.cat(
                [x, get_noise(x.shape[0], self.z_dimensions).to(self.device)],
                dim=1,
            )

            real_full_0 = torch.cat([dlls, x], dim=1)
            generated_1 = torch.cat([self.generator_model(noized_x_1), x], dim=1)
            generated_2 = torch.cat([self.generator_model(noized_x_2), x], dim=1)

            generator_loss = torch.mean(
                self.generator_loss(real_full_0, generated_2) * weight
                - self.generator_loss(generated_1, generated_2) * weight
            )

            generator_loss.backward()
            self.generator_optimizer.step()

            self.logger.log_metrics(
                {
                    "Generator loss": generator_loss.item(),
                    "Critic loss": critic_loss.item(),
                    "Gradient Norm": torch.norm(
                        gradient.view(len(gradient), -1).norm(2, dim=1)
                    ),
                },
                step=epoch,
            )

            self.generator_scheduler.step()
            self.discriminator_scheduler.step()

            if epoch % self.display_step == 0:
                with torch.no_grad():
                    if self.weights_exist:
                        x, weight, dlls = [
                            i.to(self.device) for i in next(self.validation_loader)
                        ]
                    else:
                        x, dlls = [i.to(self.device) for i in next(self.validation_loader)]
                        weight = torch.ones((x.shape[0]))
                    real_full_0 = torch.cat([dlls, x], dim=1)
                    noized_x = torch.cat(
                        [x, get_noise(x.shape[0], self.z_dimensions).to(self.device)],
                        dim=1,
                    )
                    generated = self.generator_model(noized_x)
                    self.draw(dlls, generated)
        self.logger.end()

    def draw(
        self,
        dlls,
        generated,
        dll_name: List[str] = [
            "RichDLLe",
            "RichDLLk",
            "RichDLLmu",
            "RichDLLp",
            "RichDLLbt",
        ],
    ):
        clear_output(False)
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        for INDEX, ax in zip([i for i in range(len(dll_name))], axs.flatten()):
            _, bins, _ = ax.hist(dlls[:, INDEX].cpu(), bins=100, label="data")
            ax.hist(generated[:, INDEX].cpu(), bins=bins, label="generated", alpha=0.5)
            ax.legend()
            ax.set_title(dll_name[INDEX])
        self.logger.log_figure()
        plt.show()
