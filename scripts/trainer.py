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

from typing import Callable
from torch.nn import functional as F
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
        super(RichNodeWGANTrainer, self).__init__()
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
            for x, dlls in self.train_loader:
                x = x.to(self.device)
                dlls = dlls.to(self.device)
                dlls = dlls.unsqueeze(dim=1)

                for _ in range(self.critic_iteration):
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
                    
                    generator_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred)

                    epsilon = torch.rand(real_full.shape[0], 1)
                    epsilon = epsilon.expand(real_full.size())
                    epsilon = epsilon.to(self.device)

                    gradient = get_gradient(
                        self.discriminator_model, real_full, generated, None, epsilon
                    )
                    gp = gradient_penalty(gradient)

                    critic_loss = 10 * gp + generator_loss
                    critic_loss.backward(retain_graph=True)
                    self.discriminator_optimizer.step()

            self.generator_model.train()
            self.discriminator_model.eval()
            self.generator_optimizer.zero_grad()

            noized_x = torch.cat(
                [x, get_noise(x.shape[0], self.z_dimensions).to(self.device)],
                dim=1,
            )

            real_full = torch.cat([dlls, x], dim=1)
            generated = torch.cat([self.generator_model(noized_x), x], dim=1)
            crit_fake_pred = self.discriminator_model(generated)

            generator_loss = -torch.mean(crit_fake_pred)

            generator_loss.backward()
            self.generator_optimizer.step()

            self.logger.log_metric("Generator loss", generator_loss.item() / len(self.train_loader))
            self.logger.log_metric("Critic loss", critic_loss.item() / len(self.train_loader))
            
            if self.generator_scheduler is not None:
                self.generator_scheduler.step()
            if self.discriminator_scheduler is not None:
                self.discriminator_scheduler.step()

            if epoch % self.display_step == 0:
                with torch.no_grad():
                    x, dlls = next(iter(self.validation_loader))
                    x = x.to(self.device)
                    dlls = dlls.to(self.device)
                    dlls = dlls.unsqueeze(dim=1)

                    real_full_0 = torch.cat([dlls, x], dim=1)
                    noized_x = torch.cat(
                        [x, get_noise(x.shape[0], self.z_dimensions).to(self.device)],
                        dim=1,
                    )
                    generated = self.generator_model(noized_x)
                    self.draw(dlls, generated)

    def draw(
        self,
        dlls,
        generated,
        dll_name: List[str] = [
            "RichDLLe",
            # "RichDLLk",
            # "RichDLLmu",
            # "RichDLLp",
            # "RichDLLbt",
        ],
    ):
        clear_output(False)
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        for INDEX, ax in zip([i for i in range(len(dll_name))], axs.flatten()):
            _, bins, _ = ax.hist(dlls[:, INDEX].cpu(), bins=100, label="data")
            ax.hist(generated[:, INDEX].cpu(), bins=bins, label="generated", alpha=0.5)
            ax.legend()
            ax.set_title(dll_name[INDEX])
        self.logger.log_image('diagram', fig)

class RichNodeCramerTrainer(BaseTrainer):
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
        super(RichNodeCramerTrainer, self).__init__()
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
            for x, dlls in self.train_loader:
                x = x.to(self.device)
                dlls = dlls.to(self.device)
                dlls = dlls.unsqueeze(dim=1)

                for _ in range(self.critic_iteration):
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

                    generator_loss = torch.mean(self.cramer_critic(real_full_0, generated_2) - 
                                    self.cramer_critic(generated_1, generated_2))
        
                    gradient_penalty = self.calc_gradient_penalty(real_full_0, generated_1, generated_2)
                    
                    critic_loss = 10 * gradient_penalty - generator_loss
                    critic_loss.backward()
                    optC.step()

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
            
            generator_loss = torch.mean(self.cramer_critic(real_full_0, generated_2) -
                self.cramer_critic(generated_1, generated_2))


            generator_loss.backward()
            self.generator_optimizer.step()

            self.logger.log_metric("Generator loss", generator_loss.item() / len(self.train_loader))
            self.logger.log_metric("Critic loss", critic_loss.item() / len(self.train_loader))
            
            if self.generator_scheduler is not None:
                self.generator_scheduler.step()
            if self.discriminator_scheduler is not None:
                self.discriminator_scheduler.step()

            if epoch % self.display_step == 0:
                with torch.no_grad():
                    x, dlls = next(iter(self.validation_loader))
                    x = x.to(self.device)
                    dlls = dlls.to(self.device)
                    dlls = dlls.unsqueeze(dim=1)

                    real_full_0 = torch.cat([dlls, x], dim=1)
                    noized_x = torch.cat(
                        [x, get_noise(x.shape[0], self.z_dimensions).to(self.device)],
                        dim=1,
                    )
                    generated = self.generator_model(noized_x)
                    self.draw(dlls, generated)

    def draw(
        self,
        dlls,
        generated,
        dll_name: List[str] = [
            "RichDLLe",
            # "RichDLLk",
            # "RichDLLmu",
            # "RichDLLp",
            # "RichDLLbt",
        ],
    ):
        clear_output(False)
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        for INDEX, ax in zip([i for i in range(len(dll_name))], axs.flatten()):
            _, bins, _ = ax.hist(dlls[:, INDEX].cpu(), bins=100, label="data")
            ax.hist(generated[:, INDEX].cpu(), bins=bins, label="generated", alpha=0.5)
            ax.legend()
            ax.set_title(dll_name[INDEX])
        self.logger.log_image('diagram', fig)


    def cramer_critic(self, x, y):
        discriminated_x = self.discriminator_model(x)
        return torch.norm(discriminated_x - self.discriminator_model(y), dim=1) - torch.norm(discriminated_x, dim=1)

    
    def calc_gradient_penalty(self, real_data, fake_data, fake_data2):
        alpha = torch.rand(real_data.shape[0], 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)


        disc_interpolates = self.cramer_critic(interpolates, fake_data2)

        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

class RichNodeGANTrainer(BaseTrainer):
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
        critic_type = 'FCN_critic',
        generator_type = 'NODE_generator',
    ):
        super(RichNodeGANTrainer, self).__init__()
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
        self.critic_type = critic_type
        self.generator_type = generator_type

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            for x, dlls in self.train_loader:
                x = x.to(self.device)
                dlls = dlls.to(self.device)
                dlls = dlls.unsqueeze(dim=1)

                for _ in range(self.critic_iteration):
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

                    real_loss = F.binary_cross_entropy_with_logits(crit_real_pred, torch.ones_like(crit_real_pred) * 0.9)
                    fake_loss = F.binary_cross_entropy_with_logits(crit_fake_pred, torch.zeros_like(crit_fake_pred))

                    critic_loss = (real_loss + fake_loss) / 2
                    
                    critic_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.discriminator_model.parameters(), 1)
                    self.discriminator_optimizer.step()

            self.generator_model.train()
            self.discriminator_model.eval()
            self.generator_optimizer.zero_grad()

            noized_x = torch.cat(
                [x, get_noise(x.shape[0], self.z_dimensions).to(self.device)],
                dim=1,
            )

            real_full = torch.cat([dlls, x], dim=1)
            generated = torch.cat([self.generator_model(noized_x), x], dim=1)
            crit_fake_pred = self.discriminator_model(generated)

            generator_loss = F.binary_cross_entropy_with_logits(crit_fake_pred, torch.ones_like(crit_fake_pred) * 0.9)

            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator_model.parameters(), 1)
            self.generator_optimizer.step()

            self.logger.log_metric("Generator loss", generator_loss.item() / len(self.train_loader))
            self.logger.log_metric("Critic loss", critic_loss.item() / len(self.train_loader))

            if self.generator_scheduler is not None:
                self.generator_scheduler.step()
            if self.discriminator_scheduler is not None:
                self.discriminator_scheduler.step()

            if epoch % self.display_step == 0:
                torch.save(self.generator_model.state_dict(), f"./drive/MyDrive/rich/{self.generator_type}_{epoch}.pth")
                torch.save(self.discriminator_model.state_dict(), f"./drive/MyDrive/rich/{self.critic_type}_{epoch}.pth")
                with torch.no_grad():
                    x, dlls = next(iter(self.validation_loader))
                    x = x.to(self.device)
                    dlls = dlls.to(self.device)
                    dlls = dlls.unsqueeze(dim=1)

                    real_full_0 = torch.cat([dlls, x], dim=1)
                    noized_x = torch.cat(
                        [x, get_noise(x.shape[0], self.z_dimensions).to(self.device)],
                        dim=1,
                    )
                    generated = self.generator_model(noized_x)
                    self.draw(dlls, generated)

    def draw(
        self,
        dlls,
        generated,
        dll_name: List[str] = [
            "RichDLLe",
            # "RichDLLk",
            # "RichDLLmu",
            # "RichDLLp",
            # "RichDLLbt",
        ],
    ):
        clear_output(False)
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        for INDEX, ax in zip([i for i in range(len(dll_name))], axs.flatten()):
            _, bins, _ = ax.hist(dlls[:, INDEX].cpu(), bins=100, label="data")
            ax.hist(generated[:, INDEX].cpu(), bins=bins, label="generated", alpha=0.5)
            ax.legend()
            ax.set_title(dll_name[INDEX])
        self.logger.log_image('diagram', fig)