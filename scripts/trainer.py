from typing import Callable
from utils.modelutils import get_noise, get_gradient, gradient_penalty
from torch.utils.data import DataLoader
from abc import ABC
from tqdm import tqdm
import torch
import numpy as np


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

    # def draw(self):
    #     raise NotImplementedError


class RichNodeTrainer(BaseTrainer):
    def __init__(
        self,
        generator_model: torch.nn.Module,
        discriminator_model: torch.nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        generator_optimizer: torch.optim,
        discriminator_optimizer: torch.optim,
        generator_loss: Callable = None,
        critic_loss: Callable = None,
        z_dimensions: int = 32,
        epochs: int = 10000,
        display_step: int = 1000,
        critic_iteration: int = 8,
        generator_iteration: int = 1,
        weights_exist: bool = True,
        device: str = "cuda",
    ):
        super(RichNodeTrainer, self).__init__()
        self.generator_model = generator_model
        self.discriminator_model = discriminator_model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss = generator_loss
        self.critic_loss = critic_loss
        self.z_dimensions = z_dimensions
        self.epochs = epochs
        self.critic_iteration = critic_iteration
        self.generator_iteration = generator_iteration
        self.weights_exist = weights_exist
        self.device = device

    def train(self):
        for epoch in tqdm(self.epochs):
            for critic_iteration in range(critic_iteration):
                if self.weights_exist:
                    x, weight, dlls = [
                        i.to(self.device) for i in next(self.train_loader)
                    ]
                else:
                    x, dlls = [i.to(self.device) for i in next(self.train_loader)]
                    weight = torch.ones((x.shape[0]))

                self.discriminator_optimizer.zero_grad()

                noized_x = torch.cat(
                    [x, get_noise(x.shape[0], self.z_dimensions).to(self.device)],
                    dim=1,
                )

                real_full_0 = torch.cat([dlls, x], dim=1)
                generated_1 = torch.cat([self.generator_model(noized_x), x], dim=1)
                generated_2 = torch.cat([self.generator_model(noized_x), x], dim=1)

                generator_loss = torch.mean(
                    self.critic_loss(real_full_0, generated_2) * weight - 
                    self.critic_loss(generated_1, generated_2) * weight
                )
                # epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
                epsilon = torch.rand(real_full_0.shape[0], 1)
                epsilon = epsilon.expand(real_full_0.size())
                epsilon = epsilon.to(self.device)

                gradient = get_gradient(self.critic_loss, real_full_0, generated_1, generated_2, epsilon)
                gradient_penalty = gradient_penalty(gradient)
                
                critic_loss = 0.7 * gradient_penalty - generator_loss
                critic_loss.backward()
                self.discriminator_optimizer.step()

            if self.weights_exist:
                x, weight, dlls = [
                    i.to(self.device) for i in next(self.train_loader)
                ]
            else:
                x, dlls = [i.to(self.device) for i in next(self.train_loader)]
                weight = torch.ones((x.shape[0]))

            self.generator_model.train()
            self.discriminator_model.eval()
            self.generator_optimizer.zero_grad()

            noized_x = torch.cat(
                    [x, get_noise(x.shape[0], self.z_dimensions).to(self.device)],
                    dim=1,
                )
            real_full_0 = torch.cat([dlls, x], dim=1)
            generated_1 = torch.cat([self.generator_model(noized_x), x], dim=1)
            generated_2 = torch.cat([self.generator_model(noized_x), x], dim=1)

            generator_loss = torch.mean(
                    self.generator_loss(real_full_0, generated_2) * weight - 
                    self.generator_loss(generated_1, generated_2) * weight
                )

            generator_loss.backward()
            self.generator_optimizer.step()

