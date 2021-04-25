import torch
import logging
import typing as tp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

from .metrics import calculate_roc_auc
from models import (get_gradient, 
                    get_noise, 
                    gradient_penalty)
from models import (create_node_model, 
                    create_fcn_model)

logger = logging.getLogger(__name__)

class JSGAN():
    def __init__(self, config: tp.Dict[str, tp.Any]):
        self.params = config
        self.create_generator()
        self.create_critic()

    def configure_optimizers(self):
        if self.params['generator']['optimizer'] == 'adam':
            generator_optimizer = torch.optim.Adam(
                self.generator_model.parameters(),
                lr=self.params['generator']['learning_rate'],
                weight_decay=self.params['generator']['weight_decay'],
                betas=(self.params["generator"]["beta1"], self.params["generator"]["beta2"])
            )
        else:
            raise NameError("Unknown optimizer name")
        
        if self.params['critic']['optimizer'] == 'adam':
            critic_optimizer = torch.optim.Adam(
                self.critic_model.parameters(),
                lr=self.params['critic']['learning_rate'],
                weight_decay=self.params['critic']['weight_decay'],
                betas=(self.params["critic"]["beta1"], self.params["critic"]["beta2"])
            )
        else:
            raise NameError("Unknown optimizer name")

        return generator_optimizer, critic_optimizer

    def save_models(self, path, epoch):
        torch.save({
            'epoch': epoch,
            'generator_model_state_dict': self.generator_model.state_dict(),
            'critic_model_state_dict': self.critic_model.state_dict(),
            }, path)

    def load_models(self, path):
        checkpoint = torch.load(path)
        self.generator_model.load_state_dict(checkpoint['generator_model_state_dict'])
        self.critic_model.load_state_dict(checkpoint['critic_model_state_dict'])
        return checkpoint

    def create_generator(self):
        if self.params["generator_architecture"].lower() == "node":
            logger.info("Creating NODE model")
            logger.info(f"params: \n{self.params['generator']}")
            self.generator_model = create_node_model(self.params['generator']['params'], 
                                    model_type=self.params['generator_model_type'])
        elif self.params["critic_architecture"].lower() == "fcn":
            logger.info("Creating FCN model")
            logger.info(f"params: \n{self.params['generator']}")
            self.generator_model = create_fcn_model(self.params['generator'], 
                                    model_type=self.params['generator_model_type'])
        else:
            raise NameError(
                "Unknown generator architecture: {}".format(self.params["generator_architecture"])
            )
        
        generator_checkpoint_path = self.params['checkpoint_path']
        if generator_checkpoint_path:
            generator_checkpoint = torch.load(generator_checkpoint_path)
            self.generator_model.load_state_dict(generator_checkpoint["generator_model_state_dict"])

        self.generator_model.to(self.params["device"])

    def create_critic(self):
        if self.params["critic_architecture"].lower() == "node":
            logger.info("Creating NODE model")
            logger.info(f"params: \n{self.params['critic']}")
            self.critic_model = create_node_model(self.params['critic'], 
                                    model_type=self.params['critic_model_type'])
        elif self.params["critic_architecture"].lower() == "fcn":
            logger.info("Creating FCN model")
            logger.info(f"params: \n{self.params['critic']}")
            self.critic_model = create_fcn_model(self.params['critic'], 
                                    model_type=self.params['critic_model_type'])
        else:
            raise NameError(
                "Unknown critic architecture: {}".format(self.params['critic_architecture'])
            )

        critic_checkpoint_path = self.params["checkpoint_path"]
        if critic_checkpoint_path:
            critic_checkpoint = torch.load(critic_checkpoint_path)
            self.critic_model.load_state_dict(critic_checkpoint["critic_model_state_dict"])

        self.critic_model.to(self.params["device"])

    def train_generator(self, batch):
        if self.params['data']['drop_weights']:
            x, dlls = batch
        else:
            x, dlls, weight = batch

        x = x.to(self.params["device"]).type(torch.float)
        dlls = dlls.to(self.params["device"]).type(torch.float)

        noized_x = torch.cat(
            [x, get_noise(x.shape[0], self.params['noise_dim']).to(self.params["device"])],
            dim=1,
        )

        real_full = torch.cat([dlls, x], dim=1)
        generated = torch.cat([self.generator_model(noized_x), x], dim=1)
        crit_fake_pred = self.critic_model(generated)

        generator_loss = F.binary_cross_entropy_with_logits(crit_fake_pred, torch.ones_like(crit_fake_pred) * 0.9)

        generator_result = {
            'G/loss' : generator_loss
        }

        return generator_result

    def train_critic(self, batch):
        if self.params['data']['drop_weights']:
            x, dlls = batch
        else:
            x, dlls, weight = batch
        x = x.to(self.params["device"]).type(torch.float)
        dlls = dlls.to(self.params["device"]).type(torch.float)

        noized_x = torch.cat(
            [x, get_noise(x.shape[0], self.params['noise_dim']).to(self.params["device"])],
            dim=1,
        )

        real_full = torch.cat([dlls, x], dim=1)
        generated = torch.cat([self.generator_model(noized_x), x], dim=1)

        crit_fake_pred = self.critic_model(generated.detach())
        crit_real_pred = self.critic_model(real_full)

        fake_loss = F.binary_cross_entropy_with_logits(crit_fake_pred, torch.zeros_like(crit_fake_pred))
        real_loss = F.binary_cross_entropy_with_logits(crit_real_pred, torch.ones_like(crit_real_pred) * 0.9)

        critic_loss = (real_loss + fake_loss) / 2
        critic_result = {
            'C/loss' : critic_loss,
            'C/fake_loss': fake_loss,
            'C/real_loss': real_loss,
        }

        return critic_result

    @torch.no_grad()
    def generate(self, batch):
        if self.params['data']['drop_weights']:
            x, dlls = batch
        else:
            x, dlls, weight = batch
            
        x = x.to(self.params["device"]).type(torch.float)
        dlls = dlls.to(self.params["device"]).type(torch.float)

        noize = get_noise(x.shape[0], self.params['noise_dim']).to(self.params["device"])
        noized_x = torch.cat(
            [x, noize],
            dim=1,
        )
        generated = self.generator_model(noized_x)
        return generated


    def get_histograms(self, generated, real):
        features_names = self.params['features_names']
        num_of_subplots = len(features_names)
        
        plt.clf()
        fig, axes = plt.subplots(num_of_subplots, 1, figsize=(10, 16))
        plt.suptitle("Histograms")

        for feature_index, (feature_name, ax) in enumerate(
            zip(features_names, axes.flatten())
        ):
            ax.set_title(feature_name)

            sns.distplot(
                real[:, feature_index],
                bins=100,
                label="real",
                color="b",
                hist_kws={"alpha": 1.0},
                ax=ax,
                norm_hist=True,
                kde=False,
            )

            sns.distplot(
                generated[:, feature_index].cpu(),
                bins=100,
                label="generated",
                color="r",
                hist_kws={"alpha": 0.5},
                ax=ax,
                norm_hist=True,
                kde=False,
            )

            if feature_index == 0:
                ax.legend()

        return fig

    def validation_step(self, generated, real, weights=None):
        figure = self.get_histograms(generated, real)
        roc_auc_score = self.get_roc_auc_score(generated, real)
        outputs = {
            'histogram': figure,
            'rocauc': roc_auc_score
        }
        return outputs


    def get_roc_auc_score(self, generated, real):
        X = np.concatenate((generated, real))
        y = np.array([0] * generated.shape[0] + [1] * real.shape[0])

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.params["seed"],
            stratify=y,
            shuffle=True,
        )

        classifier = CatBoostClassifier(iterations=1000, thread_count=10, silent=True)
        classifier.fit(X_train, y_train)
        predicted = classifier.predict(X_test)
        roc_auc = calculate_roc_auc(y_test, predicted)
        return roc_auc
