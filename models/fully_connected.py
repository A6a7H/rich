import torch
import torch.nn as nn
import typing as tp


class Fully_connected(nn.Module):
    """
    Simple fully connected neural network
    :param in_channel: size of input tensor
    :param hidden_channel: hidden tensor
    :param out_channel: size of output tensor
    :return: tensor
    """

    def __init__(
        self, in_channel: int = 1, hidden_channel: int = 64, out_channel: int = 1
    ):
        super(Fully_connected, self).__init__()
        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.out_channel = out_channel

        self.crit = nn.Sequential(
            self._make_crit_block(self.in_channel, self.hidden_channel),
            self._make_crit_block(self.hidden_channel, self.hidden_channel * 2),
            self._make_crit_block(
                self.hidden_channel * 2, self.out_channel, final_layer=True
            ),
        )

    def _make_crit_block(
        self,
        in_channel: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        final_layer: bool = False,
    ):
        if not final_layer:
            return nn.Sequential(
                nn.Linear(in_channel, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(0.02),
            )
        else:
            return nn.Sequential(
                nn.Linear(in_channel, out_channels),
            )

    def forward(self, image):
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)

def create_fcn_model(config: tp.Dict[str, tp.Any], model_type: str='classification'):
    if model_type == 'classification':
        model = Fully_connected(**config)
    elif model_type == 'regression':
        model = Fully_connected(**config)
    else:
        raise NameError(
            "Unknown model type: {}".format(model_type)
        )
    return model
