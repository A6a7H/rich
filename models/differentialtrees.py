import torch
import torch.nn as nn
import typing as tp

from .node import lib


class ClassificationDifferentialTree(nn.Module):
    """
    Ensemble of Differential Tree for Classification task
    :param input_size: number of features in the input tensor
    :param layer_dim: number of trees in this layer
    :param num_layers: number of layers
    :param tree_dim: number of response channels in the response of individual tree(number of class)
    :param depth: number of splits in every tree
    """

    def __init__(
        self, input_size=8, layer_dim=128, num_layers=2, tree_dim=2, depth=6, **kwargs
    ):
        super(ClassificationDifferentialTree, self).__init__()
        self.input_size = input_size
        self.layer_dim = layer_dim
        self.num_layers = num_layers
        self.tree_dim = tree_dim
        self.depth = depth

        self.layers = self._get_clf_tree_block(
            input_size, layer_dim, num_layers, tree_dim, depth
        )

    def _get_clf_tree_block(
        self, input_size=8, layer_dim=128, num_layers=2, tree_dim=2, depth=6
    ):
        return nn.Sequential(
            lib.DenseBlock(
                input_dim=input_size,
                layer_dim=layer_dim,
                num_layers=num_layers,
                tree_dim=tree_dim + 1,
                depth=depth,
                flatten_output=False,
                choice_function=lib.entmax15,
                bin_function=lib.sparsemax,
            ),
            lib.Lambda(lambda x: x[..., :tree_dim].mean(dim=-2)),
        )

    def forward(self, x, noise=None):
        return self.layers(x)


class RegressionDifferentialTree(nn.Module):
    """
    Ensemble of Differential Tree for Regression task
    :param input_size: number of features in the input tensor
    :param layer_dim: number of trees in this layer
    :param num_layers: number of layers
    :param tree_dim: number of response channels in the response of individual tree(number of class)
    :param depth: number of splits in every tree
    """

    def __init__(
        self, input_size=8, layer_dim=128, num_layers=2, tree_dim=2, depth=6, **kwargs
    ):
        super(RegressionDifferentialTree, self).__init__()
        self.input_size = input_size
        self.layer_dim = layer_dim
        self.num_layers = num_layers
        self.tree_dim = tree_dim
        self.depth = depth

        self.dlle = self._get_reg_tree_block(
            input_size, layer_dim, num_layers, tree_dim, depth
        )
        self.dllk = self._get_reg_tree_block(
            input_size, layer_dim, num_layers, tree_dim, depth
        )
        self.dllmu = self._get_reg_tree_block(
            input_size, layer_dim, num_layers, tree_dim, depth
        )
        self.dllp = self._get_reg_tree_block(
            input_size, layer_dim, num_layers, tree_dim, depth
        )
        self.dllbt = self._get_reg_tree_block(
            input_size, layer_dim, num_layers, tree_dim, depth
        )

    def _get_reg_tree_block(
        self, input_size=8, layer_dim=128, num_layers=2, tree_dim=2, depth=6
    ):
        return nn.Sequential(
            lib.DenseBlock(
                input_dim=input_size,
                layer_dim=layer_dim,
                num_layers=num_layers,
                tree_dim=tree_dim,
                depth=depth,
                flatten_output=False,
                choice_function=lib.entmax15,
                bin_function=lib.entmax15,
            ),
            lib.Lambda(lambda x: x[..., 0].mean(dim=-1)),
        )

    def forward(self, x, noise=None):
        result = torch.cat(
            [
                self.dlle(x).unsqueeze(1),
                self.dllk(x).unsqueeze(1),
                self.dllmu(x).unsqueeze(1),
                self.dllp(x).unsqueeze(1),
                self.dllbt(x).unsqueeze(1),
            ],
            axis=1,
        )
        return result


def create_node_model(config: tp.Dict[str, tp.Any], model_type: str = "classification"):
    if model_type == "classification":
        model = ClassificationDifferentialTree(**config)
    elif model_type == "regression":
        model = RegressionDifferentialTree(**config)
    else:
        raise NameError("Unknown model type: {}".format(model_type))
    return model