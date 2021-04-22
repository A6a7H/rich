import os
import torch
import numpy as np
import pandas as pd

from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, QuantileTransformer, StandardScaler
from torch.utils.data import TensorDataset


def scale_pandas(dataframe, scaler):
    return pd.DataFrame(scaler.transform(dataframe.values), 
                        columns=dataframe.columns)

class RichDatasetLoader:
    def __init__(
        self,
        data_dir,
        weight_col="probe_sWeight",
        test_size=0.5,
        list_particles=["kaon", "pion", "proton", "muon"],
    ):
        self.list_particles = list_particles
        self.data_dir = data_dir
        self.test_size = test_size
        self.datasets = {
            particle: self.get_particle_dataset(particle) for particle in list_particles
        }

        self.dll_columns = [
            "RichDLLe",
            "RichDLLk",
            "RichDLLmu",
            "RichDLLp",
            "RichDLLbt",
        ]
        self.raw_feature_columns = ["Brunel_P", "Brunel_ETA", "nTracks_Brunel"]
        self.weight_col = "probe_sWeight"

        self.y_count = len(self.dll_columns)

    def get_particle_dataset(self, particle):
        return [
            self.data_dir + name
            for name in os.listdir(self.data_dir)
            if particle in name
        ]

    def split(self, data):
        data_train, data_val = train_test_split(
            data, test_size=self.test_size, random_state=42
        )
        data_val, data_test = train_test_split(
            data_val, test_size=self.test_size, random_state=1812
        )
        return (
            data_train.reset_index(drop=True),
            data_val.reset_index(drop=True),
            data_test.reset_index(drop=True),
        )

    def get_all_particles_dataset(self, dtype=None, log=False, n_quantiles=100000):
        data_train_all = []
        data_val_all = []
        scaler_all = {}
        for index, particle in enumerate(self.list_particles):
            data_train, data_val, scaler = self.get_merged_typed_dataset(
                particle, dtype=dtype, log=log, n_quantiles=n_quantiles
            )
            ohe_table = pd.DataFrame(
                np.zeros((len(data_train), len(self.list_particles))),
                columns=["is_{}".format(i) for i in self.list_particles],
            )
            ohe_table["is_{}".format(particle)] = 1

            data_train_all.append(
                pd.concat(
                    [
                        data_train.iloc[:, : self.y_count],
                        ohe_table,
                        data_train.iloc[:, self.y_count :],
                    ],
                    axis=1,
                )
            )

            data_val_all.append(
                pd.concat(
                    [
                        data_val.iloc[:, : self.y_count],
                        ohe_table[: len(data_val)].copy(),
                        data_val.iloc[:, self.y_count :],
                    ],
                    axis=1,
                )
            )
            scaler_all[index] = scaler
        data_train_all = pd.concat(data_train_all, axis=0).astype(dtype, copy=False)
        data_val_all = pd.concat(data_val_all, axis=0).astype(dtype, copy=False)
        return data_train_all, data_val_all, scaler_all

    def get_merged_typed_dataset(
        self, particle_type, dtype=None, log=False, n_quantiles=100000
    ):
        file_list = self.datasets[particle_type]
        if log:
            print("Reading and concatenating datasets:")
            for fname in file_list:
                print("\t{}".format(fname))
        data_full = self.load_and_merge_and_cut(file_list)
        # Must split the whole to preserve train/test split""
        if log:
            print("splitting to train/val/test")
        data_train, data_val, _ = self.split(data_full)
        if log:
            print("fitting the scaler")
        print("scaler train sample size: {}".format(len(data_train)))
        start_time = time()
        if n_quantiles == 0:
            scaler = StandardScaler().fit(
                data_train.drop(self.weight_col, axis=1).values
            )
        else:
            scaler = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=n_quantiles,
                subsample=int(1e10),
            ).fit(data_train.drop(self.weight_col, axis=1).values)
        print(
            "scaler n_quantiles: {}, time = {}".format(n_quantiles, time() - start_time)
        )
        if log:
            print("scaling train set")
        data_train = pd.concat(
            [
                scale_pandas(data_train.drop(self.weight_col, axis=1), scaler),
                data_train[self.weight_col],
            ],
            axis=1,
        )
        if log:
            print("scaling test set")
        data_val = pd.concat(
            [
                scale_pandas(data_val.drop(self.weight_col, axis=1), scaler),
                data_val[self.weight_col],
            ],
            axis=1,
        )
        if dtype is not None:
            if log:
                print("converting dtype to {}".format(dtype))
            data_train = data_train.astype(dtype, copy=False)
            data_val = data_val.astype(dtype, copy=False)
        return data_train, data_val, scaler

    def load_and_cut(self, file_name):
        data = pd.read_csv(file_name, delimiter="\t")
        return data[self.dll_columns + self.raw_feature_columns + [self.weight_col]]

    def load_and_merge_and_cut(self, filename_list):
        return pd.concat(
            [self.load_and_cut(fname) for fname in filename_list],
            axis=0,
            ignore_index=True,
        )


def get_RICH(particle, drop_weights, path):
    flow_shape = (5,)

    train_data, test_data, scaler = RichDatasetLoader(path).get_merged_typed_dataset(
        particle, dtype=np.float32, log=True
    )

    condition_columns = ["Brunel_P", "Brunel_ETA", "nTracks_Brunel"]
    signal_columns = ['is_Kaon', 'is_Muon', 'is_Pion', 'is_Proton']
    dll_columns = ["RichDLLe", "RichDLLk", "RichDLLmu", "RichDLLp", "RichDLLbt"]
    weight_column = "probe_sWeight"

    if drop_weights:
        train_dataset, test_dataset = pd.read_csv(f"{path}/MCRICH_train.csv"), pd.read_csv(f"{path}/MCRICH_test.csv")

        drop_signals=['Electron', 'Ghost']

        train_dataset = train_dataset.loc[~train_dataset.Signal.isin(drop_signals)]
        train_dataset = pd.get_dummies(train_dataset.Signal, prefix='is').join(train_dataset)
        train_dataset.drop('Signal', axis=1, inplace=True)

        test_dataset = test_dataset.loc[~test_dataset.Signal.isin(drop_signals)]
        test_dataset = pd.get_dummies(test_dataset.Signal, prefix='is').join(test_dataset)
        test_dataset.drop('Signal', axis=1, inplace=True)

        scaler = StandardScaler().fit(train_dataset.drop(signal_columns, axis=1).values)

        onehot_train = train_dataset[signal_columns].reset_index(drop=True)
        train_dataset = pd.concat([onehot_train, scale_pandas(train_dataset.drop(signal_columns, axis=1), scaler)], axis=1)

        onehot_val = test_dataset[signal_columns].reset_index(drop=True)
        test_dataset = pd.concat([onehot_val, scale_pandas(test_dataset.drop(signal_columns, axis=1), scaler)], axis=1)
    else:
        train_dataset = TensorDataset(
            torch.from_numpy(train_data[condition_columns].values),
            torch.from_numpy(train_data[dll_columns].values),
            torch.from_numpy(train_data[weight_column].values),
        )
        test_dataset = TensorDataset(
            torch.from_numpy(test_data[condition_columns].values),
            torch.from_numpy(test_data[dll_columns].values),
            torch.from_numpy(test_data[weight_column].values),
        )

    return len(condition_columns), flow_shape, train_dataset, test_dataset, scaler
