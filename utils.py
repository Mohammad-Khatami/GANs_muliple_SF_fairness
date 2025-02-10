from __future__ import annotations
import argparse
from typing import Literal, TYPE_CHECKING
from shutil import move
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import os.path
import torch

if TYPE_CHECKING:
    from torch.nn import Module


def mv_file(src_path: str, dest_path: str):
    try:
        if os.path.isfile(dest_path):
            move (dest_path, "#"+dest_path+".previos_run_resullts")
        move(src_path, dest_path)
        print(f"File moved successfully from {src_path} to {dest_path}.")
    except Exception as e:
        print(f"Error moving file: {e}")


def generate_filename(
    config: dict[str, int | str],
    stack: int | str,
    output_source: Literal["train", "valid", "eval", "model"],
    discrimination_metric: Literal["DP", "EO"],
    trailing_text: str = "",
    file_extension=".csv",
) -> str:
    """
    Note that while output_source can technically be any valid str, the intended values
    are only "train", "valid", "eval", "model"
    """
    # make sure there's no extra underscore
    if trailing_text:
        _separator = "_"
    else:
        _separator=""
    return (
        f'st_{stack}_e_{config["epoch_num"]}_num_layers_{config["num_layers"]}'
        f'_beta_{config["beta"]}_rn_{config["seed"]}_lr_{config["lr"]}_batch_{config["batch_size"]}_p_{config["dropout_rate"]}'
        f"_{output_source}_{discrimination_metric}{_separator}{trailing_text}{file_extension}"
        # f"_{output_source}_{discrimination_metric}_{trailing_text}.csv"
    )

def save_model_state(filename: str, model: Module):
    try:
        model_state_dict = model.state_dict()
        torch.save(model_state_dict, filename)
        print(f"Model state dictionary saved successfully to {filename}.")
    except Exception as e:
        print(f"Error saving model state dictionary: {e}")


def accuracy_calculator_per_batch(predicted_label, label):
    predicted = (predicted_label > 0.5).float()  # Threshold predictions
    accuracy = (predicted.squeeze() == label).float().mean().item()
    return accuracy


def make_data_loader_path_multiple(
    path, device: torch.device = torch.device("cpu"), batch_size=32, seed=42
):
    train_path = path + "data"
    test_path = path + "test"
    # train_path = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    # test_path = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'

    # Labels in the dataset
    names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income>50k",
    ]

    train_df = pd.read_csv(train_path, names=names, index_col=False, comment="|")
    test_df = pd.read_csv(test_path, names=names, index_col=False, comment="|")
    all_df = pd.concat([train_df, test_df])
    all_df[all_df == "?"] = np.nan
    all_df[all_df == " ?"] = np.nan
    all_df.dropna(inplace=True)

    all_df["income>50k"] = (
        all_df["income>50k"].str.strip().str.replace(".", "", regex=False)
    )

    all_inputs = pd.get_dummies(
        all_df.drop(["income>50k", "education-num", "fnlwgt"], axis=1),
        #all_df.drop(["income>50k", "education-num", "fnlwgt", "sex", "race"], axis=1),
        dtype="float",
    )

    all_labels, _ = all_df["income>50k"].factorize()
    all_sensitive_gender, _ = all_df["sex"].factorize()  # male = 0, female = 1
    all_sensitive_race = (
        (all_df["race"].str.strip() == "White").astype(int).values
    )  # white = 1, other = 0

    # Combine `race` and `sex` into a single categorical variable
    all_sensitive_combined = all_sensitive_race * 2 + all_sensitive_gender  # Produces 0-> NW-M, 1->W-M, 2->NW-F, or 3->W-F

    print (all_sensitive_combined.shape)

    all_inputs = F.normalize(torch.Tensor(all_inputs.values))
    all_labels = torch.Tensor(all_labels)
    all_sensitive_combined = torch.Tensor(all_sensitive_combined)  # Combined sensitive attribute

    train_size = np.around(35000 / len(all_sensitive_combined), decimals=5)
    TEST_size = 1.0 - train_size

    train_inputs, temp_inputs, train_labels, temp_labels = train_test_split(
        all_inputs, all_labels, test_size=TEST_size, random_state=seed, shuffle=True
    )
    train_sensitive_combined, temp_sensitive_combined = train_test_split(
        all_sensitive_combined, test_size=TEST_size, random_state=seed, shuffle=True
    )

    valid_inputs, test_inputs, valid_labels, test_labels = train_test_split(
        temp_inputs, temp_labels, test_size=0.5, random_state=seed, shuffle=False
    )
    valid_sensitive_combined, test_sensitive_combined = train_test_split(
        temp_sensitive_combined, test_size=0.5, random_state=seed, shuffle=False
    )

    # Create a dataset object that pairs the input samples and target labels
    train_dataset = TensorDataset(train_inputs, train_labels, train_sensitive_combined)
    valid_dataset = TensorDataset(valid_inputs, valid_labels, valid_sensitive_combined)
    test_dataset = TensorDataset(test_inputs, test_labels, test_sensitive_combined)

    # Dataloaders for training and testing
    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    validloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    testloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    return trainloader, validloader, testloader





def get_config_from_args() -> dict[str, float]:
    parser = argparse.ArgumentParser(
        description="""
Parse configuration for the model.

Sample usage: python your_script.py --num_layers 3 --hidden_dim 200 --alpha 0.1 --beta 30 --gamma 2 --epoch_num 20 --seed 123 --learning_rate 0.001"""
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Number of layers for every stack",
    )
    parser.add_argument("-a", "--alpha", type=float, default=0, help="Alpha value")
    parser.add_argument(
        "--beta", type=float, default=50, help="Beta value"
    )
    parser.add_argument("-g", "--gamma", type=float, default=1, help="Gamma value")
    parser.add_argument(
        "--epoch_num_level_1",
        type=int,
        default=10,
        help="[Integer] number of epochs for level 1",
    )
    parser.add_argument(
        "--epoch_num_level_2",
        type=int,
        default=10,
        help="[Integer] number of epochs for level 2",
    )
    parser.add_argument("--seed", type=int, default=42, help="[Integer] random seed")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for both models"
    )
    parser.add_argument(
        "--lr", "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument(
        "--discrimination_metric",
        type=str,
        choices=["DP", "EO"],
        default="DP",
    )
    parser.add_argument(
        "--input_dim", type=int, default=102, help="Input dimensions for level 1"
    )

    parser.add_argument("--hidden_dim_level_1", type=int, default=60)
    parser.add_argument("--encoded_dim_level_1", type=int, default=40)
    parser.add_argument("--hidden_dim_level_2", type=int, default=20)
    parser.add_argument("--encoded_dim_level_2", type=int, default=10)
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.1,
        help="Dropout percentage used for regularization in the range [0.0, 1.0]",
    )

    parser.add_argument(
        "--mode",
        type=str.upper,
        choices=["CPU", "GPU"],
        default="CPU",
        help="Specifies what device to train on",
    )

    parser.add_argument(
        "--termination_epoch_threshold",
        type=int,
        help="Specifies how many epochs to have without accuracy improvement before terminating the training prematurely",
    )
    parser.add_argument(
        "--margin_threshold",
        type=float,
        # default=0.0,
        help="margin threshold to activate the thermination [0.0 1.0]. 0.01 means 1 percent ",
    )

    args = parser.parse_args()

    return vars(args)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # random.seed(seed)

def encode_input(
    trained_model: AdversarialAutoEncoder,
    dataloader: DataLoader, 
    device: torch.device = torch.device("cpu"),
    batch_size: int = 32,
    Shuffle: bool = True,
) -> DataLoader:
    """A helper function that creates a new Dataloader where
    the first tensor is encoded with the trained_model's encoder.
    All other tensors are left unchanged. 
    Args:
        trained_model (AdversarialAutoEncoder): trained AdversarialAutoEncoder model
        dataloader (DataLoader): a loader to extract data from. Must have 4 tensors, where
          the first tensor is to be encoded
        device (torch.device, optional): device to place new tensors on. Defaults to "cpu"
        batch_size (int, optional): New Dataloader's batch size. Defaults to 32.
    
    Returns:
        DataLoader: a Dataloader with the first dimension encoded by trained_model
    """
    encoded_inputs_list, labels_list, protected_gender_race_list = (
        [],
        [],
        [],
    )
    with torch.inference_mode():
        for inputs, labels, protected_gender_race in dataloader:
            # print(f"{inputs=}\n{labels=}\n{protected_gender=}\n{protected_race=}")
            encoded_inputs_list.append(trained_model.encoder(inputs))
            labels_list.append(labels)
            protected_gender_race_list.append(protected_gender_race)

    new_dataset = TensorDataset(
        torch.concat(encoded_inputs_list),
        torch.concat(labels_list),
        torch.concat(protected_gender_race_list),
    )
    if Shuffle:
        return DataLoader(new_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        return DataLoader(new_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


