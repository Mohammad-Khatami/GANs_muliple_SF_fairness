import os.path
from shutil import move
import pandas as pd
import numpy as np
import pytz
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from AAE_model import AdversarialAutoEncoder #,CustomFarthestLoss 
from datetime import datetime
from utils import (generate_filename,
                get_config_from_args, 
                save_model_state,
                make_data_loader_path_multiple,encode_input ,set_seed)
import time
from train_test_functions import train_model, test_model_DP, test_model_EO, select_best_model

device = "cuda" if torch.cuda.is_available() else "cpu"

def run_algorithm(config: dict, device: torch.device = torch.device("cpu")):
    if config["discrimination_metric"] == "DP":
        _test_function=test_model_DP
    elif config["discrimination_metric"] == "EO":
        _test_function=test_model_EO
    else:
        raise NotImplementedError(
            f"Discrimination Metric {config['discrimination_metric']!r} is not supported"
        )
    url = "./adult."

##----------------------------Data loader-1 ---------------------------------------#

    train_loader_level_1, validation_loader_level_1, test_loader_level_1 = make_data_loader_path_multiple(
            path=url,
            batch_size=config["batch_size"],
            device=device,
            seed=config["seed"],
        )
####################################################################################
##----------------------------define model stack-1  ---------------------------------------#
####################################################################################
##----------------------------train/valid stack-1  ---------------------------------------#

    train_loss_all, valid_loss_all, train_acc_all, valid_acc_all, valid_disc_gender_all, valid_disc_race_all, selected_moldels_1 = train_model(
        train_loader_level_1,
        validation_loader_level_1,
        batch=config["batch_size"],
        metric_type=config["discrimination_metric"],
        alpha=config["alpha"],
        beta=config["beta"],
        gamma=config["gamma"],
        epoch_num=config["epoch_num_level_1"],
        lr=config["lr"],
        log_filename_tail="temp",
        termination_epoch_threshold=config.get("termination_epoch_threshold"),
        margin_threshold=config.get("margin_threshold"),
        config=config,
        stack_number=1,
        # num_layers=config["num_layers"],
    )

    best_model_1,best_model_epoch_1,best_model_rn_1 =select_best_model(selected_moldels_1)    
    print(f"stack: {1}, the epoch chosen from the list: {best_model_epoch_1}, the random number chosen: {best_model_rn_1}")  

    config_new_1=config
    config_new_1.update({'epoch_num': best_model_epoch_1, 'seed': best_model_rn_1})


    model_filename = generate_filename(
        config=dict(**config_new_1),
        stack=1,
        output_source="model",
        discrimination_metric=config["discrimination_metric"],
        file_extension=".pth",
    )

    if os.path.isfile(model_filename):
        move (model_filename, "#"+model_filename+".previos_run_resullts")
    
    save_model_state(model_filename, best_model_1)
##----------------------------test stack-1  ---------------------------------------#

    test_filename = generate_filename(
        config=dict(**config_new_1),
        stack=1,
        output_source="test",
        discrimination_metric=config["discrimination_metric"],
        file_extension=".csv",
    )
    if os.path.isfile(test_filename):
        move (test_filename, "#"+test_filename+".previos_run_resullts")
     
    _test_function(
        best_model_1,
        test_loader_level_1,
        criterion_bce= nn.BCELoss(),
        criterion_mse = nn.MSELoss(),
        batch=config["batch_size"],
        alpha=config["alpha"],
        beta=config["beta"],
        gamma=config["gamma"],
        log_filename=test_filename,
    )
####################################################################################
##----------------------------define model stack-2  ---------------------------------------#
####################################################################################
##----------------------------Data loader-2 ---------------------------------------#

    train_loader_level_2 = encode_input(
        best_model_1, train_loader_level_1, batch_size=config["batch_size"], device=device, Shuffle=True
    )
    validation_loader_level_2 = encode_input(
        best_model_1, validation_loader_level_1, batch_size=config["batch_size"], device=device, Shuffle=False
    )
    test_loader_level_2 = encode_input(
        best_model_1, test_loader_level_1, batch_size=config["batch_size"], device=device, Shuffle=False
    )
##----------------------------train/valid stack-2  ---------------------------------------#
    train_loss_all, valid_loss_all, train_acc_all, valid_acc_all, valid_disc_gender_all, valid_disc_race_all, selected_moldels_2 = train_model(
        #model_level_2,
        train_loader_level_2,
        validation_loader_level_2,
        batch=config["batch_size"],
        metric_type=config["discrimination_metric"],
        alpha=config["alpha"],
        beta=config["beta"],
        gamma=config["gamma"],
        epoch_num=config["epoch_num_level_2"],
        lr=config["lr"],
        log_filename_tail="temp",
        termination_epoch_threshold=config.get("termination_epoch_threshold"),
        margin_threshold=config.get("margin_threshold"),
        config=config,
        stack_number=2,
    )
    best_model_2,best_model_epoch_2,best_model_rn_2 =select_best_model(selected_moldels_2)
    print(f"stack: {2}, the epoch chosen from the list: {best_model_epoch_2}, the random number chosen: {best_model_rn_2}")

    config_new_2=config
    config_new_2.update({'epoch_num': best_model_epoch_2, 'seed': best_model_rn_2})
    model_filename = generate_filename(
        config=dict(**config_new_2),
        stack=2,
        output_source="model",
        discrimination_metric=config["discrimination_metric"],
        file_extension=".pth",
    )

    if os.path.isfile(model_filename):
        move (model_filename, "#"+model_filename+".previos_run_resullts")

    save_model_state(model_filename, best_model_2)
##----------------------------test stack-2  ---------------------------------------#

    test_filename = generate_filename(
        config=dict(**config_new_2),
        stack=2,
        output_source="test",
        discrimination_metric=config["discrimination_metric"],
        file_extension=".csv",
    )

    if os.path.isfile(test_filename):
        move (test_filename, "#"+test_filename+".previos_run_resullts")

    _test_function(
        best_model_2,
        test_loader_level_2,
        criterion_bce= nn.BCELoss(),
        criterion_mse = nn.MSELoss(),
        batch=config["batch_size"],
        alpha=config["alpha"],
        beta=config["beta"],
        gamma=config["gamma"],
        log_filename=test_filename,
    )

    return (
        train_loss_all,
        valid_loss_all,
        train_acc_all,
        valid_acc_all,
        valid_disc_gender_all,
        valid_disc_race_all,
    )


