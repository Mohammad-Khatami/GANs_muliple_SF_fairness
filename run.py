import os.path
import pandas as pd
import numpy as np
import pytz
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from AAE_model import AdversarialAutoEncoder
from datetime import datetime
from utils import get_config_from_args 

import time
#from train_test_functions import train_model_DP, test_model_DP, train_model_EO, test_model_EO
from algorithm import run_algorithm

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    start_time = time.time()
    config = get_config_from_args()

    # used for logging purposes
    # Get the current time in the Toronto timezone
    toronto_tz = pytz.timezone("America/Toronto")
    execution_start_time: str = f"{datetime.now(tz=toronto_tz):%Y-%m-%d-%H:%M}"
    config["execution_start_time"] = execution_start_time

    device = torch.device("cpu")
    if config["mode"] == "GPU":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")

    (
        train_loss_all,
        valid_loss_all,
        train_acc_all,
        valid_acc_all,
        valid_disc_gender_all,
        valid_disc_race_all,
    ) = run_algorithm(config, device)
    print(torch.__version__)
    print("--- %s seconds ---" % (time.time() - start_time))

