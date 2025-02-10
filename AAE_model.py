from __future__ import annotations

# the bottom line avoids expensive import during runtime
# but still lets type checker resolve things
from typing import TYPE_CHECKING
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

torch.autograd.set_detect_anomaly(True)

if TYPE_CHECKING:
    from torch import Tensor

class AdversarialAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        encoded_dim: int,
        num_layers: int,
        dropout_rate: float = 0.1,
        stack_number: int = 1,
    ):
        super(AdversarialAutoEncoder, self).__init__()
        self.stack_number = stack_number
#######encoder
        inp_dim=input_dim
        hid_dim=hidden_dim
        lat_dim=encoded_dim
        self.layers_encoder = []
        for i in range(num_layers):
          self.layers_encoder.append(nn.Linear(inp_dim, hid_dim))
          self.layers_encoder.append(nn.Dropout(p=dropout_rate))
          self.layers_encoder.append(nn.ReLU())
          inp_dim=hid_dim
        if num_layers==1:
          hid_dim=inp_dim

        self.layers_encoder.append(nn.Linear(hid_dim, lat_dim))
        self.layers_encoder.append(nn.ReLU())
        self.encoder= nn.Sequential(*self.layers_encoder)
#######decoder
        inp_dim=input_dim
        hid_dim=hidden_dim
        lat_dim=encoded_dim
        self.layers_decoder = []
        for i in range(num_layers):
          self.layers_decoder.append(nn.Linear(lat_dim, hid_dim))
          self.layers_decoder.append(nn.Dropout(p=dropout_rate))
          self.layers_decoder.append(nn.ReLU())
          lat_dim=hid_dim
        if num_layers==1:
          hid_dim=lat_dim

        self.layers_decoder.append(nn.Linear(hid_dim, inp_dim))
        self.layers_decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*self.layers_decoder)
#######adversary
        inp_dim=input_dim
        hid_dim=hidden_dim
        lat_dim=encoded_dim
        self.layers_adv = []
        for i in range(num_layers):
          self.layers_adv.append(nn.Linear(lat_dim, hid_dim))
          self.layers_adv.append(nn.Dropout(p=dropout_rate))
          self.layers_adv.append(nn.ReLU())
          lat_dim=hid_dim
        if num_layers==1:
          hid_dim=lat_dim

        self.layers_adv.append(nn.Linear(hid_dim, 4))
        self.adv_multi = nn.Sequential(*self.layers_adv)
#######predic
        inp_dim=input_dim
        hid_dim=hidden_dim
        lat_dim=encoded_dim
        self.layers_predictor = []
        for i in range(num_layers):
          self.layers_predictor.append(nn.Linear(lat_dim, hid_dim))
          self.layers_predictor.append(nn.Dropout(p=dropout_rate))
          self.layers_predictor.append(nn.ReLU())
          lat_dim=hid_dim
        if num_layers==1:
          hid_dim=lat_dim

        self.layers_predictor.append(nn.Linear(hid_dim, 1))
        self.layers_predictor.append(nn.Sigmoid())
        self.predictor = nn.Sequential(*self.layers_predictor)
        

    def forward(self, x: Tensor):
        NN_encoder_output: Tensor = self.encoder(x)
        NN_decoder_output: Tensor = self.decoder(NN_encoder_output)
        NN_predict_output: Tensor = self.predictor(NN_encoder_output)
        NN_adversary_multi_output: Tensor = self.adv_multi(NN_encoder_output)
        return (
            NN_decoder_output,
            NN_predict_output,
            NN_adversary_multi_output,
        )


def params_optims(ML_model: AdversarialAutoEncoder, lr: float):
    encoder_params = list(ML_model.encoder.parameters())
    decoder_params = list(ML_model.decoder.parameters())
    predictor_params = list(ML_model.predictor.parameters())
    adversary_multi_params = list(ML_model.adv_multi.parameters())

    #optimizer_encoder_decoder_predictor = torch.optim.Adam(encoder_params+decoder_params+predictor_params, lr=lr)
    #optimizer_encoder_adv_multi = torch.optim.Adam(encoder_params+adversary_multi_params, lr=lr)
    optimizer_encoder= torch.optim.Adam(encoder_params, lr=lr)
    optimizer_decoder= torch.optim.Adam(decoder_params, lr=lr)
    optimizer_predictor= torch.optim.Adam(predictor_params, lr=lr)
    optimizer_adv_multi = torch.optim.Adam(adversary_multi_params, lr=lr)

    # optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    return (
    optimizer_encoder,
    optimizer_decoder,
    optimizer_predictor,
    optimizer_adv_multi,
    #optimizer_encoder_decoder_predictor,
    #optimizer_encoder_adv_multi,
    )

