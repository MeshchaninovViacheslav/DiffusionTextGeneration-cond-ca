import torch
import numpy as np
from transformers import BertConfig, BertModel, BertLMHeadModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from bert_model.score_estimator import ScoreEstimatorEMB
from diffusion_utils.diffusion_dynamic_sde import create_sde


class Buffer(torch.nn.Module):
    def __init__(self):
        super(Buffer, self).__init__()

        self.size = 0
        self.buffer = []
        self.max_size = 10

    def push(self, value):
        if self.size == self.max_size:
            self.pop()
        self.buffer.append(value)
        self.size += 1

    def pop(self):
        del self.buffer[0]
        self.size -= 1

    def get(self):
        return np.mean(self.buffer)

    # def state_dict(self):
    #     return dict(
    #         size=self.size,
    #         buffer=self.buffer,
    #         max_size=self.max_size
    #     )
    #
    # def load_state_dict(self, state_dict):
    #     self.size = state_dict["size"]
    #     self.buffer = state_dict["buffer"]
    #     self.max_size = state_dict["max_size"]


class End2EndDLM(torch.nn.Module):
    def __init__(self, config, bert_config):
        super(End2EndDLM, self).__init__()

        self.bert_config = bert_config
        self.bert = BertLMHeadModel.from_pretrained("bert-base-uncased")
        self.encoder = self.bert.bert
        self.decoder = self.bert.cls
        self.diffusion = ScoreEstimatorEMB(self.bert_config.hidden_size, self.bert_config)
        self.sde = create_sde(config=config)

        # self.mu_store = torch.nn.Parameter(data=torch.empty((bert_config.hidden_size)), requires_grad=False)
        # self.std_store = torch.nn.Parameter(data=torch.empty((bert_config.hidden_size)), requires_grad=False)
        self.mu = None
        self.std = None

    def encode(self, X):
        return self.encoder(**X).last_hidden_state

    def decode(self, z):
        return self.decoder(z)

    def forward(self, X, t):
        latent = self.encode(X)

        # diffusion
        clean_x = self.normalize(latent)
        mask = X["attention_mask"]
        x_t = self.sde.marginal_forward(clean_x, t)['x_t']
        x_0 = self.diffusion(x_t=x_t, time_t=t, attention_mask=mask)

        recon_x_0 = self.decode(self.denormalize(x_0))
        recon = self.decode(latent)
        return dict(
            latent=latent,
            clean_x=clean_x,
            x_0=x_0,
            recon_x_0=recon_x_0,
            recon=recon,
        )

    def normalize(self, z):
        # if not self.training or (self.mu is None or self.std is None):
        #     self.mu = self.mu_store
        #     self.std = self.std_store
        if self.training:
            # self.mu_store.data = torch.mean(z, dim=[0, 1])
            # self.std_store.data = torch.std(z, dim=[0, 1])

            self.mu = torch.mean(z, dim=[0, 1])
            self.std = torch.std(z, dim=[0, 1])

        return (z - self.mu) / self.std

    def denormalize(self, z):
        # if not self.training or (self.mu is None or self.std is None):
        #     self.mu = self.mu_store
        #     self.std = self.std_store
        return z * self.std + self.mu

    def predict_x0(self,
                   x_t: torch.Tensor,
                   time_t,
                   mask):
        return self.diffusion(x_t, time_t, attention_mask=mask)