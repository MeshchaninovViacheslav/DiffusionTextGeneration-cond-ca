import os
import torch
import wandb
import numpy as np
import pandas as pd
import torch.distributed as dist
from ml_collections import ConfigDict
from collections import defaultdict
from typing import Optional, Union, Dict
from tqdm.auto import trange
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from tqdm import tqdm
from torch.nn.functional import cross_entropy, normalize
from copy import deepcopy
from timm.scheduler.cosine_lr import CosineLRScheduler
from collections import defaultdict
from torch.cuda.amp import GradScaler
from time import time

from diffusion_utils.diffusion_dynamic_sde import create_sde, create_solver
from utils.ema_model import ExponentialMovingAverage
from bert_model.modelling_bert import BertLMHeadModel
from bert_model.score_estimator_cond import ScoreEstimatorEMB
from utils.util import dict_to_cuda, reduce_tensor, masked_mean, \
    masked_std, make_mask_wo_SEP_CLS, set_seed
from data.dataset import create_dataset

from estimation_utils.util import estimate_model, gather_texts, reduce_metrics
from estimation_utils.metrics import BloomMetric, GPTMetric


class Loss_ema_tracker:
    def __init__(self):
        self.alpha = 0.001
        self.num_step_to_fill = 100
        self._loss = 0.
        self.num_steps = 0

    def update(self, loss):
        self.num_steps += 1
        if self.num_steps < self.num_step_to_fill:
            self._loss = (self._loss * (self.num_steps - 1) + loss) / self.num_steps
        else:
            self._loss = self._loss * (1 - self.alpha) + loss * self.alpha

    @property
    def loss(self):
        return self._loss


class DiffusionRunner:
    def __init__(
            self,
            config: ConfigDict,
            eval: bool = False,
            latent_mode: str = "embeddings"
    ):
        self.config = config
        self.latent_mode = latent_mode
        self.eval = eval

        if self.config.model.enc_type == "base":
            pretrained_enc_type = "bert-base-uncased"
        elif self.config.model.enc_type == "large":
            pretrained_enc_type = "bert-large-uncased"
        else:
            raise Exception("Unknown embedding type")

        self.checkpoints_folder = config.training.checkpoints_folder
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_enc_type)
        self.encoder = BertLMHeadModel.from_pretrained(pretrained_enc_type).cuda().eval()
        self.decoder = self.encoder.cls.cpu()
        self.restore_decoder()
        self.decoder = self.decoder.cuda().eval()

        # self.load_sde()
        self.bert_config = config.bert_config
        self.score_estimator = ScoreEstimatorEMB(input_size=self.encoder.config.hidden_size,
                                                 config=self.bert_config).cuda()
        self.ddp_score_estimator = self.score_estimator
        if self.config.ddp:
            self.ddp_score_estimator = torch.nn.parallel.DistributedDataParallel(
                self.score_estimator,
                device_ids=[config.local_rank],
                broadcast_buffers=False,
            )
        self.total_number_params = sum(p.numel() for p in self.score_estimator.parameters() if p.requires_grad)
        self.config.model.total_number_params = self.total_number_params
        self.device = next(self.score_estimator.parameters()).device

        self.sde = create_sde(config=config)
        self.diff_eq_solver = create_solver(config, self.sde, ode_sampling=config.sde.ode_sampling)

        if eval:
            self.ema = ExponentialMovingAverage(self.score_estimator.parameters(), config.model.ema_rate)
            self.restore_parameters(self.device)
            self.switch_to_ema()
            self.score_estimator.eval()

        self.grad_expl_dict = defaultdict(list)

        self.train_datasets_iter = create_dataset(dataset_name=config.model.dataset,
                                                  downstream_task=config.model.downstream_task)(
            split="train",
            tokenizer=self.tokenizer,
            max_sequence_len=self.config.data.max_sequence_len,
            p_uncond=0
        ).get_data()
        self.train_dataset = None

        self.valid_datasets_iter = create_dataset(dataset_name=config.model.dataset,
                                                  downstream_task=config.model.downstream_task)(
            split="test",
            tokenizer=self.tokenizer,
            max_sequence_len=self.config.data.max_sequence_len,
            p_uncond=0
        ).get_data()
        self.valid_dataset = next(self.valid_datasets_iter)

    @torch.no_grad()
    def sampler_emb_impl(self, X):
        if self.latent_mode == "embeddings":
            return self.encoder.bert.embeddings.get_word_emb(
                **self.get_dict_to_emb(X))  # get input_ids [batch_size; num_tokens; emb_dim]
        elif self.latent_mode == "encodings":
            return self.encoder(**X, return_encoding=True)
        elif self.latent_mode == "mid_embeddings":
            return self.encoder(**X, return_mid_embs=True)
        else:
            raise Exception("Unknown latent mode")

    @torch.no_grad()
    def get_mean_std_data(self):
        suffix = "wikipedia"  # self.config.model.dataset

        if self.latent_mode == "embeddings":
            if not hasattr(self, 'emb_mean'):
                self.emb_mean = torch.load(
                    f"/home/vmeshchaninov/DiffusionTextGeneration/data/{self.config.model.embeddings_type}"
                    f"-bert_base-{suffix}-mean.pt")
                self.emb_std = torch.load(
                    f"/home/vmeshchaninov/DiffusionTextGeneration/data/{self.config.model.embeddings_type}"
                    f"-bert_base-{suffix}-std.pt")

            return self.emb_mean[None, None, :], self.emb_std[None, None, :]
        if self.latent_mode == "encodings":
            if not hasattr(self, 'emb_mean'):
                if self.config.model.enc_type == "base":
                    self.emb_mean = torch.load(
                        f"/home/vmeshchaninov/DiffusionTextGeneration/data/encodings-bert_base-{suffix}-mean.pt")
                    self.emb_std = torch.load(
                        f"/home/vmeshchaninov/DiffusionTextGeneration/data/encodings-bert_base-{suffix}-std.pt")
                elif self.config.model.enc_type == "large":
                    self.emb_mean = torch.load(
                        f"/home/vmeshchaninov/DiffusionTextGeneration/data/encodings-bert_large-mean.pt")
                    self.emb_std = torch.load(
                        f"/home/vmeshchaninov/DiffusionTextGeneration/data/encodings-bert_large-std.pt")
                else:
                    raise Exception("Unknown embedding type")
            return self.emb_mean[None, None, :], self.emb_std[None, None, :]
        if self.latent_mode == "mid_embeddings":
            if not hasattr(self, 'emb_mean'):
                self.emb_mean = torch.load("/home/vmeshchaninov/DiffusionTextGeneration/data/mid_embeddings_mean.pt")
                self.emb_std = torch.load("/home/vmeshchaninov/DiffusionTextGeneration/data/mid_embeddings_std.pt")
            return self.emb_mean[None, None, :], self.emb_std[None, None, :]
        raise Exception("Not implemented yet")

    @torch.no_grad()
    def sampler_emb(self, X):
        codes = self.sampler_emb_impl(X)
        # [-1; 1]
        mean, std = self.get_mean_std_data()
        device = codes.device
        mean = mean.to(device)
        std = std.to(device)
        norm_codes = (codes - mean) / std
        return norm_codes

    def denormalize(self, X):
        mean, std = self.get_mean_std_data()
        device = X.device
        mean = mean.to(device)
        std = std.to(device)
        return X * std + mean

    def restore_parameters(self, device: Optional[torch.device] = None) -> None:
        checkpoints_folder: str = self.checkpoints_folder
        if device is None:
            device = torch.device('cpu')
        prefix = ''
        if self.config.checkpoints_prefix:
            prefix = self.config.checkpoints_prefix
        ema_ckpt = torch.load(checkpoints_folder + '/' + prefix + '.pth')["ema"]
        # ema_ckpt = torch.load(checkpoints_folder + '/' + prefix + '.pth', map_location=device)
        self.ema.load_state_dict(ema_ckpt)

    def restore_decoder(self):
        decoder_path = ""
        if self.config.model.dataset == "rocstory":
            decoder_path = "decoder-v1.1-pad.pth"
        elif self.config.model.dataset == "c4":
            decoder_path = "decoder-c4-pad-64.pth"
        elif self.config.model.dataset == "wikipedia":
            decoder_path = "decoder-wikipedia-128.pth"
        else:
            decoder_path = "decoder-wikipedia-128.pth"
        self.decoder.load_state_dict(torch.load(os.path.join(self.checkpoints_folder, decoder_path))["decoder"])

    def switch_to_ema(self) -> None:
        ema = self.ema
        score_model = self.score_estimator
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())

    def switch_back_from_ema(self) -> None:
        ema = self.ema
        score_model = self.score_estimator
        ema.restore(score_model.parameters())

    def set_optimizer(self) -> None:
        optimizer = torch.optim.AdamW(
            self.score_estimator.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
            betas=(self.config.optim.beta_1, self.config.optim.beta_2),
            eps=self.config.optim.eps,
        )
        self.warmup = self.config.optim.linear_warmup
        self.grad_clip_norm = self.config.optim.grad_clip_norm
        self.optimizer = optimizer

    def set_scheduler(self) -> None:
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.config.training.training_iters,
            lr_min=self.config.optim.min_lr,
            warmup_lr_init=self.config.optim.warmup_lr,
            warmup_t=self.config.optim.linear_warmup,
            cycle_limit=1,
            t_in_epochs=False,
        )

    def set_grad_scaler(self) -> None:
        self.grad_scaler = GradScaler()

    def set_train_data_generator(self) -> None:
        num_tasks = 1
        del self.train_dataset
        self.train_dataset = next(self.train_datasets_iter)
        print("Dataset length:", len(self.train_dataset))

        if self.config.ddp:
            num_tasks = dist.get_world_size()
            global_rank = dist.get_rank()

            sampler_train = torch.utils.data.DistributedSampler(
                self.train_dataset,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=True,
            )
        else:
            sampler_train = None

        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=sampler_train,
            batch_size=self.config.training.batch_size // num_tasks,
            num_workers=80,
            pin_memory=False,
        )

    def set_valid_data_generator(self) -> None:
        num_tasks = 1

        if self.config.ddp:
            num_tasks = dist.get_world_size()
            sampler_valid = torch.utils.data.distributed.DistributedSampler(
                self.valid_dataset,
                shuffle=False
            )
        else:
            sampler_valid = None

        self.valid_loader = DataLoader(
            self.valid_dataset,
            sampler=sampler_valid,
            batch_size=self.config.validation.batch_size,
            num_workers=8,
            pin_memory=False,
        )

    def log_metric(self, metric_name: str, loader_name: str, value: Union[float, torch.Tensor, wandb.Image]):
        if dist.get_rank() == 0:
            wandb.log({f'{metric_name}/{loader_name}': value}, step=self.step)

    def optimizer_step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()

        self.grad_scaler.unscale_(self.optimizer)

        grad_norm = torch.sqrt(sum([torch.sum(t.grad ** 2) for t in self.score_estimator.parameters()]))

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.score_estimator.parameters(),
                max_norm=self.grad_clip_norm
            )

        clipped_grad_norm = torch.sqrt(sum([torch.sum(t.grad ** 2) for t in self.score_estimator.parameters()]))

        self.log_metric('lr', 'train', self.optimizer.param_groups[0]['lr'])
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.ema.update(self.score_estimator.parameters())
        self.scheduler.step_update(self.step)
        return grad_norm, clipped_grad_norm

    def sample_time(self, batch_size: int, eps: float = 1e-5):
        # return torch.rand(batch_size) * (0.05 - eps) + eps
        return torch.cuda.FloatTensor(batch_size).uniform_() * (self.sde.T - eps) + eps

    def get_dict_to_emb(self, X: Dict) -> Dict:
        keys = [
            'input_ids',
            'token_type_ids',
            'position_ids',
            'inputs_embeds',
            'past_key_values_length'
        ]
        res = dict()
        for k in keys:
            if k in X:
                res[k] = X[k]
        return res

    def bert_acc(self, targets, outputs, mask):
        pred_tokens = outputs.argmax(dim=-1)

        mask = deepcopy(mask)
        mask.scatter_(dim=1, index=(mask.sum(dim=1) - 1).reshape(-1, 1), src=torch.zeros_like(mask))
        mask[:, 0] = 0
        return torch.sum(mask * (targets == pred_tokens)) / torch.sum(mask)

    def mse_loss(self, inputs, targets, mask):
        losses = torch.mean(torch.square(inputs - targets), dim=-1)
        losses = losses * mask
        loss = torch.sum(losses) / torch.sum(mask)
        return loss

    def recon_loss(self, inputs, outputs, mask):
        losses = cross_entropy(
            input=inputs.reshape(-1, inputs.shape[-1]),
            target=outputs.reshape(-1),
            reduce=False,
        )
        losses = losses * mask.reshape(-1)
        loss = torch.sum(losses) / torch.sum(mask)
        return loss

    def get_stat(self, z, mask):
        mask_SEP_CLS = make_mask_wo_SEP_CLS(mask)
        mean = masked_mean(z, mask_SEP_CLS)
        std = masked_std(z, mask_SEP_CLS)
        norm = torch.sum(torch.norm(z, dim=2) * mask_SEP_CLS) / torch.sum(mask_SEP_CLS)
        return torch.mean(mean), torch.mean(std), norm

    def calc_loss(
            self,
            clean_x,
            cond=None,
            X=None,
            eps: float = 1e-5,
    ) -> Dict[str, torch.Tensor]:
        # Noizing
        batch_size = clean_x.size(0)

        t = self.sample_time(batch_size, eps=eps)
        marg_forward = self.sde.marginal_forward(clean_x, t)
        x_t, noise, score_clean = marg_forward['x_t'], marg_forward['noise'], marg_forward['score']

        # bert_model prediction
        scores = self.sde.calc_score(self.ddp_score_estimator, x_t, t, cond=cond, attention_mask=X["input_mask"],
                                     cond_mask=X["cond_mask"])

        # MSE losses
        x_0, eps_theta, score = scores["x_0"], scores['eps_theta'], scores["score"]
        mask = X["input_mask"]

        loss_x_0 = self.mse_loss(clean_x, x_0, mask)
        loss_eps = self.mse_loss(noise, eps_theta, mask)
        loss_score = self.mse_loss(score_clean, score, mask)

        # Decoder reconstruction

        logits = self.pred_logits(pred_embeddings=x_0, input_ids=X["input_ids"])
        ce_loss = self.recon_loss(logits, X["input_ids"], mask)

        # Statistics

        if self.config.model.loss == "L_x_0":
            loss = loss_x_0
        elif self.config.model.loss == "L_eps":
            loss = loss_eps
        elif self.config.model.loss == "L_score":
            loss = loss_score
        loss = loss + ce_loss * self.config.loss.ce_coef
        loss_dict = {
            'loss': loss,
            'total_loss': loss,
            'loss_eps': loss_eps,
            'loss_x_0': loss_x_0,
            'loss_score': loss_score,
            'loss_ce': ce_loss,
            'accuracy': self.bert_acc(targets=X["input_ids"], outputs=logits, mask=mask)
        }

        clean_x_mean, clean_x_std, clean_x_norm = self.get_stat(clean_x, mask)
        x_0_mean, x_0_std, x_0_norm = self.get_stat(x_0, mask)
        stat_dict = {
            "clean_x_mean": clean_x_mean,
            "clean_x_std": clean_x_std,
            "clean_x_norm": clean_x_norm,
            "x_0_mean": x_0_mean,
            "x_0_std": x_0_std,
            "x_0_norm": x_0_norm,
        }
        return loss_dict, stat_dict

    def train(
            self,
            project_name: str = 'bert_diffusion',
            experiment_name: str = 'bert_emb'
    ) -> None:
        if dist.get_rank() == 0:
            wandb.init(project=project_name, name=experiment_name, config=dict(self.config), mode="offline")

        self.tracker = Loss_ema_tracker()
        self.set_optimizer()
        self.set_scheduler()
        self.set_grad_scaler()

        self.step = 0
        self.ema = ExponentialMovingAverage(self.score_estimator.parameters(), self.config.model.ema_rate)
        self.refresh_checkpoint()

        self.train_range = trange(self.step + 1, self.config.training.training_iters + 1)
        self.train_range_iter = iter(self.train_range)

        self.set_valid_data_generator()

        self.file = open("log.txt", "w")

        while True:
            self.set_train_data_generator()
            self.ddp_score_estimator.train()
            self.train_epoch()

            if self.step >= self.config.training.training_iters:
                break

        self.score_estimator.eval()
        self.save_checkpoint(last=True)
        self.switch_to_ema()

    def train_epoch(self):
        for _, X in enumerate(self.train_loader):
            if self.step >= self.config.training.training_iters:
                return
            _ = next(self.train_range_iter)

            loss_dict, stat_dict = self.train_step(X)
            loss = loss_dict["loss"]

            if self.step % self.config.training.checkpoint_freq == 0:
                self.save_checkpoint()
                #self.estimate()

            if self.step % self.config.training.eval_freq == 0:
                self.validate()
                # self.compute_restoration_loss(suffix="train")
                # self.compute_restoration_loss(suffix="valid")

            self.tracker.update(loss.item())
            self.train_range.set_description(
                f"loss: {self.tracker.loss:0.4f}, "
                f"loss_eps: {loss_dict['loss_eps'].item():0.4f}, "
                f"loss_x_0: {loss_dict['loss_x_0'].item():0.4f}, "
                f"grad_norm: {stat_dict['grad_norm'].item():0.4f}, "
                f"accuracy: {loss_dict['accuracy'].item():0.4f}"
            )

        # torch.cuda.synchronize()

    def train_step(self, X):
        self.step += 1

        X = dict_to_cuda(X)
        clean_X = self.sampler_emb({"input_ids": X["input_ids"], "attention_mask": X["input_mask"]})
        cond = self.sampler_emb({"input_ids": X["cond_ids"], "attention_mask": X["cond_mask"]})

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss_dict, stat_dict = self.calc_loss(clean_x=clean_X, cond=cond, X=X)

        stat_dict["grad_norm"], stat_dict["clipped_grad_norm"] = self.optimizer_step(loss_dict['total_loss'])

        if self.step % 10 == 0:
            stat_dict["weight_norm"] = torch.sqrt(
                sum([torch.sum(t.data ** 2) for t in self.score_estimator.parameters()]))

            for k, v in loss_dict.items():
                self.log_metric(k, 'train', v.item())

            for k, v in stat_dict.items():
                self.log_metric(k, 'train', v.item())

        return loss_dict, stat_dict

    def validate(self) -> None:
        prev_mode = self.ddp_score_estimator.training

        self.ddp_score_estimator.eval()
        self.switch_to_ema()

        valid_loss: Dict[str, torch.Tensor] = dict()
        valid_count = torch.Tensor([0.0])

        with torch.no_grad():
            for text in self.valid_loader:
                X = text
                X = dict_to_cuda(X)
                clean_X = self.sampler_emb({"input_ids": X["input_ids"], "attention_mask": X["input_mask"]})
                cond = self.sampler_emb({"input_ids": X["cond_ids"], "attention_mask": X["cond_mask"]})

                loss_dict, _ = self.calc_loss(clean_x=clean_X, cond=cond, X=X)
                for k, v in loss_dict.items():
                    if k in valid_loss:
                        valid_loss[k] += v.item() * clean_X.size(0)
                    else:
                        valid_loss[k] = torch.Tensor([v.item() * clean_X.size(0)])
                valid_count += clean_X.size(0)

        valid_count = reduce_tensor(valid_count.cuda())
        for k, v in valid_loss.items():
            valid_loss[k] = reduce_tensor(valid_loss[k].cuda())

        for k, v in valid_loss.items():
            valid_loss[k] = v / valid_count
        for k, v in valid_loss.items():
            self.log_metric(k, 'valid_loader', v)

        self.switch_back_from_ema()
        self.ddp_score_estimator.train(prev_mode)

    def save_checkpoint(self, last: bool = False) -> None:
        if dist.get_rank() == 0:
            if not os.path.exists(self.checkpoints_folder):
                os.makedirs(self.checkpoints_folder)

            prefix = ''
            if self.config.checkpoints_prefix:
                prefix = self.config.checkpoints_prefix + '_'
            if last:
                prefix = prefix + 'last_'
            else:
                prefix = prefix + str(self.step) + '_'

            torch.save(
                {
                    "bert_model": self.score_estimator.state_dict(),
                    "ema": self.ema.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "step": self.step,
                },
                os.path.join(self.checkpoints_folder, prefix + ".pth")
            )
            print(f"Save bert_model to: {os.path.join(self.checkpoints_folder, prefix + f'bert_model.pth')}")

    def refresh_checkpoint(self):
        if not self.config.refresh.true:
            return
        load = torch.load(f'{self.config.refresh.prefix}', map_location="cpu")
        self.score_estimator.load_state_dict(load["bert_model"])
        self.score_estimator.cuda()
        self.optimizer.load_state_dict(load["optimizer"])
        self.scheduler.load_state_dict(load["scheduler"])
        self.step = load["step"]
        print(f"Checkpoint refreshed {self.config.refresh.prefix}")

    def generate_text(self, batch_size, cond=None, way="sde"):
        cond = dict_to_cuda(cond)
        cond_X = self.sampler_emb({"input_ids": cond["cond"], "attention_mask": cond["cond_mask"]})

        if way == "sde":
            pred_embeddings = self.pred_embeddings(batch_size, cond=cond_X, cond_mask=cond["cond_mask"])
        elif way == "ddpm":
            pred_embeddings = self.pred_embeddings_DDPM(batch_size)
        elif way == "ddim":
            pred_embeddings = self.pred_embeddings_DDIM(batch_size)
        else:
            raise Exception("way of sampling doesn't exist")
        # pred_embeddings = normalize(pred_embeddings, dim=-1) * np.sqrt(pred_embeddings.shape[-1])
        output = self.pred_logits(pred_embeddings)
        tokens = output.argmax(dim=-1)
        text = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        return text, pred_embeddings

    def pred_logits(self, pred_embeddings, input_ids=None):
        pred_embeddings = self.denormalize(pred_embeddings)
        if self.latent_mode == "embeddings":
            output = self.encoder(input_ids=input_ids, pred_embeddings=pred_embeddings).logits
        elif self.latent_mode == "encodings":
            output = self.decoder(
                pred_embeddings)  # self.encoder(input_ids=input_ids, pred_encodings=pred_embeddings).logits
        elif self.latent_mode == "mid_embeddings":
            output = self.encoder(input_ids=input_ids, pred_mid_embs=pred_embeddings).logits
        else:
            raise NotImplemented()
        return output

    @torch.no_grad()
    def pred_embeddings(
            self, batch_size: int,
            eps: float = 1e-5,
            **kwargs,
    ) -> torch.Tensor:
        shape = (
            batch_size,
            self.config.data.max_sequence_len,
            self.encoder.config.hidden_size
        )
        with torch.no_grad():
            x = self.sde.prior_sampling(shape).to(self.device)
            eps_t = 3 / self.diff_eq_solver.sde.N
            timesteps = torch.linspace(self.sde.T, eps_t, self.sde.N, device=self.device)
            for i in tqdm(range(self.sde.N)):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                output = self.diff_eq_solver.step(model=self.score_estimator, x_t=x, t=vec_t, **kwargs)
                x, x_mean = output["x"], output["x_mean"]
            pred_embeddings = x_mean

        return pred_embeddings

    # def pred_embeddings_DDPM(self, batch_size):
    #     def q_x_t_rev(x_t, x_0, t):
    #         dt = 1 / self.diff_eq_solver.sde.N
    #         alpha_t = self.sde.scheduler.alpha_std(t)[0] ** 2
    #         alpha_t_1 = self.sde.scheduler.alpha_std(t - dt)[0] ** 2
    #         beta_t = self.sde.scheduler.beta_t(t)[:, None, None] * dt
    #
    #         mu = torch.sqrt(alpha_t_1) * beta_t / (1 - alpha_t) * x_0 + \
    #              torch.sqrt(1 - beta_t) * (1 - alpha_t_1) / (1 - alpha_t) * x_t
    #         std = torch.sqrt((1 - alpha_t_1) / (1 - alpha_t) * beta_t)
    #         return mu, std
    #
    #     shape = (
    #         batch_size,
    #         self.config.data.max_sequence_len,
    #         self.encoder.config.hidden_size
    #     )
    #     with torch.no_grad():
    #         x_t = self.sde.prior_sampling(shape).to(self.device)
    #         n = 4
    #         eps_t = n / self.diff_eq_solver.sde.N
    #         timesteps = torch.linspace(self.sde.T, eps_t, self.sde.N - n + 1, device=self.device)
    #         for i in tqdm(range(self.sde.N - n + 1)):
    #             t = timesteps[i]
    #             vec_t = torch.ones(shape[0], device=t.device) * t
    #             # print(f"{t:0.3f}: {torch.mean(torch.norm(x_t, dim=-1)):0.3f}")
    #
    #             scores = self.sde.calc_score(self.score_estimator, x_t, vec_t)
    #             x_0 = scores.pop("x_0")
    #             mu, std = q_x_t_rev(x_t, x_0, vec_t)
    #             x_t = mu + std * torch.randn_like(x_t)
    #
    #         pred_embeddings = mu
    #     return pred_embeddings

    # def pred_embeddings_DDIM(self, batch_size):
    #     def q_x_t_rev(x_t, x_0, t, sigma_t):
    #         dt = 1 / self.diff_eq_solver.sde.N
    #         alpha_t = self.sde.scheduler.alpha_std(t)[0] ** 2
    #         alpha_t_1 = self.sde.scheduler.alpha_std(t - dt)[0] ** 2
    #
    #         sigma_t = torch.sqrt((1 - alpha_t_1) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_1) * 0.1
    #
    #         noise_t = (x_t - torch.sqrt(alpha_t) * x_0) / torch.sqrt(1 - alpha_t)
    #         mu = torch.sqrt(alpha_t_1) * x_0 + \
    #              torch.sqrt(1 - alpha_t_1 - sigma_t ** 2) * noise_t
    #         std = sigma_t
    #         return mu, std
    #
    #     sigma_t = 0
    #     shape = (
    #         batch_size,
    #         self.config.data.max_sequence_len,
    #         self.encoder.config.hidden_size
    #     )
    #     with torch.no_grad():
    #         x_t = self.sde.prior_sampling(shape).to(self.device)
    #         n = 50
    #         eps_t = n / self.diff_eq_solver.sde.N
    #         timesteps = torch.linspace(self.sde.T, eps_t, self.sde.N - n + 1, device=self.device)
    #         for i in tqdm(range(self.sde.N - n + 1)):
    #             t = timesteps[i]
    #
    #             # sigma_t = torch.sqrt(30 * t ** 2 / 1000)
    #             vec_t = torch.ones(shape[0], device=t.device) * t
    #             # print(f"{t:0.3f}: {torch.mean(torch.norm(x_t, dim=-1)):0.3f}")
    #
    #             scores = self.sde.calc_score(self.score_estimator, x_t, vec_t)
    #             x_0 = scores.pop("x_0")
    #             mu, std = q_x_t_rev(x_t, x_0, vec_t, sigma_t)
    #             x_t = mu + std * torch.randn_like(x_t)
    #
    #         pred_embeddings = mu
    #     return pred_embeddings

    @torch.no_grad()
    def restore_text(self, X, mask):
        X = dict_to_cuda(X)
        mask = mask.cuda()
        # mask = deepcopy(X["attention_mask"])
        # mask.scatter_(dim=1, index=(mask.sum(dim=1) - 1).reshape(-1, 1), src=torch.zeros_like(mask))
        # X["attention_mask"] = mask

        clean_X = self.sampler_emb(dict({key: X[key] for key in ["input_ids", "attention_mask"]}))
        mask = mask[:, :, None]
        clean_X = clean_X * mask
        pred_embeddings = self.restore_embeddings(mask.shape[0], clean_X, mask)
        tokens = self.pred_logits(pred_embeddings, X["input_ids"]).argmax(dim=-1)
        text = self.tokenizer.batch_decode(tokens)
        return text, tokens

    @torch.no_grad()
    def restore_embeddings(
            self, batch_size: int,
            masked_x, mask,
            eta: float = 0.2,
            verbose: bool = True
    ) -> torch.Tensor:
        shape = (
            batch_size,
            self.config.data.max_sequence_len,
            768
        )
        import numpy as np
        eta_value = np.sqrt(1 - eta * eta)
        device = torch.device(self.config.device)
        eps_t = 1 / self.diff_eq_solver.sde.N
        gamma = 1
        dt = -1. / self.sde.N

        with torch.no_grad():
            x = x_mean = self.sde.prior_sampling(shape).to(device)

            timesteps = torch.linspace(self.sde.T, eps_t, self.sde.N, device=self.device)
            rang = tqdm(range(self.sde.N))
            #
            # for idx in rang:
            #     t = timesteps[idx]
            #     input_t = t * torch.ones(shape[0], device=device)
            #
            #     y_t = self.sde.marginal_forward(masked_x, input_t)['x_t']
            #
            #     t = input_t
            #     x_t = x
            #
            #     z = torch.randn_like(x_t)
            #     rsde_params = self.diff_eq_solver.rsde.sde(self.score_estimator, x_t, t)
            #     drift, dif = rsde_params['drift'], rsde_params['diffusion']
            #
            #     x_mean = gamma * mask * y_t + (1 - gamma) * mask * x_t + (1 - mask) * x_t
            #     x_mean = x_mean + drift * dt
            #
            #     if not self.diff_eq_solver.ode_sampling:
            #         x = x_mean + dif[:, None, None] * np.sqrt(-dt) * z
            #     else:
            #         x = x_mean
            # x = normalize(x, dim=-1) * np.sqrt(x.shape[-1])

            for idx in rang:
                t = timesteps[idx]
                vec_t = t * torch.ones(shape[0], device=device)

                y_t = self.sde.marginal_forward(masked_x, vec_t)['x_t']
                x_t = x
                input_t = gamma * mask * y_t + (1 - gamma) * mask * x_t + (1 - mask) * x_t
                output = self.diff_eq_solver.step(model=self.score_estimator, x_t=input_t, t=vec_t)
                x, x_mean = output["x"], output["x_mean"]
                # x = normalize(x, dim=-1) * np.sqrt(x.shape[-1])

        return x_mean

    # def generate_latent(self, embs, mask):
    #     def get_x_t_next(x_t, vec_t, mask):
    #         params = self.sde.sde(x_t, vec_t)
    #         drift, dif = params["drift"], params["diffusion"]
    #         dt = 1. / self.diff_eq_solver.rsde.N
    #         if torch.allclose(t, torch.zeros_like(t)):
    #             fn = drift
    #         else:
    #             score = self.sde.calc_score(self.score_estimator, x_t, vec_t, mask)["score"]
    #             fn = drift - 0.5 * dif[:, None, None] ** 2 * score
    #         x_t_next = x_t + fn * dt
    #         return x_t_next
    #
    #     norm_xt = []
    #     batch_size = embs.shape[0]
    #     x_t = embs
    #     timesteps = torch.linspace(0, self.sde.T, self.sde.N + 1, device=self.device)
    #
    #     with torch.no_grad():
    #         for t in tqdm(timesteps):
    #             vec_t = torch.ones(batch_size, device=t.device) * t
    #             x_t = get_x_t_next(x_t, vec_t, mask)
    #             norm_xt.append(torch.mean(torch.norm(x_t, dim=[2])).item())
    #
    #     return x_t, norm_xt

    # def compute_likelihood(self, embs, mask):
    #     def get_x_t_next_drift(x_t, vec_t, mask):
    #         params = self.sde.sde(x_t, vec_t)
    #         drift, dif = params["drift"], params["diffusion"]
    #         if torch.allclose(t, torch.zeros_like(t)):
    #             fn = drift
    #         else:
    #             score = self.sde.calc_score(self.score_estimator, x_t, vec_t, mask)["score"]
    #             fn = drift - 0.5 * dif[:, None, None] ** 2 * score
    #         return fn
    #
    #     def get_x_t_next(x_t, vec_t, mask):
    #         params = self.sde.sde(x_t, vec_t)
    #         drift, dif = params["drift"], params["diffusion"]
    #         dt = 1. / self.diff_eq_solver.rsde.N
    #         if torch.allclose(t, torch.zeros_like(t)):
    #             fn = drift
    #         else:
    #             score = self.sde.calc_score(self.score_estimator, x_t, vec_t, mask)["score"]
    #             fn = drift - 0.5 * dif[:, None, None] ** 2 * score
    #         x_t_next = x_t + fn * dt
    #         return x_t_next
    #
    #     def compute_trace(x_t, t, mask):
    #         eps = torch.randn_like(x_t)
    #         with torch.enable_grad():
    #             x_t.requires_grad_(True)
    #             fn = get_x_t_next_drift(x_t, t, mask)
    #             fn_eps = torch.sum(fn * eps)
    #             grad_fn_eps = torch.autograd.grad(fn_eps, x_t)[0]
    #         x_t.requires_grad_(False)
    #         trace = torch.sum(grad_fn_eps * eps, dim=[1, 2])
    #         return trace
    #
    #     def prior_logp(z):
    #         shape = z.shape
    #         N = np.prod(shape[1:])
    #         return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2)) / 2.
    #
    #     x_t = embs
    #     batch_size = embs.shape[0]
    #     dt = 1 / self.diff_eq_solver.sde.N
    #     timesteps = torch.linspace(0, self.sde.T, self.sde.N + 1, device=self.device)
    #     probability = torch.zeros(batch_size).to("cuda:0")
    #
    #     with torch.no_grad():
    #         for t in tqdm(timesteps):
    #             vec_t = torch.ones(batch_size, device=t.device) * t
    #             proba = compute_trace(x_t, vec_t, mask) * dt
    #             probability += proba
    #             x_t = get_x_t_next(x_t, vec_t, mask)
    #
    #     probability = probability + prior_logp(x_t)
    #     return probability, x_t

    @torch.no_grad()
    def compute_restoration_loss(self, suffix="valid"):
        if dist.get_rank() != 0:
            return

        self.score_estimator.eval()
        self.switch_to_ema()

        if suffix == "train":
            X = next(iter(self.train_loader))
        elif suffix == "valid":
            X = next(iter(self.valid_loader))
        X = dict_to_cuda(X)
        clean_X = self.sampler_emb({"input_ids": X["input_ids"], "attention_mask": X["input_mask"]})
        cond = self.sampler_emb({"input_ids": X["cond_ids"], "attention_mask": X["cond_mask"]})
        batch_size = clean_X.shape[0]
        mask = X["input_mask"]

        losses_x_0 = []
        losses_eps = []
        losses_score = []
        losses_ce = []

        for t in range(0, self.diff_eq_solver.sde.N):
            t = t * 1. / self.diff_eq_solver.sde.N
            vec_t = t * torch.ones(batch_size, device=self.device)
            marg_forward = self.sde.marginal_forward(clean_X, vec_t)
            x_t = marg_forward['x_t']
            noise, score_clean = marg_forward['noise'], marg_forward['score']

            scores = self.sde.calc_score(self.score_estimator, x_t=x_t, t=vec_t, cond=cond,
                                         attention_mask=X["input_mask"], cond_mask=X["cond_mask"])
            x_0, eps_theta, score = scores["x_0"], scores['eps_theta'], scores["score"]

            loss_x_0 = self.mse_loss(clean_X, x_0, mask)
            loss_eps = self.mse_loss(noise, eps_theta, mask)
            loss_score = self.mse_loss(score_clean, score, mask)
            loss_ce = self.recon_loss(self.pred_logits(pred_embeddings=x_0), X["input_ids"], mask)

            losses_x_0.append(loss_x_0.item())
            losses_eps.append(loss_eps.item())
            losses_score.append(loss_score.item())
            losses_ce.append(loss_ce.item())

        self.score_estimator.train()
        self.switch_back_from_ema()

        timesteps = np.arange(0., 1., 1. / self.diff_eq_solver.sde.N)
        keys = ["losses_x_0", "losses_eps", "losses_score", "losses_ce"]
        for i, loss in enumerate([losses_x_0, losses_eps, losses_score, losses_ce]):
            data = [[x, y] for (x, y) in zip(timesteps, loss)]
            table = wandb.Table(data=data, columns=["time", "loss"])
            wandb.log({f"{suffix}-{keys[i]}": wandb.plot.line(table, "time", "loss", title=f"{suffix}-{keys[i]}")},
                      step=self.step)

    @torch.no_grad()
    def estimate(self):
        self.score_estimator.eval()
        self.switch_to_ema()

        if not hasattr(self, 'metric_bloom_fn'):
            self.metric_bloom_fn = BloomMetric(device=f"cuda:{dist.get_rank()}")

        self.metric_bloom_fn.model.cuda()

        num_texts = int(self.config.validation.num_gen_texts / dist.get_world_size())
        seed = self.config.seed + dist.get_rank()
        set_seed(seed)
        metrics, texts = estimate_model(self, num_texts, self.config.validation.batch_size,
                                        self.metric_bloom_fn, None, type_="cond")
        texts = gather_texts(texts)
        metrics = reduce_metrics(metrics)
        if dist.get_rank() == 0:
            texts_path = "./generated_texts"
            print(f"Bloom metric: {metrics['Bloom metric']:0.5f}")
            with open(f"{texts_path}/{self.config.checkpoints_prefix}-{self.step}.txt", "w") as file:
                for text in texts:
                    print(text, file=file)

        self.log_metric(metric_name="bloom loss", loader_name="", value=metrics['Bloom metric'])

        self.metric_bloom_fn.model.cpu()
        self.score_estimator.train()
        self.switch_back_from_ema()
